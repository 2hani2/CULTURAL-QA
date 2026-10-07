"""
train_moe.py
------------
Trains the MoE fusion layer (moe_fusion.py) using real supervision derived
from ALL SIX datasets' own structure -- no separate manually labeled
relevance dataset needed to get started.

Two of the six (DRISHTIKON, Durga QA) already have a natural question
field, so their anchor text is just that field, same as before. The other
four (DOSA, SCALE, DIWALI, Indian Cultural Dataset) don't ask anything --
they're a riddle-style clue/artifact pair, a glossary, a concept list, and
a labeled-text corpus respectively -- so a natural-language query is
synthesized from whatever fields each one does have (see
get_anchor_texts / DATASET_ANCHOR_BUILDERS below). Positive text is always
the record's own record_to_text() -- exactly what gets indexed as a
retrievable chunk -- same idea as before, just applied to all 6 instead
of 1.

DRISHTIKON alone is ~64K of the ~79K total raw records (82%) -- pooling
everything uncapped would just re-teach the gate to expect DRISHTIKON's
shape and starve it of the other five almost entirely. So loading now
caps PER DATASET rather than as one global total:
--max_examples now means "up to N records FROM EACH of the 6 datasets",
not "N records total from DRISHTIKON" like before. At the same
--max_examples value, a run now produces up to ~6x more triplets than it
used to (Durga QA yields up to 4 anchors/record on top of that, since it
has 4 paraphrased questions per answer).

Negatives are sampled from the SAME dataset as the anchor, not pooled
globally -- the actual retrieval pipeline builds per-dataset-scoped
retrievers (see main.py), so a same-dataset negative (e.g. a different
SCALE item) is the realistic hard case the gate needs to learn to reject;
a cross-dataset negative (e.g. a Durga QA answer for a SCALE query) is
too easy to be a useful training signal.

This only trains MoEFusion's parameters (gate + experts + input_proj).
BERT and CLIP stay completely frozen -- this teaches the fusion/routing
layer to tell relevant (query, chunk) pairs from irrelevant ones, not
fine-tuning the underlying encoders themselves.

Usage:
  python train_moe.py                                    # defaults, text-only, all 6 datasets
  python train_moe.py --max_examples 5000 --epochs 5      # cap is now PER DATASET
  python train_moe.py --enable_images                     # also use real DRISHTIKON
                                                             images (the only dataset with
                                                             real images -- slower first run,
                                                             cached after -- see encoders.py)

After training, main.py auto-detects the checkpoint at ./checkpoints/moe_fusion.pt
and switches MoE re-scoring on automatically -- no other code changes needed.
"""

import argparse
import random
import os

import torch
import torch.nn.functional as F
from torch.optim import Adam

from structured_store import load_all_raw_records, record_to_text
from encoders import MultimodalEncoder
from moe_fusion import MoEFusion

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "moe_fusion.pt")

ALL_DATASETS = ["DOSA", "SCALE", "DIWALI", "DRISHTIKON", "Indian Cultural Dataset", "Durga QA"]


# ---------------------------------------------------------------------------
# Dataset-aware anchor (query) text. DRISHTIKON and Durga QA already ask
# something -- use that directly. The other four don't, so a
# natural-language query is built from their real, confirmed fields
# (structured_store.py's schema comments/docstrings) instead of a made-up
# generic field name.
# ---------------------------------------------------------------------------
def _drishtikon_anchors(r):
    q = (r.get("question") or "").strip()
    return [q] if q else []


def _durga_anchors(r):
    # up to 4 paraphrased questions per record -- each is a valid anchor
    # for the same positive/answer, so each becomes its own triplet
    return [q.strip() for q in (r.get("q1"), r.get("q2"), r.get("q3"), r.get("q4")) if q and str(q).strip()]


def _dosa_anchors(r):
    # DOSA is riddle-shaped: "clues" describes something, "artifact" names
    # it. The natural query IS the clue, reformulated as a question --
    # same pattern as DRISHTIKON's anchor overlapping with its own
    # positive text by design.
    clues = (r.get("clues") or "").strip().replace("\n", " ")
    if not clues:
        return []
    return [f"What DOSA artifact or cultural practice matches this description: {clues}"]


def _scale_anchors(r):
    item = (r.get("Item") or "").strip()
    if not item:
        return []
    domain = (r.get("domain") or "").strip()
    if domain:
        return [f"What does '{item}' mean in the context of {domain}?"]
    return [f"What does '{item}' mean?"]


def _diwali_anchors(r):
    concept = (r.get("concept") or "").strip()
    if not concept:
        return []
    state = (r.get("state") or "").strip()
    if state:
        return [f"What is {concept} in {state}?"]
    return [f"What is {concept}?"]


def _indian_cultural_anchors(r):
    category = (r.get("category") or "").strip()
    if not category:
        return []
    return [f"Tell me about {category} in Indian culture"]


DATASET_ANCHOR_BUILDERS = {
    "DRISHTIKON": _drishtikon_anchors,
    "Durga QA": _durga_anchors,
    "DOSA": _dosa_anchors,
    "SCALE": _scale_anchors,
    "DIWALI": _diwali_anchors,
    "Indian Cultural Dataset": _indian_cultural_anchors,
}


def get_anchor_texts(record: dict):
    """Returns 0+ natural-language query strings for a record. Durga QA
    can yield up to 4 (one per paraphrase); everything else yields 0 or 1."""
    builder = DATASET_ANCHOR_BUILDERS.get(record.get("dataset"))
    if builder is None:
        return []
    return builder(record)


# BERT truncates to max_length=256 tokens internally anyway (roughly the
# first ~1500 characters of English text), so capping the text BEFORE it
# reaches the tokenizer doesn't lose any signal BERT would actually use --
# but it avoids a real, confirmed slowdown: a fast (Rust-backed) tokenizer
# still fully pre-tokenizes the ENTIRE input before truncation trims the
# token list, so a single ~17M-character string took ~8.6 SECONDS to
# tokenize even with truncation=True (tested directly: a normal-length
# string took 0.0002s). Indian Cultural Dataset rows run up to ~15.4M
# characters -- structured_store.py already caps this for the search
# index (LARGE_ROW_CHAR_THRESHOLD), but that path isn't used here, so
# without this cap, encoding could silently stall for a long stretch with
# zero progress output any time an oversized record gets pulled in as a
# positive OR as a cross-dataset negative for some other dataset's anchor.
TRAINING_TEXT_CHAR_CAP = 2000


def _capped_text(record: dict) -> str:
    text = record_to_text(record)
    return text[:TRAINING_TEXT_CHAR_CAP] if text else text


def load_multi_dataset_records(raw_dir: str = "./data/raw", max_examples_per_dataset: int = 5000):
    """Loads and caps records from ALL SIX datasets. Returns a dict of
    dataset_name -> list of records, keeping datasets separate so
    build_triplets can sample negatives from within the same dataset."""
    all_records = load_all_raw_records(raw_dir)

    by_dataset = {}
    for r in all_records:
        by_dataset.setdefault(r["dataset"], []).append(r)

    capped = {}
    print("[train_moe] per-dataset record counts (available -> used after cap):")
    for ds in ALL_DATASETS:
        records = list(by_dataset.get(ds, []))
        random.shuffle(records)
        available = len(records)
        if max_examples_per_dataset:
            records = records[:max_examples_per_dataset]
        capped[ds] = records
        print(f"[train_moe]   {ds}: {available} -> {len(records)}")

    total = sum(len(v) for v in capped.values())
    if total == 0:
        raise SystemExit(
            "No records found for any of the 6 datasets -- make sure ./data/raw/ has "
            "the real per-dataset folders (DOSA, SCALE, DIWALI, DRISHTIKON, "
            "'Indian Cultural Dataset', 'Durga QA') and you've run `python main.py --build` "
            "at least once."
        )
    print(f"[train_moe] {total} records total across "
          f"{sum(1 for v in capped.values() if v)}/{len(ALL_DATASETS)} datasets")
    return capped


def build_triplets(records_by_dataset: dict, num_negatives: int = 1, cross_dataset_negative_prob: float = 0.5):
    """Each (record, anchor) pair gives num_negatives triplets. Each
    negative is drawn either:
      - from the SAME dataset as the anchor ("hard" negative -- forces
        fine-grained discrimination, e.g. telling one SCALE item apart
        from another SCALE item), or
      - from the GLOBAL pool across all 6 datasets ("cross-dataset"
        negative -- teaches the fused similarity scores to be
        comparable/rankable ACROSS datasets)

    The cross-dataset negatives are the important addition: retrieval is
    meant to find the best match across all 6 datasets combined for a
    general question, not scoped to whichever dataset the anchor happens
    to belong to. Training with same-dataset negatives only (the previous
    version of this function) never taught the model to compare a SCALE
    record against a DRISHTIKON record at all -- cosine similarity scores
    from different datasets had no reason to be on the same scale.
    cross_dataset_negative_prob controls the mix; default 50/50."""
    all_records = [r for records in records_by_dataset.values() for r in records]
    n_total = len(all_records)

    triplets = []
    for dataset, records in records_by_dataset.items():
        n = len(records)
        if n == 0:
            continue
        dataset_triplet_count = 0
        cross_count = 0
        for i, r in enumerate(records):
            anchors = get_anchor_texts(r)
            if not anchors:
                continue
            positive = _capped_text(r)
            for anchor in anchors:
                for _ in range(num_negatives):
                    use_cross = (random.random() < cross_dataset_negative_prob) or n < 2
                    if use_cross and n_total >= 2:
                        neg_record = random.choice(all_records)
                        while neg_record is r:
                            neg_record = random.choice(all_records)
                        cross_count += 1
                    elif n >= 2:
                        j = random.randrange(n)
                        while j == i:
                            j = random.randrange(n)
                        neg_record = records[j]
                    else:
                        continue  # single-record dataset, no global pool either -- skip
                    negative = _capped_text(neg_record)
                    triplets.append((anchor, positive, negative, r))
                    dataset_triplet_count += 1
        print(f"[train_moe]   {dataset}: {dataset_triplet_count} triplets from {n} records "
              f"({cross_count} cross-dataset negatives, {dataset_triplet_count - cross_count} same-dataset)")
    random.shuffle(triplets)
    return triplets


def train(
    max_examples: int = 5000,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    margin: float = 0.2,
    enable_images: bool = False,
    raw_dir: str = "./data/raw",
    cross_dataset_negative_prob: float = 0.5,
):
    records_by_dataset = load_multi_dataset_records(raw_dir, max_examples_per_dataset=max_examples)

    print("[train_moe] building triplets (dataset-aware anchors, "
          f"{cross_dataset_negative_prob:.0%} cross-dataset negatives)...")
    triplets = build_triplets(records_by_dataset, num_negatives=1,
                               cross_dataset_negative_prob=cross_dataset_negative_prob)
    if not triplets:
        raise SystemExit(
            "No triplets could be built -- every dataset either had 0 usable records or "
            "had fewer than 2 records to sample a negative from."
        )
    print(f"[train_moe] {len(triplets)} training triplets total, {epochs} epochs, batch size {batch_size}")

    encoder = MultimodalEncoder(enable_live_fetch=enable_images)
    moe = MoEFusion()
    optimizer = Adam(moe.parameters(), lr=lr)

    # PRE-ENCODE all unique texts/images ONCE, in real batches, before
    # training starts. The original version of this script called
    # encoder.text_encoder.encode() one text at a time inside a Python
    # loop during every mini-batch -- BERT supports batched input
    # (padding + one forward pass for many texts at once), so encoding
    # texts one-by-one wastes most of that batching benefit and made a
    # 5000-example run look frozen for a long time with zero progress
    # output. This does it properly: collect every unique text/image
    # needed across the whole triplet set, encode them in batches of
    # `encode_batch_size` with a running progress counter, then the
    # training loop below is pure dictionary lookups -- fast.
    print("[train_moe] pre-encoding all unique texts (one-time cost, not repeated per epoch)...")
    unique_texts = sorted({t for triplet in triplets for t in triplet[:3]})
    text_emb_cache = {}
    encode_batch_size = 32
    for i in range(0, len(unique_texts), encode_batch_size):
        batch_texts = unique_texts[i:i + encode_batch_size]
        batch_embs = encoder.text_encoder.encode(batch_texts)
        for t, e in zip(batch_texts, batch_embs):
            text_emb_cache[t] = e
        done = min(i + encode_batch_size, len(unique_texts))
        if done % (encode_batch_size * 10) == 0 or done == len(unique_texts):
            print(f"[train_moe]   encoded {done}/{len(unique_texts)} unique texts")

    image_emb_cache = {}
    if enable_images:
        # only DRISHTIKON records carry a real image_url -- the other 5
        # datasets simply don't have this key, so this naturally only
        # ever picks up DRISHTIKON images without any dataset-specific
        # branching needed here
        unique_image_refs = sorted({
            r.get("image_url")
            for _, _, _, r in triplets
            if r.get("image_url")
        })
        print(f"[train_moe] pre-encoding {len(unique_image_refs)} unique images "
              f"(all DRISHTIKON -- the only dataset with real images; "
              f"cached to disk after this -- see encoders.py)...")
        for i, ref in enumerate(unique_image_refs, start=1):
            image_emb_cache[ref] = encoder.image_encoder.encode(ref)[0]
            if i % 50 == 0 or i == len(unique_image_refs):
                print(f"[train_moe]   encoded {i}/{len(unique_image_refs)} unique images")

    def embed_text(text):
        return text_emb_cache[text]

    def embed_image(ref):
        if not ref:
            return encoder.image_encoder.null_image_embedding[0]
        return image_emb_cache.get(ref, encoder.image_encoder.null_image_embedding[0])

    print("[train_moe] pre-encoding done -- starting training (this part is fast, pure lookups + small MoE forward/backward passes)")
    moe.train()
    for epoch in range(1, epochs + 1):
        random.shuffle(triplets)
        total_loss = 0.0
        total_triplet_loss = 0.0
        total_lb_loss = 0.0
        num_batches = 0

        for batch_start in range(0, len(triplets), batch_size):
            batch = triplets[batch_start:batch_start + batch_size]
            if not batch:
                continue

            anchor_text_embs = torch.stack([embed_text(a) for a, _, _, _ in batch])
            pos_text_embs = torch.stack([embed_text(p) for _, p, _, _ in batch])
            neg_text_embs = torch.stack([embed_text(n) for _, _, n, _ in batch])

            # queries are text-only (no image attached to a typed question),
            # regardless of which of the 6 datasets they came from
            anchor_img_embs = torch.zeros(len(batch), encoder.image_encoder.embed_dim)
            pos_img_embs = torch.stack(
                [embed_image(r.get("image_url")) for _, _, _, r in batch]
            )
            # approximation: negative images treated as absent rather than
            # fetching a second unrelated image per triplet -- keeps
            # training fast; the text signal already carries most of the
            # positive/negative distinction for this bootstrap dataset.
            neg_img_embs = torch.zeros(len(batch), encoder.image_encoder.embed_dim)

            anchor_fused = moe(anchor_text_embs, anchor_img_embs)
            anchor_lb_loss = moe.load_balancing_loss()
            pos_fused = moe(pos_text_embs, pos_img_embs)
            pos_lb_loss = moe.load_balancing_loss()
            neg_fused = moe(neg_text_embs, neg_img_embs)
            neg_lb_loss = moe.load_balancing_loss()

            pos_sim = F.cosine_similarity(anchor_fused, pos_fused, dim=-1)
            neg_sim = F.cosine_similarity(anchor_fused, neg_fused, dim=-1)

            # triplet hinge: push pos_sim above neg_sim by at least `margin`
            triplet_loss = F.relu(margin - pos_sim + neg_sim).mean()

            # load-balancing: discourage the gate from always routing to the
            # same 1-2 experts out of the 3 routed ones (attention/tfn/mult) --
            # small coefficient since this is a regularizer, not the main objective
            lb_loss = (anchor_lb_loss + pos_lb_loss + neg_lb_loss) / 3
            loss = triplet_loss + 0.01 * lb_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_triplet_loss += triplet_loss.item()
            total_lb_loss += lb_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_triplet_loss = total_triplet_loss / max(num_batches, 1)
        avg_lb_loss = total_lb_loss / max(num_batches, 1)
        print(f"[train_moe] epoch {epoch}/{epochs} -- avg loss {avg_loss:.4f} "
              f"(triplet {avg_triplet_loss:.4f}, load-balance {avg_lb_loss:.4f})")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    moe.save_checkpoint(CHECKPOINT_PATH)
    encoder.close()  # flush any cached image embeddings to disk
    print(f"[train_moe] done. Restart main.py -- it auto-detects and loads {CHECKPOINT_PATH}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_examples", type=int, default=5000,
                         help="How many records to use PER DATASET (applied to each of the "
                              "6 datasets separately, not as one global total -- DRISHTIKON "
                              "alone is ~64K/~79K records, so an uncapped pool would drown "
                              "out the other 5)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--enable_images", action="store_true",
                         help="Also fetch and encode real DRISHTIKON images -- the only one "
                              "of the 6 datasets with real images (cached to disk after first "
                              "fetch -- see encoders.py)")
    parser.add_argument("--raw_dir", type=str, default="./data/raw")
    parser.add_argument("--cross_dataset_negative_prob", type=float, default=0.5,
                         help="Fraction of negatives drawn from the GLOBAL pool across all 6 "
                              "datasets rather than the anchor's own dataset. Higher = better "
                              "calibrated for 'best match across all datasets combined' "
                              "ranking; lower = sharper within-dataset discrimination. 0.0 "
                              "reproduces the old same-dataset-only behavior.")
    args = parser.parse_args()

    train(
        max_examples=args.max_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        enable_images=args.enable_images,
        raw_dir=args.raw_dir,
        cross_dataset_negative_prob=args.cross_dataset_negative_prob,
    )