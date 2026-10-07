"""
encoders.py
-----------
Matches Fig. 1 / Section IV-C(4) of the CulturalQA paper: text encoded via
BERT/RoBERTa, images via CLIP/ViT. Generalized here to run across all 6
datasets -- image encoding only actually fires for chunks tagged
has_image=True by ingestion.py (DOSA, DRISHTIKON); the other four
datasets get a fixed learned "no-image" placeholder vector instead of a
real CLIP embedding, so every chunk still has a consistent (text_emb,
image_emb) pair going into the MoE fusion layer.

Image fetching is OFF by default (enable_live_fetch=False) -- flip it on
via MultimodalEncoder(enable_live_fetch=True) once you actually want real
visual signal. When on, results are cached to disk (both successes and
failures) so a given image URL is only ever fetched/encoded once, ever,
across all future runs -- not re-fetched over the network on every single
query the way it worked before this cache existed.

NEW: real image RETRIEVAL (not just encoding). Inspired by the RetVQA
paper's Relevance Encoder (Penamakuri et al., IJCAI 2023) -- given a
question, rank a pool of real images by actual relevance, rather than
passively encoding whatever image happens to be attached to a
text-retrieved record. RetVQA's own Relevance Encoder is a dedicated
jointly-attending transformer over Faster R-CNN region features; that's
more machinery than this project needs. The core transferable idea --
score a question against every image in the pool, keep the top-K -- is
implemented here using CLIP's OWN text encoder instead, since CLIP is
already in this stack and is contrastively pretrained specifically so its
text and image encoders share one embedding space. This was previously
unused: ImageEncoder only ever called CLIP's image side; CLIP's text side
sat completely idle while BERT handled all text for the MoE layer.
"""

import os
import hashlib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, CLIPModel, CLIPProcessor
from PIL import Image
import requests
from io import BytesIO

TEXT_MODEL_NAME = "bert-base-uncased"      # swap for "roberta-base" if preferred
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_CACHE_DIR = "./image_embedding_cache"
IMAGE_CACHE_FILE = os.path.join(IMAGE_CACHE_DIR, "image_embeddings.pt")


class TextEncoder:
    def __init__(self, model_name: str = TEXT_MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE).eval()

    @torch.no_grad()
    def encode(self, texts, max_length: int = 256):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(DEVICE)
        out = self.model(**inputs)
        # CLS-token pooled representation (standard for BERT-family encoders)
        cls_emb = out.last_hidden_state[:, 0, :]
        return cls_emb.cpu()  # (batch, 768)


class ImageEncoder:
    def __init__(self, model_name: str = CLIP_MODEL_NAME, enable_live_fetch: bool = False):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
        self.embed_dim = self.model.config.projection_dim  # 512 for ViT-B/32

        # Fixed learned placeholder for datasets/chunks with no image
        # (SCALE, DIWALI, Durga QA, Indian Cultural Dataset, and any
        # text-only rows within DOSA/DRISHTIKON). Zeros are used
        # rather than random noise so the MoE gate can learn to treat an
        # all-zero image vector as "no visual signal" cleanly.
        self.null_image_embedding = torch.zeros(1, self.embed_dim)

        # OFF by default -- flip on via MultimodalEncoder(enable_live_fetch=True).
        # Real caveat that stays true even with caching: DRISHTIKON's
        # image_link values were sourced via "controlled Google Image
        # Search" per its own report -- some point to a webpage ABOUT an
        # image rather than a direct image file, so PIL fails to decode
        # them ("cannot identify image file"). That failure gets cached
        # too, so it only costs one network round-trip ever, not one per
        # query -- but it's still a real, expected miss rate, not a bug.
        self.enable_live_fetch = enable_live_fetch
        self._cache = self._load_cache()
        self._cache_dirty = False

        # BUG FIX: this used to only save every 50 new entries, or when
        # .close() was called explicitly (which only ever happened in
        # train_moe.py). In normal interactive/chat use, a session with
        # under 50 image lookups never triggered a save at all -- meaning
        # every cache entry from every chat session was silently lost on
        # exit, confirmed concretely: after several image-fetch attempts
        # across multiple sessions, image_embedding_cache/image_embeddings.pt
        # didn't exist on disk at all. Now saves after every single new
        # entry (image fetches are already slow/rare enough that this
        # extra disk write is negligible), plus an atexit hook as a second
        # safety net in case the process exits some other way.
        if enable_live_fetch:
            import atexit
            atexit.register(self.save_cache)

    def _load_cache(self):
        if os.path.exists(IMAGE_CACHE_FILE):
            try:
                cache = torch.load(IMAGE_CACHE_FILE, weights_only=True)
                print(f"[encoders] loaded {len(cache)} cached image embeddings from {IMAGE_CACHE_FILE}")
                return cache
            except Exception as e:
                print(f"[encoders] could not load image cache ({e}), starting fresh")
        return {}

    def save_cache(self):
        if not self._cache_dirty:
            return
        os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
        torch.save(self._cache, IMAGE_CACHE_FILE)
        print(f"[encoders] saved {len(self._cache)} image embeddings to {IMAGE_CACHE_FILE}")
        self._cache_dirty = False

    @staticmethod
    def _cache_key(ref: str) -> str:
        return hashlib.sha1(ref.encode("utf-8")).hexdigest()

    def _load_image(self, ref: str):
        try:
            if ref.startswith("http"):
                resp = requests.get(ref, timeout=5)
                return Image.open(BytesIO(resp.content)).convert("RGB")
            return Image.open(ref).convert("RGB")
        except Exception as e:
            print(f"[image load failed] {ref}: {e}")
            return None

    @torch.no_grad()
    def encode(self, image_ref: str = None):
        if not image_ref or not self.enable_live_fetch:
            return self.null_image_embedding

        key = self._cache_key(image_ref)
        if key in self._cache:
            return self._cache[key]  # covers both prior successes AND prior failures

        img = self._load_image(image_ref)
        if img is None:
            result = self.null_image_embedding
        else:
            inputs = self.processor(images=img, return_tensors="pt").to(DEVICE)
            feats = self.model.get_image_features(**inputs)

            # BUG FIX: get_image_features is documented to return a plain
            # (batch, projection_dim) tensor, but on the installed
            # transformers version this instead returned a full
            # BaseModelOutputWithPooling object (confirmed via a real
            # training run: "AttributeError: 'BaseModelOutputWithPooling'
            # object has no attribute 'cpu'"). Rather than trust the
            # convenience wrapper's return type on any given version,
            # detect this and fall back to manually replicating what
            # get_image_features does internally in every stable release
            # (vision_model -> pooled output -> visual_projection), using
            # the lower-level components directly since those are far less
            # likely to have changed shape/type across versions.
            if not torch.is_tensor(feats):
                vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
                pooled_output = getattr(vision_outputs, "pooler_output", None)
                if pooled_output is None:
                    pooled_output = vision_outputs[1]  # (last_hidden_state, pooler_output, ...) tuple form
                feats = self.model.visual_projection(pooled_output)

            result = feats.cpu()  # (1, 512)

        self._cache[key] = result
        self._cache_dirty = True
        self.save_cache()  # save immediately -- see bug-fix note in __init__
        return result

    @torch.no_grad()
    def encode_text_for_retrieval(self, text: str):
        """CLIP's OWN text encoder -- distinct from TextEncoder (BERT)
        above, which is what feeds the MoE fusion layer. CLIP's text and
        image encoders are contrastively pretrained into ONE shared
        embedding space, so cosine similarity between this and encode()'s
        image output is directly meaningful -- unlike comparing a BERT
        text embedding against a CLIP image embedding, which live in
        unrelated spaces with no reason to be comparable at all. This is
        what makes real cross-modal image RETRIEVAL possible (see
        retrieve_relevant_images below), as opposed to only ever encoding
        whichever image happens to already be attached to a text-matched
        record."""
        inputs = self.processor(text=[text], return_tensors="pt",
                                 padding=True, truncation=True).to(DEVICE)
        text_features = self.model.get_text_features(**inputs)

        # Same version quirk already hit and fixed for get_image_features
        # above: documented to return a plain (batch, projection_dim)
        # tensor, but on this installed transformers version it instead
        # returns a full BaseModelOutputWithPooling object. Same fix:
        # detect it and manually replicate what get_text_features does
        # internally (text_model -> pooled output -> text_projection)
        # using the lower-level components directly.
        if not torch.is_tensor(text_features):
            text_outputs = self.model.text_model(**inputs)
            pooled_output = getattr(text_outputs, "pooler_output", None)
            if pooled_output is None:
                pooled_output = text_outputs[1]
            text_features = self.model.text_projection(pooled_output)

        return text_features.cpu()  # (1, 512) -- same space as encode()'s output


class MultimodalEncoder:
    """Wraps both encoders and produces one (text_emb, image_emb) pair per
    chunk, respecting each dataset's HAS_IMAGE_MODALITY flag from
    ingestion.py."""

    def __init__(self, enable_live_fetch: bool = False):
        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder(enable_live_fetch=enable_live_fetch)

    def encode_chunk(self, chunk):
        text_emb = self.text_encoder.encode(chunk.page_content)[0]  # (768,)

        image_ref = None
        if chunk.metadata.get("has_image"):
            image_ref = (
                chunk.metadata.get("image_url")
                or chunk.metadata.get("image_path")
                or chunk.metadata.get("images")
            )
        image_emb = self.image_encoder.encode(image_ref)[0]  # (512,) or null vector

        return text_emb, image_emb

    def retrieve_relevant_images(self, question: str, candidate_records: list, top_k: int = 3):
        """Real cross-modal image retrieval, the piece the pipeline didn't
        have before: given a question and a list of raw records that each
        carry a real image_url (in practice, currently only DRISHTIKON --
        see structured_store.py), ranks them by genuine CLIP text-image
        similarity and returns the top_k. This requires enable_live_fetch
        to have been turned on (real images must actually be encoded, not
        the null vector) -- returns an empty list otherwise rather than
        silently ranking null vectors against each other, which would be
        meaningless.

        This is deliberately independent of the main text retrieval path
        (retrieval.py) -- it answers a different question ("which image
        in this pool actually matches the question") rather than "which
        text record matches the question, with its image along for the
        ride." How to combine the two (e.g. as an additional candidate
        source feeding into MoE re-scoring) is a pipeline-integration
        decision worth making deliberately, not something this function
        assumes for you.
        """
        if not self.image_encoder.enable_live_fetch:
            print("[encoders] retrieve_relevant_images called but enable_live_fetch is off -- "
                  "returning no results rather than ranking meaningless null vectors")
            return []

        records_with_images = [r for r in candidate_records if r.get("image_url")]
        if not records_with_images:
            return []

        q_emb = self.image_encoder.encode_text_for_retrieval(question)  # (1, 512)

        scored = []
        for record in records_with_images:
            img_emb = self.image_encoder.encode(record["image_url"])  # (1, 512)
            score = F.cosine_similarity(q_emb, img_emb).item()
            scored.append((score, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def close(self):
        """Flush any pending image cache writes. Call this when you're done
        with a batch of encoding (e.g. at the end of training) so the last
        <50 entries since the previous auto-save aren't lost."""
        self.image_encoder.save_cache()