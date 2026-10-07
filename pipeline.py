"""
pipeline.py
-----------
End-to-end wiring, mirroring Fig. 1 top to bottom -- three answer paths:

  PATH A (exact lookup): "what's artifact #412 in DOSA?"
    -> query_router detects this is an ID question
    -> structured_store returns the exact row directly, no search needed
    -> deterministic, 100% accurate (if the record exists)

  PATH B (everything else factual): normal flow -- FAISS+BM25 hybrid ->
    MultiQuery expansion -> Cross-Encoder rerank -> MoE re-score ->
    Extractive reader

  PATH C (filter/list, e.g. "which artifacts relate to weddings"):
    same retrieval stack as Path B, but skips the single-span reader and
    returns the top matching records directly as a list.

Runs across all 6 datasets at once (dataset_filter=None, fair fan-out --
see retrieval.py), or restrict to one via --dataset.

force_moe_rescore (new): an explicit override for whether MoE re-scoring
runs, independent of whether a trained checkpoint happens to be loaded.
None (default) = auto-detect (on if a checkpoint loaded, per
self.moe_is_trained) -- unchanged prior behavior. True/False = always use
that setting regardless of checkpoint status. This is what lets
main_with_moe.py and main_without_moe.py exist as genuinely separate,
simple entry points without duplicating any pipeline logic -- they just
construct CulturalQAPipeline with this set one way or the other.
"""

import os
import json
import torch

from ingestion import load_all_documents
from chunking import chunk_documents
from retrieval import build_faiss_index, load_faiss_index, build_full_retriever, INDEX_DIR
from encoders import MultimodalEncoder
from moe_fusion import MoEFusion
from reader import ExtractiveReader
from structured_store import (
    load_all_raw_records,
    records_to_documents,
    build_record_index,
    format_record_answer,
)
from query_router import is_lookup_query, parse_lookup, detect_dataset, is_filter_query

CHUNK_CACHE = os.path.join(INDEX_DIR, "chunks_cache.json")
RAW_RECORDS_CACHE = os.path.join(INDEX_DIR, "raw_records_cache.json")


def build_index(data_dir: str = "./data"):
    report_docs = load_all_documents(data_dir)
    chunks = chunk_documents(report_docs) if report_docs else []

    raw_records = load_all_raw_records(os.path.join(data_dir, "raw"))
    raw_docs = records_to_documents(raw_records)

    all_index_docs = chunks + raw_docs
    if not all_index_docs:
        raise SystemExit(
            "Nothing to index -- add EDA report files to ./data and/or "
            "real dataset files to ./data/raw/<Dataset>/."
        )

    build_faiss_index(all_index_docs)

    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(CHUNK_CACHE, "w") as f:
        json.dump([{"content": d.page_content, "metadata": d.metadata} for d in all_index_docs], f)
    with open(RAW_RECORDS_CACHE, "w") as f:
        json.dump(raw_records, f)

    print(f"[done] index built: {len(chunks)} report chunks + {len(raw_docs)} raw records "
          f"= {len(all_index_docs)} total, across all 6 datasets.")


def _load_chunks_cache():
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document
    with open(CHUNK_CACHE) as f:
        raw = json.load(f)
    return [Document(page_content=r["content"], metadata=r["metadata"]) for r in raw]


def _load_raw_records_cache():
    if not os.path.exists(RAW_RECORDS_CACHE):
        return []
    with open(RAW_RECORDS_CACHE) as f:
        return json.load(f)


class CulturalQAPipeline:
    def __init__(self, dataset_filter: str = None, enable_images: bool = False,
                 force_moe_rescore: bool = None):
        self.vectordb = load_faiss_index()
        self.chunks = _load_chunks_cache()

        self._fixed_dataset_filter = dataset_filter
        self.retriever = build_full_retriever(self.vectordb, self.chunks, dataset_filter=dataset_filter)

        self._retrievers_by_dataset = {}

        self.encoder = MultimodalEncoder(enable_live_fetch=enable_images)
        self.moe = MoEFusion().eval()
        self._load_trained_moe_weights()
        self.reader = ExtractiveReader()

        # None = auto (self.moe_is_trained decides); True/False = always
        # that setting, regardless of whether a checkpoint loaded. This is
        # what main_with_moe.py / main_without_moe.py set.
        self.force_moe_rescore = force_moe_rescore
        if force_moe_rescore is True and not self.moe_is_trained:
            print("[pipeline] WARNING: force_moe_rescore=True but no trained checkpoint is loaded -- "
                  "MoE re-scoring will run with RANDOM, untrained weights. Results will not be meaningful.")

        raw_records = _load_raw_records_cache()
        self.record_index = build_record_index(raw_records)
        print(f"[pipeline] {len(self.record_index)} raw records available for exact lookup "
              f"across datasets: {sorted(set(ds for ds, _ in self.record_index))}")

    def _resolve_use_moe_rescore(self, explicit: bool = None) -> bool:
        """Single source of truth for 'should MoE re-scoring run right
        now', used by both _rag_answer and _filter_answer so the two
        can't drift out of sync. Priority: an explicit per-call override
        (rarely used) > self.force_moe_rescore (set at construction,
        e.g. by main_with_moe.py/main_without_moe.py) > auto-detect from
        whether a trained checkpoint actually loaded."""
        if explicit is not None:
            return explicit
        if self.force_moe_rescore is not None:
            return self.force_moe_rescore
        return self.moe_is_trained

    def _get_retriever_for(self, dataset_name: str):
        """Returns a retriever that BOOSTS this dataset -- extra retrieval
        budget and ensemble weight -- while still fanning out across all 6.
        Not a hard exclusion of the other 5. Built and cached on first use."""
        if dataset_name not in self._retrievers_by_dataset:
            print(f"[pipeline] building a {dataset_name}-boosted retriever across all datasets "
                  f"(first use, cached after this)")
            self._retrievers_by_dataset[dataset_name] = build_full_retriever(
                self.vectordb, self.chunks, boost_dataset=dataset_name
            )
        return self._retrievers_by_dataset[dataset_name]

    def _load_trained_moe_weights(self):
        checkpoint_path = os.path.join("checkpoints", "moe_fusion.pt")
        self.moe_is_trained = False
        if os.path.exists(checkpoint_path):
            try:
                self.moe.load_checkpoint(checkpoint_path)
                self.moe_is_trained = True
            except Exception as e:
                print(f"[pipeline] found {checkpoint_path} but couldn't load it ({e}) -- "
                      f"using untrained MoE weights instead")
        else:
            print(f"[pipeline] no trained MoE checkpoint at {checkpoint_path} -- "
                  f"using random weights (run train_moe.py to train one)")

    # -------------------------------------------------------------- Path A
    def _try_exact_lookup(self, question: str):
        if not is_lookup_query(question):
            return None

        dataset, record_id = parse_lookup(question)
        if record_id is None:
            return None

        if dataset:
            record = self.record_index.get((dataset, record_id))
            if record:
                return {
                    "answer": format_record_answer(record),
                    "path": "exact_lookup",
                    "dataset": dataset,
                    "record_id": record_id,
                    "score": 1.0,
                }
            ids_for_dataset = sorted(rid for ds, rid in self.record_index if ds == dataset)
            if not ids_for_dataset:
                return {
                    "answer": None,
                    "message": f"No raw records loaded for {dataset} yet -- add its real data "
                               f"files under ./data/raw/{dataset}/ and rebuild the index.",
                }
            return {
                "answer": None,
                "message": f"No record #{record_id} found in {dataset}. "
                           f"Valid range is #{ids_for_dataset[0]}\u2013#{ids_for_dataset[-1]}.",
            }

        matches = {ds: self.record_index[(ds, record_id)]
                   for (ds, rid) in self.record_index if rid == record_id}
        if not matches:
            return {"answer": None, "message": f"No record #{record_id} found in any loaded dataset. "
                                                f"Try naming the dataset, e.g. 'artifact #{record_id} in DOSA'."}
        if len(matches) == 1:
            ds, record = next(iter(matches.items()))
            return {"answer": format_record_answer(record), "path": "exact_lookup",
                    "dataset": ds, "record_id": record_id, "score": 1.0}
        return {
            "answer": None,
            "message": f"Record #{record_id} exists in multiple datasets ({', '.join(matches)}) -- "
                       f"specify which one, e.g. 'artifact #{record_id} in {list(matches)[0]}'.",
        }

    # -------------------------------------------------------------- Path B
    def _moe_rescore(self, question: str, candidates):
        q_text_emb = self.encoder.text_encoder.encode(question)
        q_image_emb = torch.zeros(1, self.encoder.image_encoder.embed_dim)

        scored = []
        for chunk in candidates:
            c_text_emb, c_image_emb = self.encoder.encode_chunk(chunk)
            moe_score = self.moe.score_relevance(
                q_text_emb, q_image_emb, c_text_emb.unsqueeze(0), c_image_emb.unsqueeze(0)
            ).item()

            routing = self.moe.last_aux
            fired = [routing["routed_names"][i] for i in routing["topk_idx"][0].tolist()]
            weights = [round(w, 2) for w in routing["topk_weights"][0].tolist()]
            experts_str = " + ".join(f"{name}({w})" for name, w in zip(fired, weights))

            image_ref = None
            if chunk.metadata.get("has_image"):
                image_ref = (chunk.metadata.get("image_url")
                             or chunk.metadata.get("image_path")
                             or chunk.metadata.get("images"))
            used_real_image = bool(image_ref) and self.encoder.image_encoder.enable_live_fetch
            if used_real_image:
                image_tag = "real image"
            elif chunk.metadata.get("has_image"):
                image_tag = "image available but --enable_images is off"
            else:
                image_tag = "no image for this record"

            ds = chunk.metadata.get("dataset")
            rid = chunk.metadata.get("record_id")
            print(f"[moe] {ds} #{rid}: score={moe_score:.3f} | routed via {experts_str} | {image_tag}")

            scored.append((moe_score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    def _rag_answer(self, question: str, use_moe_rescore: bool = None, top_k_for_reader: int = 3):
        use_moe_rescore = self._resolve_use_moe_rescore(use_moe_rescore)

        retriever = self.retriever
        if not self._fixed_dataset_filter:
            named_dataset = detect_dataset(question)
            if named_dataset:
                retriever = self._get_retriever_for(named_dataset)

        candidates = retriever.invoke(question)
        if use_moe_rescore and candidates:
            candidates = self._moe_rescore(question, candidates)

        top_candidates = candidates[:top_k_for_reader]
        result = self.reader.answer(question, top_candidates)
        result["path"] = "rag"
        result["moe_rescore_used"] = use_moe_rescore
        result["retrieved_from"] = [
            {"dataset": c.metadata.get("dataset"), "source_file": c.metadata.get("source_file"),
             "source_type": c.metadata.get("source_type", "report_chunk")}
            for c in top_candidates
        ]
        return result

    # -------------------------------------------------------------- Path C
    def _filter_answer(self, question: str, top_k_for_list: int = 8, use_moe_rescore: bool = None):
        use_moe_rescore = self._resolve_use_moe_rescore(use_moe_rescore)

        retriever = self.retriever
        if not self._fixed_dataset_filter:
            named_dataset = detect_dataset(question)
            if named_dataset:
                retriever = self._get_retriever_for(named_dataset)

        candidates = retriever.invoke(question)
        if use_moe_rescore and candidates:
            candidates = self._moe_rescore(question, candidates)

        top_candidates = candidates[:top_k_for_list]
        if not top_candidates:
            return {
                "answer": None,
                "message": "No matching records found for this question.",
                "path": "filter_list",
                "moe_rescore_used": use_moe_rescore,
            }

        items = []
        for c in top_candidates:
            items.append({
                "dataset": c.metadata.get("dataset"),
                "record_id": c.metadata.get("record_id"),
                "preview": c.page_content[:200].replace("\n", " ").strip(),
            })

        answer_text = "\n".join(
            f"- [{it['dataset']}" + (f" #{it['record_id']}]" if it["record_id"] is not None else "]")
            + f" {it['preview']}"
            for it in items
        )

        return {
            "answer": answer_text,
            "path": "filter_list",
            "moe_rescore_used": use_moe_rescore,
            "items": items,
            "retrieved_from": [
                {"dataset": c.metadata.get("dataset"), "source_file": c.metadata.get("source_file"),
                 "source_type": c.metadata.get("source_type", "report_chunk")}
                for c in top_candidates
            ],
        }

    # -------------------------------------------------------------- Entry point
    def ask(self, question: str, **rag_kwargs):
        lookup_result = self._try_exact_lookup(question)
        if lookup_result is not None:
            return lookup_result
        if is_filter_query(question):
            return self._filter_answer(question)
        return self._rag_answer(question, **rag_kwargs)

    # -------------------------------------------------------------- Batch mode
    def run_batch(self, questions: list, show_progress: bool = True):
        results = []
        total = len(questions)
        for i, q in enumerate(questions, start=1):
            if show_progress:
                print(f"[batch] {i}/{total}: {q}")
            try:
                result = self.ask(q)
            except Exception as e:
                result = {"answer": None, "message": f"[error] {e}"}
            result["question"] = q
            results.append(result)
        return results

    # -------------------------------------------------------------- Ablation
    def compare_moe_ablation(self, question: str, top_k_for_reader: int = 3):
        if self._try_exact_lookup(question) is not None:
            return {"question": question,
                    "note": "Path A (exact lookup) -- MoE was never in the loop, no ablation to compare"}
        if is_filter_query(question):
            return {"question": question,
                    "note": "Path C (filter/list) -- MoE only reorders the list here, doesn't change "
                            "which records are shown; ablation is most meaningful for Path B"}

        retriever = self.retriever
        if not self._fixed_dataset_filter:
            named_dataset = detect_dataset(question)
            if named_dataset:
                retriever = self._get_retriever_for(named_dataset)

        candidates = retriever.invoke(question)

        baseline_candidates = candidates[:top_k_for_reader]
        baseline_result = self.reader.answer(question, baseline_candidates)

        moe_candidates = None
        moe_result = None
        if self.moe_is_trained and candidates:
            moe_candidates = self._moe_rescore(question, candidates)[:top_k_for_reader]
            moe_result = self.reader.answer(question, moe_candidates)

        def _refs(cands):
            return [f"{c.metadata.get('dataset')}#{c.metadata.get('record_id')}" for c in (cands or [])]

        baseline_answer = baseline_result.get("answer") or baseline_result.get("message")
        moe_answer = (moe_result.get("answer") or moe_result.get("message")) if moe_result else None

        return {
            "question": question,
            "baseline_answer": baseline_answer,
            "baseline_confidence": baseline_result.get("score"),
            "baseline_top_candidates": _refs(baseline_candidates),
            "moe_answer": moe_answer,
            "moe_confidence": moe_result.get("score") if moe_result else None,
            "moe_top_candidates": _refs(moe_candidates),
            "candidate_set_changed": _refs(baseline_candidates) != _refs(moe_candidates) if moe_candidates else None,
            "answer_changed": (baseline_answer != moe_answer) if moe_result else None,
            "note": "" if self.moe_is_trained else "no trained MoE checkpoint loaded -- moe_* columns are empty",
        }

    def run_ablation_comparison(self, questions: list, show_progress: bool = True):
        results = []
        total = len(questions)
        for i, q in enumerate(questions, start=1):
            if show_progress:
                print(f"[ablation] {i}/{total}: {q}")
            try:
                result = self.compare_moe_ablation(q)
            except Exception as e:
                result = {"question": q, "note": f"[error] {e}"}
            results.append(result)
        return results


def ask(question: str, dataset_filter: str = None, enable_images: bool = False,
        force_moe_rescore: bool = None):
    pipeline = CulturalQAPipeline(dataset_filter=dataset_filter, enable_images=enable_images,
                                   force_moe_rescore=force_moe_rescore)
    result = pipeline.ask(question)

    print("\n--- ANSWER ---")
    print(result.get("answer") or result.get("message"))
    if result.get("path") == "exact_lookup":
        print(f"(direct lookup, dataset={result['dataset']}, record #{result['record_id']})")
    elif result.get("answer"):
        print(f"(confidence: {result['score']:.3f}, from: {result.get('dataset')} / {result.get('source_file')})")
    if "moe_rescore_used" in result:
        print(f"(MoE re-scoring: {'ON' if result['moe_rescore_used'] else 'OFF'})")
    if result.get("retrieved_from"):
        print("\n--- CANDIDATES CONSIDERED ---")
        for r in result["retrieved_from"]:
            print(f"- {r['dataset']} ({r.get('source_file') or r.get('source_type')})")
    return result


def run_questions_file(input_path: str, output_path: str = None, dataset_filter: str = None,
                        enable_images: bool = False, force_moe_rescore: bool = None):
    import csv
    import datetime

    with open(input_path, encoding="utf-8") as f:
        questions = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not questions:
        print(f"No questions found in {input_path} (check for blank file or all-comment lines).")
        return

    print(f"[batch] loaded {len(questions)} questions from {input_path}")
    pipeline = CulturalQAPipeline(dataset_filter=dataset_filter, enable_images=enable_images,
                                   force_moe_rescore=force_moe_rescore)
    results = pipeline.run_batch(questions)

    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./batch_results_{timestamp}.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "path", "dataset", "record_id_or_source", "confidence",
                          "moe_rescore_used"])
        for r in results:
            answer = r.get("answer") or r.get("message") or ""
            path = r.get("path", "")
            dataset = r.get("dataset", "")
            record_or_source = r.get("record_id", "") or r.get("source_file", "") or ""
            confidence = r.get("score", "")
            moe_used = r.get("moe_rescore_used", "")
            writer.writerow([r["question"], answer, path, dataset, record_or_source, confidence, moe_used])

    print(f"[batch] done -- {len(results)} answers written to {output_path}")
    return results


def run_ablation_file(input_path: str, output_path: str = None, dataset_filter: str = None,
                       enable_images: bool = False):
    import csv
    import datetime

    with open(input_path, encoding="utf-8") as f:
        questions = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not questions:
        print(f"No questions found in {input_path} (check for blank file or all-comment lines).")
        return

    print(f"[ablation] loaded {len(questions)} questions from {input_path}")
    pipeline = CulturalQAPipeline(dataset_filter=dataset_filter, enable_images=enable_images)
    if not pipeline.moe_is_trained:
        print("[ablation] WARNING: no trained MoE checkpoint loaded -- "
              "the moe_* columns will be empty. Run train_moe.py first for a real comparison.")
    results = pipeline.run_ablation_comparison(questions)

    if output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./ablation_results_{timestamp}.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "baseline_answer", "baseline_confidence", "baseline_top_candidates",
                          "moe_answer", "moe_confidence", "moe_top_candidates",
                          "candidate_set_changed", "answer_changed", "note"])
        for r in results:
            writer.writerow([
                r.get("question", ""),
                r.get("baseline_answer", ""),
                r.get("baseline_confidence", ""),
                " | ".join(r.get("baseline_top_candidates") or []),
                r.get("moe_answer", ""),
                r.get("moe_confidence", ""),
                " | ".join(r.get("moe_top_candidates") or []),
                r.get("candidate_set_changed", ""),
                r.get("answer_changed", ""),
                r.get("note", ""),
            ])

    changed = sum(1 for r in results if r.get("answer_changed"))
    comparable = sum(1 for r in results if r.get("answer_changed") is not None)
    print(f"[ablation] done -- {len(results)} comparisons written to {output_path}")
    if comparable:
        print(f"[ablation] MoE changed the final answer on {changed}/{comparable} comparable questions "
              f"({changed / comparable:.0%})")
    return results