"""
reader.py
---------
Matches Fig. 1's final stage: "Extractive QA Reader ... pulls answer spans
directly from retrieved context." Deliberately NOT a generative LLM call --
per your paper's own critique of MuRAG/Mixtral/DeepSeekMoE (Section II),
generative decoders risk producing answers that aren't grounded in any
retrieved source, which is exactly the failure mode this architecture is
built to avoid for underrepresented cultural facts.

Uses a RoBERTa model fine-tuned on SQuAD2 (deepset/roberta-base-squad2),
which also supports "no answer found" -- useful when none of the retrieved
chunks (across any of the 7 datasets) actually contain the answer, so the
system can say so instead of forcing an extraction.

IMPLEMENTATION NOTE: this used to go through transformers' high-level
pipeline("question-answering", ...) convenience wrapper. transformers
5.3.0 removed the "question-answering" task from that wrapper's registry
entirely (confirmed via huggingface/course#1211 -- not a renamed import,
an actual dropped feature). Rather than chase whatever the task gets
renamed to next release, this goes straight to AutoModelForQuestionAnswering
+ AutoTokenizer, which is the stable low-level API underneath the pipeline
wrapper regardless of which convenience tasks a given transformers version
chooses to register.

This is a simplified reimplementation of SQuAD2-style span extraction, not
a full reproduction of every heuristic the original pipeline used
internally (e.g. n-best beam search over multiple candidate spans,
handling contexts that overflow the model's max sequence length by
striding). For contexts under ~384 tokens -- true here, since retrieval
already narrows candidates to short chunks -- this gives materially the
same behavior.

CORRECTION: the first version of this file decided "answer vs no answer"
by multiplying softmax probabilities together (start_prob * end_prob) and
comparing the chosen span's product against the null-answer position's
product. That's not how SQuAD2 models are actually calibrated, and it was
rejecting genuinely correct answers too often (confirmed: a query that
correctly extracted "615" from a DOSA report chunk started returning "no
confident answer" on the same content after this file was introduced).
Fixed to use the standard approach instead: compare the RAW LOGIT SUM of
the best span against the null position's raw logit sum
(score_diff = null_logit_sum - best_span_logit_sum), which is what
SQuAD2's own null_score_diff_threshold convention uses. Softmax is now
only used afterward, to produce a human-readable confidence score for
ranking across candidate chunks -- it no longer drives the answer/no-answer
decision itself.
"""

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

READER_MODEL = "deepset/roberta-base-squad2"
MAX_SEQ_LENGTH = 384


class ExtractiveReader:
    def __init__(self, model_name: str = READER_MODEL, null_score_diff_threshold: float = 0.0):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.eval()
        # Standard SQuAD2 convention: if (null_logit_sum - best_span_logit_sum)
        # exceeds this threshold, the model is predicting "no answer here."
        # 0.0 is the well-calibrated default deepset/roberta-base-squad2 was
        # tuned against; raise it (e.g. 1.0-2.0) to bias toward giving an
        # answer more often if "no confident answer" still feels too frequent
        # once this fix is in.
        self.null_score_diff_threshold = null_score_diff_threshold

    @torch.no_grad()
    def _answer_one(self, question: str, context: str):
        inputs = self.tokenizer(
            question, context,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        outputs = self.model(**inputs)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Search for the best span starting from index 1, excluding the
        # null-answer position (index 0, [CLS]) from the "best span" search
        # itself -- it's compared separately below, not competed for directly.
        start_idx = int(torch.argmax(start_logits[1:])) + 1
        end_idx = int(torch.argmax(end_logits[1:])) + 1
        if end_idx < start_idx:
            end_idx = start_idx

        best_span_logit_sum = float(start_logits[start_idx] + end_logits[end_idx])
        null_logit_sum = float(start_logits[0] + end_logits[0])
        score_diff = null_logit_sum - best_span_logit_sum
        has_answer = score_diff <= self.null_score_diff_threshold

        input_ids = inputs["input_ids"][0]
        answer_text = self.tokenizer.decode(
            input_ids[start_idx:end_idx + 1], skip_special_tokens=True
        ).strip()

        # Softmax confidence is only used for ranking the best answer across
        # multiple candidate chunks in answer() below -- not for the
        # has_answer decision above, which uses raw logits per SQuAD2
        # convention.
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)
        confidence = float(start_probs[start_idx] * end_probs[end_idx])

        return {
            "answer": answer_text if has_answer else "",
            "score": confidence,
            "score_diff": score_diff,
        }

    def answer(self, question: str, candidate_chunks, min_score: float = 0.01):
        """Runs extraction over each candidate chunk independently (SQuAD-style
        readers score per-context) and returns the single best span across
        all candidates, tagged with which dataset it came from."""
        best = None
        for chunk in candidate_chunks:
            context = chunk.page_content
            if not context.strip():
                continue
            try:
                result = self._answer_one(question, context)
            except Exception as e:
                print(f"[reader error] {e}")
                continue

            result["dataset"] = chunk.metadata.get("dataset")
            result["source_file"] = chunk.metadata.get("source_file")
            result["context"] = context

            if best is None or result["score"] > best["score"]:
                best = result

        if best is None or not best["answer"] or best["score"] < min_score:
            return {
                "answer": None,
                "message": "No confident answer span found in retrieved context across any dataset.",
            }
        return best