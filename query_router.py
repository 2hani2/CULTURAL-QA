"""
query_router.py
----------------
Detects "give me record #N" style questions and routes them to a direct
dict lookup in structured_store's record index instead of semantic search.

This matters because embeddings/reranking/extraction are all "find the
most likely relevant text" tools -- they're the wrong tool for "return
exactly row 412," which has one correct answer that either exists or
doesn't. Routing ID questions around the RAG path makes those answers
deterministic instead of "probably right."

Also detects filter/list questions ("which artifacts relate to weddings",
"list all festivals in Bihar") -- these have a SET of correct answers,
not a single extractable span, so they're routed around the extractive
reader too (see is_filter_query / Path C in pipeline.py). Running these
through the single-span reader was producing exactly the failure mode
that motivated adding this: one terse, oddly-truncated fragment from a
single retrieved context ("soklong wedding") instead of the actual list
of matching records the question was asking for.

Everything that ISN'T an ID-lookup or filter/list question falls through
to the normal hybrid retrieval + MoE + extractive reader pipeline,
unchanged.
"""

import re
from dataset_config import DATASET_ALIASES

LOOKUP_TRIGGERS = [
    "artifact #", "artifact number", "artifact id",
    "record #", "record number", "record id",
    "row #", "row number",
    "item #", "item number", "item id",
    "sample #", "sample number",
    "index ",
]

NUMBER_PATTERN = re.compile(r"#?\s*(\d+)\b")

# catches "record 7", "artifact 412", "item 3" etc. even without a '#' or
# the word "number"/"id" attached
KEYWORD_PLUS_NUMBER = re.compile(
    r"\b(artifact|record|row|item|sample)\s*#?\s*\d+\b", re.IGNORECASE
)

# Substring triggers, same style as LOOKUP_TRIGGERS above -- phrases that
# signal "I want a SET of matching records," not one fact. Deliberately
# NOT triggering on bare "which"/"what" alone (e.g. "which language is
# most common in DRISHTIKON" is a single-answer question that happens to
# start with "which") -- requiring one of these more specific phrases
# avoids misrouting genuine single-answer questions into the list path.
FILTER_TRIGGERS = [
    "relate to", "related to", "relating to",
    "associated with", "connected to",
    "list all", "list the",
    "show me all", "show all",
    "all artifacts", "all items", "all records", "all questions",
]


def detect_dataset(question: str):
    q = question.lower()
    for key, label in DATASET_ALIASES.items():
        if key in q:
            return label
    return None


def is_lookup_query(question: str) -> bool:
    q = question.lower()
    has_trigger = (
        any(t in q for t in LOOKUP_TRIGGERS)
        or bool(re.search(r"#\s*\d+", question))
        or bool(KEYWORD_PLUS_NUMBER.search(question))
    )
    has_number = bool(NUMBER_PATTERN.search(question))
    return has_trigger and has_number


def is_filter_query(question: str) -> bool:
    q = question.lower()
    return any(phrase in q for phrase in FILTER_TRIGGERS)


def parse_lookup(question: str):
    """Returns (dataset_tag_or_None, record_id_or_None)."""
    dataset = detect_dataset(question)
    match = NUMBER_PATTERN.search(question)
    record_id = int(match.group(1)) if match else None
    return dataset, record_id