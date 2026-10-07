"""
ingestion.py
------------
Multi-source loading for all 6 datasets (SCALE, DRISHTIKON, DIWALI, DOSA,
Durga QA, Indian Cultural Dataset / ombhojane). GIMMICK removed by explicit
decision -- it's gated on HuggingFace and isn't being pursued.

Follows reference #1 (multi-source LangChain RAG): one loader per file type,
unified into a single corpus with per-chunk metadata identifying which
dataset a fact came from.

Follows reference #2 (RAG with custom unstructured data): report PDFs mix
prose/tables/captions -> we don't try to parse tables structurally here,
we lean on chunking (see chunking.py) to keep table-adjacent text intact.
Structured sources (csv/json/xlsx of the *actual* dataset records, not just
the EDA reports) are loaded row-wise so image/text fields stay paired.

Only 2 of the 6 datasets carry real per-record images (per their EDA
reports): DOSA (optional per-artifact image) and DRISHTIKON (image URL per
MCQA instance). The other four (SCALE, DIWALI, Durga QA, Indian Cultural
Dataset) are text-only by design. This is encoded in HAS_IMAGE_MODALITY
below and drives whether the multimodal encoder (encoders.py) bothers
running CLIP on a given record.
"""

import os
import glob
import re

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredExcelLoader,
    TextLoader,
)

from dataset_config import DATASET_ALIASES, HAS_IMAGE_MODALITY

# Column names across the 6 datasets' actual data files (not the EDA
# reports) that hold an image reference, per each dataset's field docs.
IMAGE_FIELD_CANDIDATES = [
    "image_url", "image_path", "images",  # DOSA / generic
    "image_link", "image_name",            # DRISHTIKON
]


def infer_dataset_tag(filepath: str) -> str:
    name = os.path.basename(filepath).lower()
    for key, label in DATASET_ALIASES.items():
        if key in name:
            return label
    return os.path.splitext(os.path.basename(filepath))[0]


def _attach_image_metadata(doc, tag: str):
    """If this row/doc has an image field and the dataset supports images,
    tag it so the multimodal encoder knows to run CLIP on it."""
    has_image_field = any(
        field in doc.metadata or field in doc.page_content.lower()
        for field in IMAGE_FIELD_CANDIDATES
    )
    doc.metadata["has_image"] = bool(HAS_IMAGE_MODALITY.get(tag, False) and has_image_field)
    return doc


def _clean_pdf_hyphenation(text: str) -> str:
    """PDF text extraction preserves line-wrap hyphens literally -- when a
    PDF renderer breaks 'Indian' across two lines as 'In-' / 'dian', the raw
    extracted text keeps the hyphen and newline exactly as laid out, giving
    'In-\\ndian' instead of 'Indian'. Confirmed happening in practice: an
    extractive-QA answer literally returned "In-\\ndian" as its span,
    verbatim, from a DHRISHTIKON_EDA report chunk.

    This joins hyphen+newline+letter patterns back into one word, which
    catches genuine line-wrap breaks (hyphen immediately followed by a
    newline) while leaving real hyphenated compound words alone (those have
    the hyphen followed by a space or another word on the same line, not a
    newline). Restricted to letters (not \\w) specifically to avoid
    mis-joining numeric ranges like "pages 4-\\n5" into "45"."""
    return re.sub(r"([a-zA-Z])-\n([a-zA-Z])", r"\1\2", text)


def load_all_documents(data_dir: str = "./data"):
    """Loads every supported file in data_dir. Report PDFs (EDA writeups)
    and structured record files (csv/json/xlsx of actual dataset rows) can
    both live in the same folder -- they're distinguished by extension,
    not by content, matching the multi-loader pattern from reference #1."""
    docs = []
    patterns_loaders = {
        "*.pdf": lambda p: PyPDFLoader(p).load(),
        "*.csv": lambda p: CSVLoader(p).load(),
        "*.json": lambda p: JSONLoader(p, jq_schema=".", text_content=False).load(),
        "*.xlsx": lambda p: UnstructuredExcelLoader(p, mode="elements").load(),
        "*.txt": lambda p: TextLoader(p, encoding="utf-8").load(),
        "*.md": lambda p: TextLoader(p, encoding="utf-8").load(),
    }

    for pattern, loader_fn in patterns_loaders.items():
        for filepath in glob.glob(os.path.join(data_dir, pattern)):
            try:
                loaded = loader_fn(filepath)
            except Exception as e:
                print(f"[skip] {filepath}: {e}")
                continue
            tag = infer_dataset_tag(filepath)
            for d in loaded:
                if pattern == "*.pdf":
                    d.page_content = _clean_pdf_hyphenation(d.page_content)
                d.metadata["dataset"] = tag
                d.metadata["source_file"] = os.path.basename(filepath)
                _attach_image_metadata(d, tag)
            docs.extend(loaded)
            print(f"[loaded] {filepath} -> {len(loaded)} docs ({tag}, "
                  f"image-capable={HAS_IMAGE_MODALITY.get(tag, False)})")

    if not docs:
        print(f"No files found in {data_dir}.")
    return docs