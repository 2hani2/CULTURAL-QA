"""
dataset_config.py
------------------
Small shared constants used by both ingestion.py (heavy, pulls in
LangChain loaders) and query_router.py (needs to stay lightweight since
it runs on every single question). Keeping this separate means asking a
question doesn't have to import PDF/Excel/CSV loaders just to check
"did they type 'DOSA' in their question."

GIMMICK removed by explicit decision -- it's gated on HuggingFace (needs
login + license acceptance) and isn't being pursued, so this pipeline now
covers 6 datasets, not 7: DOSA, SCALE, DRISHTIKON, DIWALI, Durga QA,
Indian Cultural Dataset.
"""

DATASET_ALIASES = {
    "scale": "SCALE",
    "drishtikon": "DRISHTIKON",
    "dhrishtikon": "DRISHTIKON",  # matches the actual report filename typo
                                   # ("DHRISHTIKON_EDA-3.pdf") so that chunk
                                   # gets tagged DRISHTIKON, not its own
                                   # accidental pseudo-dataset
    "diwali": "DIWALI",
    "dosa": "DOSA",
    "durga": "Durga QA",
    "durga qa": "Durga QA",
    "ombhojane": "Indian Cultural Dataset",
    # These full-name aliases matter because the internal file-tag keywords
    # above (e.g. "ombhojane") only match filenames, not what someone would
    # actually type in a question -- "what is record #1 in Indian Cultural
    # Dataset" was failing to detect the dataset at all before these were
    # added, since "indian cultural dataset" doesn't contain "ombhojane".
    "indian cultural dataset": "Indian Cultural Dataset",
    "indian culture dataset": "Indian Cultural Dataset",
}

# Which datasets can carry a real per-record image field (per each report's
# Modality/Image Statistics section). Used by encoders.py so CLIP only runs
# where images actually exist.
HAS_IMAGE_MODALITY = {
    "DOSA": True,
    "DRISHTIKON": True,
    "SCALE": False,
    "DIWALI": False,
    "Durga QA": False,
    "Indian Cultural Dataset": False,
}