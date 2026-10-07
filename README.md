# CulturalQA RAG — All 7 Datasets, Two Answer Paths, 5/7 Verified

**Path A — exact lookup.** "What is artifact #412 in DOSA?" is detected by
`query_router.py` and answered by a direct dictionary lookup in
`structured_store.py`, bypassing search entirely. Deterministic.

**Path B — everything else.** "How many artifacts does DOSA have?" /
"Which artifacts relate to weddings?" go through the full pipeline: FAISS+BM25
hybrid retrieval -> MultiQuery expansion -> Cross-Encoder rerank -> MoE
re-score -> extractive reader. Searches both EDA report chunks and real
per-row dataset records.

## Verification status — 5 of 7 confirmed against real data

| Dataset | Verified? | How | Real columns |
|---|---|---|---|
| **DOSA** | ✅ | live-cloned `github.com/microsoft/DOSA` | `artifact`, `clues` — one folder per state |
| **SCALE** | ✅ | live-cloned `github.com/google-research-datasets/SCALE-Cultural-Data` | `Item`, `Translation` — one `.xlsx` per category |
| **DIWALI** | ✅ | fetched `huggingface.co/datasets/nlip/DIWALI` viewer (ungated, 8,817 rows) | `facet`, `state`, `concept`, `description`, `source` |
| **DRISHTIKON** | ✅ | fetched `huggingface.co/datasets/13ari/DRISHTIKON` viewer (ungated, 64.3k rows) | `language`, `state`, `attribute`, `question_type`, `question`, `option1-4`, `answer`, `image_name`, `image_link`, `image` |
| **Indian Cultural Dataset** | ✅ | fetched `huggingface.co/datasets/ombhojane/indian-cultural-dataset` viewer (ungated, 96 rows) | `text`, `category` — matches its EDA report exactly; some rows run to ~15.4M chars (full Mahabharata text) |
| GIMMICK | ❌ gated | confirmed via direct fetch: "You need to agree to share your contact information to access this dataset" | needs your HF login + license acceptance |
| Durga QA | ❌ not checked | hosted on Mendeley Data, not HuggingFace | DOI `10.17632/6hcx53kywr.1` |

## Getting the real files in

**DIWALI, DRISHTIKON, Indian Cultural Dataset — no login needed:**
```python
from datasets import load_dataset
load_dataset("nlip/DIWALI")["train"].to_csv("./data/raw/DIWALI/data.csv")
load_dataset("13ari/DRISHTIKON")["train"].to_csv("./data/raw/DRISHTIKON/data.csv")
load_dataset("ombhojane/indian-cultural-dataset")["train"].to_json("./data/raw/Indian Cultural Dataset/data.json")
```
Run that locally (your machine has normal HF access, no gating), then either
upload the resulting files to me or drop them straight into the folders above.

**GIMMICK — needs HF login:**
```python
# after `huggingface-cli login` and accepting the license on the dataset page
from datasets import load_dataset
load_dataset("floschne/gimmick-civqa")["train"].to_csv("./data/raw/GIMMICK/civqa.csv")
load_dataset("floschne/gimmick-qa-base")["train"].to_csv("./data/raw/GIMMICK/qa_base.csv")
```

**Durga QA — manual download from Mendeley** (DOI `10.17632/6hcx53kywr.1`),
then drop the CSV into `./data/raw/Durga QA/`.

## Handling the oversized rows (Indian Cultural Dataset)

Most datasets' rows are small enough to index as one Document each. Indian
Cultural Dataset is the exception — some entries are the *entire* text of
the Mahabharata (~15.4M characters). `structured_store.py` checks row length
against `LARGE_ROW_CHAR_THRESHOLD` (4,000 chars) and runs oversized rows
through the same recursive chunker used for report PDFs, while every
resulting chunk keeps the parent `record_id` — so exact lookup (Path A)
still resolves to the full record, and search (Path B) only ever returns a
manageable slice instead of trying to stuff 15M characters into one hit.

## Folder layout required
```
./data/                                          # EDA report PDFs (Path B conceptual questions)
./data/raw/DOSA/<state>/original_artifacts.csv
./data/raw/DOSA/<state>/expanded_artifacts.csv
./data/raw/SCALE/cuisine.xlsx, landmarks.xlsx, ... (7 files)
./data/raw/DIWALI/*.csv
./data/raw/DRISHTIKON/*.csv
./data/raw/Indian Cultural Dataset/*.json
./data/raw/GIMMICK/*.csv                          # you'll need HF login for this one
./data/raw/Durga QA/*.csv                          # Mendeley download
```

## Setup & run
```
pip install -r requirements.txt
export GROQ_API_KEY=your_key
python main.py --build
python main.py --ask "what is artifact #3 in DOSA"              # Path A
python main.py --ask "what is concept #1 in DIWALI"              # Path A
python main.py --ask "how many artifacts does DOSA have"          # Path B
python main.py                                                     # interactive
```

## File map
| File | Purpose |
|---|---|
| `dataset_config.py` | shared dataset name/alias config, kept lightweight |
| `structured_store.py` | loads real per-row files for all 7 datasets, exact-lookup index, chunks oversized rows |
| `query_router.py` | detects ID-lookup questions, routes to Path A |
| `pipeline.py` | tries Path A first, falls back to Path B |
| `encoders.py`, `moe_fusion.py`, `retrieval.py`, `reader.py` | unchanged from the paper-architecture build |

## Still true from before
- MoE fusion layer is architecturally complete but untrained — auxiliary
  re-rank signal, not a standalone ranker.
- Extractive reader avoids hallucination by pulling exact spans.
- Only GIMMICK/DOSA/DRISHTIKON get real CLIP embeddings (DRISHTIKON's
  `image_url` field is now wired through from the verified schema); the
  other 4 get an all-zero placeholder image vector.
