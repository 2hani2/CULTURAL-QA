"""
structured_store.py
--------------------
This is the actual fix for "what's artifact #412 in DOSA?" style questions.

Semantic RAG (chunking + embedding search) answers questions about
*descriptions* of a dataset ("how many artifacts does DOSA have"). It
cannot answer "what is record #412", because no paragraph anywhere says
that -- that fact only exists as one row in the dataset's real data files,
which the report-based pipeline never loaded.

This module loads the ACTUAL dataset files (not the EDA report PDFs) row
by row, assigns each row a stable ID, and keeps an in-memory index for
O(1) exact lookup -- bypassing embeddings/reranking/extraction entirely
for ID-based questions, which also makes those answers 100% accurate
instead of "most likely chunk."

Verified against real data -- pulled directly from source (GitHub repos
reachable from this environment, HuggingFace dataset viewer pages readable
via web_fetch, and the actual downloaded files' folder/filenames):
  - DOSA:  github.com/microsoft/DOSA -> data/<state>/{original,expanded}_artifacts.csv
           columns: artifact, clues (no built-in numeric ID)
  - SCALE: github.com/google-research-datasets/SCALE-Cultural-Data -> data/*.xlsx
           columns: Item, Translation (domain = filename, no built-in ID)
  - DIWALI: huggingface.co/datasets/nlip/DIWALI (ungated, 8,817 rows)
           columns: facet, state, concept, description, source
           real downloaded file: DIWALIv1.jsonl (JSON Lines format)
  - DRISHTIKON: huggingface.co/datasets/13ari/DRISHTIKON (ungated, 64.3k rows)
           columns: language, state, attribute, question_type, question,
                    option1, option2, option3, option4, answer, image_name,
                    image_link, image
           real downloaded file: Drishtikon_all_languages_and_qtypes_merged_dataset.csv
           (exact header casing not independently confirmed -- loader uses
           _get_field for tolerance)
  - Indian Cultural Dataset: huggingface.co/datasets/ombhojane/indian-cultural-dataset
           (ungated, 96 rows) columns: text, category -- confirmed matches
           the EDA report exactly. Real downloaded files: train.jsonl +
           validation.jsonl (JSON Lines, both loaded and combined). NOTE:
           some rows run up to ~15.4M characters (full Mahabharata text) --
           these get chunked even in the "raw" path, or they'd blow out
           context on a single retrieval hit.
  - Durga QA: real downloaded file is a CSV named along the lines of
           "Question-Answer Pairs Goddess Durga.csv". Columns per the EDA
           report: Jawaban (answer), Q1, Q2, Q3, Q4 (paraphrased question
           variants). Not independently verified against the actual file's
           header -- loader uses _get_field for tolerance.

GIMMICK REMOVED ENTIRELY (was gated on HuggingFace -- "You need to agree to
share your contact information to access this dataset" -- and isn't being
pursued). It's not in DATASET_ALIASES anymore, so it's fully excluded from
every loop in this file, not just skipped-with-a-log-message. This
pipeline covers 6 datasets: DOSA, SCALE, DRISHTIKON, DIWALI, Durga QA,
Indian Cultural Dataset.

Expected folder layout:
  ./data/raw/DOSA/<state>/original_artifacts.csv
  ./data/raw/DOSA/<state>/expanded_artifacts.csv
  ./data/raw/SCALE/cuisine.xlsx, landmarks.xlsx, ... (7 files)
  ./data/raw/DIWALI/DIWALIv1.jsonl
  ./data/raw/DRISHTIKON/Drishtikon_all_languages_and_qtypes_merged_dataset.csv
  ./data/raw/Indian Cultural Dataset/train.jsonl, validation.jsonl
  ./data/raw/Durga QA/*.csv
"""

import os
import csv
import json
import glob

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


# ---------------------------------------------------------------------------
# Dataset-specific loaders (real schema where verified)
# ---------------------------------------------------------------------------
def load_dosa_raw(base_dir: str):
    """Real DOSA layout: one folder per state, each with original_artifacts.csv
    and expanded_artifacts.csv, columns = artifact, clues. No numeric ID exists
    in the source data, so we assign one deterministically: sorted state
    folders, original before expanded, row order within file.

    The GitHub repo's zip extracts with the state folders nested one level
    down inside a 'data' subfolder (DOSA-main/data/<state>/...), so if you
    dropped the whole extracted repo into ./data/raw/DOSA/, the real state
    folders end up at ./data/raw/DOSA/data/<state>/ instead of directly
    under ./data/raw/DOSA/<state>/. This auto-detects that and drills in,
    so you don't have to manually move files around."""
    records = []
    idx = 0
    if not os.path.isdir(base_dir):
        print(f"[structured_store] DOSA: {base_dir} does not exist or isn't a folder")
        return records

    def _has_state_folders(d):
        """True if d directly contains subfolders that themselves contain
        either expected CSV filename -- i.e. d is the real state-folder root."""
        if not os.path.isdir(d):
            return False
        for entry in os.listdir(d):
            p = os.path.join(d, entry)
            if os.path.isdir(p) and (
                os.path.exists(os.path.join(p, "original_artifacts.csv"))
                or os.path.exists(os.path.join(p, "expanded_artifacts.csv"))
            ):
                return True
        return False

    search_root = base_dir
    if not _has_state_folders(base_dir):
        nested = os.path.join(base_dir, "data")
        if _has_state_folders(nested):
            print(f"[structured_store] DOSA: state folders not directly under {base_dir}, "
                  f"found them under nested '{nested}' instead -- using that")
            search_root = nested

    state_dirs_found = 0
    files_found = 0
    for state_dir in sorted(os.listdir(search_root)):
        state_path = os.path.join(search_root, state_dir)
        if not os.path.isdir(state_path):
            continue
        state_dirs_found += 1
        for fname, split in [("original_artifacts.csv", "original"), ("expanded_artifacts.csv", "expanded")]:
            fpath = os.path.join(state_path, fname)
            if not os.path.exists(fpath):
                continue
            files_found += 1
            for row in _read_csv_rows(fpath):
                idx += 1
                records.append({
                    "dataset": "DOSA",
                    "id": idx,
                    "state": state_dir,
                    "split": split,
                    "artifact": (row.get("artifact") or "").strip(),
                    "clues": (row.get("clues") or "").strip(),
                })

    if idx == 0:
        sample = os.listdir(search_root)[:5] if os.path.isdir(search_root) else []
        print(f"[structured_store] DOSA: loaded 0 records. Diagnostics: "
              f"searched under {search_root}, {state_dirs_found} state subfolders found, "
              f"{files_found} matching CSV files found inside them. "
              f"First few entries seen: {sample}. "
              f"Expected each state folder to directly contain "
              f"'original_artifacts.csv' and/or 'expanded_artifacts.csv'.")
    else:
        print(f"[structured_store] DOSA: loaded {idx} real artifact records")
    return records


def load_scale_raw(base_dir: str):
    """Real SCALE layout: one xlsx per concept category, columns = Item,
    Translation. Domain comes from the filename, matching the EDA report."""
    records = []
    idx = 0
    if not os.path.isdir(base_dir) or openpyxl is None:
        return records

    for fpath in sorted(glob.glob(os.path.join(base_dir, "*.xlsx"))):
        domain = os.path.splitext(os.path.basename(fpath))[0]
        wb = openpyxl.load_workbook(fpath, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue
        header = [str(h).strip() for h in rows[0]]
        for r in rows[1:]:
            if not any(r):
                continue
            idx += 1
            rec = dict(zip(header, r))
            rec["dataset"] = "SCALE"
            rec["id"] = idx
            rec["domain"] = domain
            records.append(rec)
    print(f"[structured_store] SCALE: loaded {idx} real item records")
    return records


# ---------------------------------------------------------------------------
# Generic fallback loader for the 5 datasets we couldn't verify live.
# Works on csv/json/xlsx/parquet; assigns a sequential ID unless the file
# already has an obvious id-like column.
# ---------------------------------------------------------------------------
ID_COLUMN_HINTS = ["id", "sample_uuid", "question_id", "artifact_id", "record_id"]

# Not every CSV out there is UTF-8. Excel-exported CSVs (common for
# academic dataset releases) often end up as Windows-1252 or Latin-1
# instead, especially when the source has non-ASCII characters anywhere
# in it. Try encodings in order of likelihood until one actually decodes.
CSV_ENCODING_CANDIDATES = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


def _read_csv_rows(fpath: str):
    """Reads a CSV trying multiple encodings AND sniffing the actual
    delimiter (comma isn't universal -- Excel exports from non-US locales,
    including Indonesian, commonly default to semicolon-delimited CSV).
    If comma-delimiter parsing produces a single giant column instead of
    several, that's the signature of a delimiter mismatch, so this
    re-parses with the sniffed delimiter automatically instead of silently
    returning garbage rows with none of the expected fields populated."""
    last_error = None
    for enc in CSV_ENCODING_CANDIDATES:
        try:
            with open(fpath, encoding=enc) as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                except csv.Error:
                    dialect = csv.excel  # sniff failed -- fall back to default comma
                rows = list(csv.DictReader(f, dialect=dialect))

            if enc != CSV_ENCODING_CANDIDATES[0]:
                print(f"[structured_store] {fpath}: not UTF-8, read successfully as {enc}")
            if dialect is not csv.excel and getattr(dialect, "delimiter", ",") != ",":
                print(f"[structured_store] {fpath}: detected '{dialect.delimiter}' as delimiter, not comma")

            if rows and len(rows[0]) == 1:
                # only one column came out of the header -- almost certainly
                # a delimiter mismatch that the sniffer also missed. Show
                # the actual header so the real column names are visible
                # instead of just silently returning near-empty records.
                actual_header = list(rows[0].keys())
                print(f"[structured_store] {fpath}: WARNING -- parsed as only 1 column "
                      f"({actual_header}), delimiter detection likely failed. "
                      f"First 200 chars of file: {sample[:200]!r}")
            return rows
        except UnicodeDecodeError as e:
            last_error = e
            continue
    print(f"[structured_store] {fpath}: could not decode with any of {CSV_ENCODING_CANDIDATES} -- {last_error}")
    return []


def _rows_from_file(fpath: str):
    ext = os.path.splitext(fpath)[1].lower()
    if ext == ".csv":
        return _read_csv_rows(fpath)
    if ext == ".json":
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]
    if ext == ".jsonl":
        # JSON Lines: one JSON object per line (DIWALIv1.jsonl, train.jsonl,
        # validation.jsonl both use this format, not a single JSON array)
        rows = []
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[skip malformed jsonl line in {fpath}] {e}")
        return rows
    if ext in (".xlsx", ".xls") and openpyxl is not None:
        wb = openpyxl.load_workbook(fpath, read_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []
        header = [str(h).strip() for h in rows[0]]
        return [dict(zip(header, r)) for r in rows[1:] if any(r)]
    if ext == ".parquet" and pd is not None:
        return pd.read_parquet(fpath).to_dict(orient="records")
    return []


def _get_field(row: dict, *candidates, default=""):
    """Case-insensitive, variant-tolerant field lookup. Used where I
    couldn't personally verify the exact header spelling of a downloaded
    file (e.g. 'question_type' vs 'Question_Type' vs 'QuestionType') --
    tries each candidate key as given, then lowercased, then with
    spaces/underscores stripped, before giving up and returning default."""
    if not row:
        return default
    lower_map = {str(k).lower(): v for k, v in row.items()}
    stripped_map = {str(k).lower().replace("_", "").replace(" ", ""): v for k, v in row.items()}
    for c in candidates:
        if c in row:
            return row[c]
        if c.lower() in lower_map:
            return lower_map[c.lower()]
        key_stripped = c.lower().replace("_", "").replace(" ", "")
        if key_stripped in stripped_map:
            return stripped_map[key_stripped]
    return default


def load_generic_raw(dataset_tag: str, base_dir: str):
    records = []
    idx = 0
    if not os.path.isdir(base_dir):
        return records

    for fpath in sorted(glob.glob(os.path.join(base_dir, "*"))):
        rows = _rows_from_file(fpath)
        for row in rows:
            id_col = next((c for c in row if str(c).lower() in ID_COLUMN_HINTS), None)
            idx += 1
            rec = dict(row)
            rec["dataset"] = dataset_tag
            rec["id"] = row.get(id_col, idx) if id_col else idx
            records.append(rec)
    print(f"[structured_store] {dataset_tag}: loaded {idx} generic records from {base_dir}")
    return records


RAW_LOADERS = {
    "DOSA": load_dosa_raw,
    "SCALE": load_scale_raw,
}


# ---------------------------------------------------------------------------
# Verified-schema loaders for the 3 more datasets confirmed via HuggingFace
# dataset viewer (ungated -- readable without login). These read whatever
# CSV/JSON/parquet you've exported/downloaded into ./data/raw/<Dataset>/,
# and just enforce/expect the confirmed column names rather than guessing.
# ---------------------------------------------------------------------------
def load_diwali_raw(base_dir: str):
    """Confirmed real columns (huggingface.co/datasets/nlip/DIWALI):
    facet, state, concept, description, source. No built-in numeric ID.
    Real downloaded file is DIWALIv1.jsonl (JSON Lines), not CSV."""
    records = []
    idx = 0
    if not os.path.isdir(base_dir):
        return records
    for fpath in sorted(glob.glob(os.path.join(base_dir, "*"))):
        for row in _rows_from_file(fpath):
            idx += 1
            records.append({
                "dataset": "DIWALI",
                "id": idx,
                "facet": _get_field(row, "facet"),
                "state": _get_field(row, "state"),
                "concept": _get_field(row, "concept"),
                "description": _get_field(row, "description"),
                "source": _get_field(row, "source"),
            })
    print(f"[structured_store] DIWALI: loaded {idx} real concept records")
    return records


def load_drishtikon_raw(base_dir: str):
    """Confirmed real columns (huggingface.co/datasets/13ari/DRISHTIKON):
    language, state, attribute, question_type, question, option1-4, answer,
    image_name, image_link, image. No built-in numeric ID.
    Uses _get_field (case/format-tolerant lookup) since the exact header
    casing in the downloaded merged CSV wasn't independently confirmed.

    IMAGE LOADING: originally this pointed image_url at the remote
    image_link column, but every single one of those URLs turned out to be
    a source/attribution webpage (Wikipedia articles, news pages) rather
    than a direct image file -- confirmed via 0/5 successful live fetches
    across multiple real questions, all failing with "cannot identify
    image file" because PIL was being handed HTML, not image bytes.

    The real images live in Dataset_Images.zip on the same HF repo,
    extracted locally to a folder (e.g. All_Images_formatted_final_dataset)
    inside this dataset's raw folder. This now looks for that folder and
    matches each record's image_name to a real local file -- no network
    call needed at all for DRISHTIKON images once this is set up, which is
    both faster and actually works, unlike the remote image_link."""
    records = []
    idx = 0
    if not os.path.isdir(base_dir):
        return records

    # Auto-detect the extracted images folder rather than hardcoding one
    # exact name -- looks for any subfolder under base_dir whose name
    # contains "image" (case-insensitive), so this works whether you named
    # it All_Images_formatted_final_dataset, images, Dataset_Images, etc.
    images_dir = None
    for entry in os.listdir(base_dir):
        full = os.path.join(base_dir, entry)
        if os.path.isdir(full) and "image" in entry.lower():
            images_dir = full
            break

    if images_dir:
        print(f"[structured_store] DRISHTIKON: found local images folder at {images_dir}")
    else:
        print(f"[structured_store] DRISHTIKON: no local images folder found under {base_dir} "
              f"-- image_url will stay empty (falls back to text-only, same as before)")

    # Build a lookup of actual filenames on disk once, rather than calling
    # os.path.exists() per record -- much faster for 64K+ records, and lets
    # us do a case-insensitive fallback match cheaply too (CSV filename
    # casing doesn't always match the actual file on disk exactly).
    local_files_by_lower_name = {}
    if images_dir:
        for fname in os.listdir(images_dir):
            local_files_by_lower_name[fname.lower()] = os.path.join(images_dir, fname)

    matched_count = 0
    unmatched_count = 0

    for fpath in sorted(glob.glob(os.path.join(base_dir, "*"))):
        for row in _rows_from_file(fpath):
            idx += 1
            image_name = _get_field(row, "image_name")

            local_image_path = ""
            if image_name and images_dir:
                # try exact match first, then case-insensitive fallback
                exact = os.path.join(images_dir, image_name)
                if os.path.exists(exact):
                    local_image_path = exact
                    matched_count += 1
                elif image_name.lower() in local_files_by_lower_name:
                    local_image_path = local_files_by_lower_name[image_name.lower()]
                    matched_count += 1
                else:
                    unmatched_count += 1

            records.append({
                "dataset": "DRISHTIKON",
                "id": idx,
                "language": _get_field(row, "language"),
                "state": _get_field(row, "state"),
                "attribute": _get_field(row, "attribute"),
                "question_type": _get_field(row, "question_type", "questiontype"),
                "question": _get_field(row, "question"),
                "option1": _get_field(row, "option1", "option_1"),
                "option2": _get_field(row, "option2", "option_2"),
                "option3": _get_field(row, "option3", "option_3"),
                "option4": _get_field(row, "option4", "option_4"),
                "answer": _get_field(row, "answer", "correct_answer"),
                "image_name": image_name,
                # local file path if found, else empty -- encoders.py
                # already treats any non-"http" ref as a local path
                "image_url": local_image_path,
                # kept purely for reference/debugging, never used for
                # loading since it doesn't resolve to real image bytes
                "image_source_link": _get_field(row, "image_link"),
            })

    if images_dir:
        print(f"[structured_store] DRISHTIKON: matched {matched_count} records to local images, "
              f"{unmatched_count} had an image_name with no matching file on disk")
    print(f"[structured_store] DRISHTIKON: loaded {idx} real MCQA records")
    return records


# Row text above this length gets its own recursive chunk pass instead of
# being indexed as one giant Document. Indian Cultural Dataset has entries
# up to ~15.4M characters (full Mahabharata text) per its EDA report --
# far too large for a single retrieval hit or an LLM context window.
LARGE_ROW_CHAR_THRESHOLD = 4000

# Even chunked, a 15.4M-character row at 1200 chars/chunk produces ~12,800
# chunks from ONE record -- that's not a retrieval improvement, it's an
# accidental denial-of-service against your own embedding step (this is
# almost certainly why a --build looked "stuck": ~27,000 chunks came out of
# roughly 90 oversized rows, on top of DRISHTIKON's 64K rows, meaning
# 100K+ documents queued for CPU embedding with zero progress output).
# Cap how many chunks any single record can contribute to the SEARCH index.
# This doesn't lose data -- the full, untruncated text is always still
# retrievable via Path A exact lookup (structured_store's record index),
# which reads from the raw record cache, not from these capped chunks.
# This cap only limits how deep semantic SEARCH can reach into one giant
# record before diminishing returns aren't worth the embedding cost.
MAX_CHUNKS_PER_LARGE_ROW = 15


def load_indian_cultural_raw(base_dir: str):
    """Confirmed real columns (huggingface.co/datasets/ombhojane/indian-cultural-dataset):
    text, category. Matches the EDA report exactly. No built-in numeric ID.
    Real downloaded files are train.jsonl + validation.jsonl (JSON Lines),
    not a single .json array -- both get loaded and combined here."""
    records = []
    idx = 0
    if not os.path.isdir(base_dir):
        return records
    for fpath in sorted(glob.glob(os.path.join(base_dir, "*"))):
        for row in _rows_from_file(fpath):
            idx += 1
            records.append({
                "dataset": "Indian Cultural Dataset",
                "id": idx,
                "text": _get_field(row, "text"),
                "category": _get_field(row, "category"),
            })
    print(f"[structured_store] Indian Cultural Dataset: loaded {idx} real records "
          f"(some may be very large -- see LARGE_ROW_CHAR_THRESHOLD)")
    return records


def load_durga_raw(base_dir: str):
    """Real columns per the EDA report ('Question-Answer Pairs Goddess
    Durga'): Jawaban (answer), Q1, Q2, Q3, Q4 (four paraphrased question
    variants per answer). No built-in numeric ID -- assigned sequentially
    in file/row order. Not personally verified against the actual
    downloaded CSV header, so uses _get_field for tolerance to casing/
    naming variants (e.g. 'jawaban' vs 'Jawaban', 'Q1' vs 'question1')."""
    records = []
    idx = 0
    if not os.path.isdir(base_dir):
        return records
    printed_diagnostic = False
    for fpath in sorted(glob.glob(os.path.join(base_dir, "*"))):
        for row in _rows_from_file(fpath):
            idx += 1
            answer = _get_field(row, "Jawaban", "jawaban", "answer")
            q1 = _get_field(row, "Q1", "question1", "q1")
            if not answer and not q1 and not printed_diagnostic and row:
                # Neither expected field matched anything -- show the real
                # column names actually present so the mismatch is visible
                # instead of silently producing empty records.
                print(f"[structured_store] Durga QA: expected columns not found. "
                      f"Actual columns in this file: {list(row.keys())}")
                printed_diagnostic = True
            records.append({
                "dataset": "Durga QA",
                "id": idx,
                "answer": answer,
                "q1": q1,
                "q2": _get_field(row, "Q2", "question2", "q2"),
                "q3": _get_field(row, "Q3", "question3", "q3"),
                "q4": _get_field(row, "Q4", "question4", "q4"),
            })
    print(f"[structured_store] Durga QA: loaded {idx} real QA-pair records")
    return records


RAW_LOADERS.update({
    "DIWALI": load_diwali_raw,
    "DRISHTIKON": load_drishtikon_raw,
    "Indian Cultural Dataset": load_indian_cultural_raw,
    "Durga QA": load_durga_raw,
})


def load_all_raw_records(raw_dir: str = "./data/raw"):
    from dataset_config import DATASET_ALIASES

    all_records = []
    for dataset_tag in set(DATASET_ALIASES.values()):
        ds_dir = os.path.join(raw_dir, dataset_tag)
        if not os.path.isdir(ds_dir):
            print(f"[structured_store] no raw folder for {dataset_tag} at {ds_dir} -- "
                  f"row-level lookup unavailable for this dataset until you add it.")
            continue
        loader = RAW_LOADERS.get(dataset_tag, lambda d: load_generic_raw(dataset_tag, d))
        all_records.extend(loader(ds_dir))
    return all_records


def build_record_index(records):
    """(dataset, id) -> record dict, for O(1) exact lookup."""
    return {(r["dataset"], r["id"]): r for r in records}


def _normalize_label(value) -> str:
    """Cleans up raw folder-name/filename-derived labels ('tamil_nadu',
    'andhra_pradesh') into natural text ('Tamil Nadu', 'Andhra Pradesh')
    before they're embedded into indexed or displayed text.

    Real, confirmed bug this fixes: DOSA's state field comes straight from
    os.listdir() on the raw folder structure -- lowercase, underscore-
    separated, zero normalization -- so a DOSA record's indexed text
    literally contained the string "State: tamil_nadu (expanded)". Asking
    about "Tamil Nadu" then let the extractive reader anchor on this
    near-exact lexical match and extract the raw label itself instead of
    the actual answer sitting right next to it -- confirmed as the direct
    cause of a real garbage answer ("tamil_nadu") in production use.

    Applied defensively to every dataset with a state/domain field, not
    just DOSA, since DIWALI/DRISHTIKON's state values and SCALE's
    filename-derived domain weren't independently verified to already be
    clean either."""
    if not value:
        return value
    return str(value).replace("_", " ").replace("-", " ").strip().title()


def record_to_text(r: dict) -> str:
    if r["dataset"] == "DOSA":
        return f"Artifact: {r['artifact']}. State: {_normalize_label(r['state'])} ({r['split']}). Description: {r['clues']}"
    if r["dataset"] == "SCALE":
        return f"Item: {r.get('Item')}. Translation: {r.get('Translation')}. Domain: {_normalize_label(r.get('domain'))}"
    if r["dataset"] == "DIWALI":
        return (f"Concept: {r.get('concept')}. Facet: {r.get('facet')}. State: {_normalize_label(r.get('state'))}. "
                f"Description: {r.get('description')}")
    if r["dataset"] == "DRISHTIKON":
        # Distractor options deliberately dropped here (unlike
        # format_record_answer below, which still shows them for direct
        # lookups). Confirmed real bug: with all 4 options + the answer
        # both in context, the correct answer's text appeared TWICE
        # (once as a distractor, once after "Answer:"), and the extractive
        # reader's span selection got confused between the two
        # occurrences, producing a span that swallowed the entire options
        # list instead of the single correct answer. Since the answer is
        # already known from r['answer'], the distractors add nothing for
        # retrieval/extraction/training purposes -- only ambiguity.
        return (f"Question ({r.get('language')}, {_normalize_label(r.get('state'))}, {r.get('attribute')}): "
                f"{r.get('question')} Answer: {r.get('answer')}")
    if r["dataset"] == "Indian Cultural Dataset":
        return f"Category: {r.get('category')}. {r.get('text')}"
    if r["dataset"] == "Durga QA":
        return (f"Answer: {r.get('answer')}. Asked as: {r.get('q1')} / {r.get('q2')} / "
                f"{r.get('q3')} / {r.get('q4')}")
    parts = [f"{k}: {v}" for k, v in r.items() if k not in ("dataset", "id") and v not in (None, "")]
    return "; ".join(parts)


def records_to_documents(records):
    """One Document per row, UNCHUNKED (rows are already small) -- these
    feed straight into the same FAISS/BM25 index as the report chunks, so
    conceptual questions ('artifacts related to weddings') can also
    surface real rows, not just report prose.

    Exception: rows whose text exceeds LARGE_ROW_CHAR_THRESHOLD (e.g. Indian
    Cultural Dataset's full-text Mahabharata entries) get split into
    multiple sub-chunks via the same recursive splitter used for report
    text, so a single retrieval hit doesn't try to swallow 15 million
    characters. Each sub-chunk keeps the parent record_id so exact lookup
    (Path A) still resolves to the whole record, while search (Path B)
    only ever returns a manageable slice."""
    from chunking import chunk_documents

    small_docs = []
    large_docs = []
    for r in records:
        text = record_to_text(r)
        meta = {"dataset": r["dataset"], "record_id": r["id"], "source_type": "raw_record"}
        meta.update({k: v for k, v in r.items() if k not in ("dataset", "id")})
        # This is the flag encoders.py checks before running CLIP. It was
        # only ever being set for EDA report chunks (ingestion.py), never
        # for real per-row records -- meaning DRISHTIKON's genuine
        # image_url data was silently never reaching the image encoder
        # despite being present. Set it here based on whether this record
        # actually carries a non-empty image_url.
        meta["has_image"] = bool(meta.get("image_url"))
        doc = Document(page_content=text, metadata=meta)
        if len(text) > LARGE_ROW_CHAR_THRESHOLD:
            large_docs.append(doc)
        else:
            small_docs.append(doc)

    if large_docs:
        print(f"[structured_store] chunking {len(large_docs)} oversized rows "
              f"(> {LARGE_ROW_CHAR_THRESHOLD} chars) instead of indexing whole")
        large_docs = chunk_documents(large_docs, chunk_size=1200, chunk_overlap=200)

        # Cap chunks per record_id -- without this, a handful of
        # million-character rows can silently balloon into tens of
        # thousands of embeddings and make --build look frozen.
        from collections import defaultdict
        by_record = defaultdict(list)
        for d in large_docs:
            by_record[d.metadata["record_id"]].append(d)

        capped_docs = []
        dropped_total = 0
        for record_id, docs in by_record.items():
            capped_docs.extend(docs[:MAX_CHUNKS_PER_LARGE_ROW])
            dropped_total += max(0, len(docs) - MAX_CHUNKS_PER_LARGE_ROW)
        if dropped_total:
            print(f"[structured_store] capped oversized-row chunks to "
                  f"{MAX_CHUNKS_PER_LARGE_ROW}/record -- dropped {dropped_total} chunks "
                  f"from the search index (full text still available via exact lookup)")
        large_docs = capped_docs

    return small_docs + large_docs


def format_record_answer(record: dict) -> str:
    if record["dataset"] == "DOSA":
        return (f"DOSA artifact #{record['id']} ({_normalize_label(record['state'])}, {record['split']} set): "
                f"\"{record['artifact']}\" -- {record['clues']}")
    if record["dataset"] == "SCALE":
        return (f"SCALE item #{record['id']} ({_normalize_label(record.get('domain'))}): "
                f"\"{record.get('Item')}\" (translation: {record.get('Translation')})")
    if record["dataset"] == "DIWALI":
        return (f"DIWALI concept #{record['id']} ({record.get('facet')}, {_normalize_label(record.get('state'))}): "
                f"\"{record.get('concept')}\" -- {record.get('description')}")
    if record["dataset"] == "DRISHTIKON":
        return (f"DRISHTIKON question #{record['id']} ({_normalize_label(record.get('state'))}, {record.get('attribute')}, "
                f"{record.get('language')}): {record.get('question')} "
                f"[Options: {record.get('option1')}, {record.get('option2')}, "
                f"{record.get('option3')}, {record.get('option4')}] Answer: {record.get('answer')}")
    if record["dataset"] == "Indian Cultural Dataset":
        preview = (record.get("text") or "")[:300]
        suffix = "..." if len(record.get("text") or "") > 300 else ""
        return f"Indian Cultural Dataset record #{record['id']} ({record.get('category')}): {preview}{suffix}"
    if record["dataset"] == "Durga QA":
        return (f"Durga QA record #{record['id']}: \"{record.get('q1')}\" -> {record.get('answer')}")
    parts = [f"{k}={v}" for k, v in record.items() if k not in ("dataset", "id")]
    return f"{record['dataset']} record #{record['id']}: " + ", ".join(parts)