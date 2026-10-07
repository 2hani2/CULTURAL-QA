"""
chunking.py
-----------
Reference #2 (unstructured data) + reference #3 (large datasets): report
text mixes dense prose, table-as-text, and figure captions. A layered
recursive splitter keeps table rows attached to the sentence explaining
them instead of cutting mid-row. Overlap ensures facts sitting on a chunk
boundary still appear whole in at least one chunk.

Every chunk gets a stable content-hash ID scoped to its dataset tag, so
re-running ingestion on updated files doesn't create duplicate vectors.
"""

import hashlib

# LangChain moved text splitters into a standalone langchain_text_splitters
# package in newer releases -- langchain.text_splitter no longer exists in
# recent pip installs. Try the new location first, fall back to the old one
# so this works regardless of which LangChain version got installed.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(docs, chunk_size: int = 900, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for c in chunks:
        h = hashlib.sha1(c.page_content.encode("utf-8")).hexdigest()[:10]
        c.metadata["chunk_id"] = f"{c.metadata.get('dataset', 'unk')}-{h}"

    per_dataset = {}
    for c in chunks:
        per_dataset[c.metadata.get("dataset", "unk")] = per_dataset.get(c.metadata.get("dataset", "unk"), 0) + 1
    print(f"[chunked] {len(docs)} docs -> {len(chunks)} chunks")
    for ds, n in per_dataset.items():
        print(f"    {ds}: {n} chunks")
    return chunks