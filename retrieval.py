"""
retrieval.py
------------
Matches Fig. 1's "FAISS Retrieval" + "Cross-Encoder Reranker" stages, and
layers in the large-dataset strategies from reference #3:
  - Hybrid dense (FAISS) + sparse (BM25) retrieval via EnsembleRetriever,
    so exact numbers/proper nouns (BM25) and semantic matches (dense) both
    get caught -- matters more as 6 datasets' worth of chunks pile up.
  - MultiQueryRetriever expands a vague question into several phrasings
    before searching, improving recall on a large/varied corpus.
  - CrossEncoderReranker narrows the pooled candidates back down to a
    small, high-precision set before anything expensive happens downstream.

FAN-OUT (was: hard per-dataset scoping): a general question used to search
either the WHOLE unfiltered pool, or -- if the question happened to
contain a dataset name -- get walled off to ONLY that one dataset
(dataset_filter, an exact metadata match, excluding the other 5 entirely).
That hard exclusion was originally added to fix a real bug (DRISHTIKON's
64K+ chunks statistically dominating an unfiltered global search for any
generic-sounding question), but it overcorrected: naming a dataset now
means "search a general question across all six datasets, and let the
one that's actually right win" -- exactly what pooling everything under a
correctly-trained MoE fusion layer is supposed to make possible.

Three retrieval modes now, controlled by build_hybrid_retriever /
build_full_retriever:
  - dataset_filter set (explicit, e.g. `main.py --dataset SCALE`): HARD
    scope to exactly that one dataset. Unchanged from before -- this is a
    deliberate user choice, not an inferred one, so excluding the rest is
    correct here.
  - boost_dataset set, dataset_filter unset (a question named a dataset,
    but wasn't given an explicit --dataset flag): fan out across ALL 6
    datasets with guaranteed baseline budget each, but give the named
    dataset boost_factor x the retrieval budget/ensemble weight. A named
    dataset is a strong hint, not a wall.
  - neither set (fully general question): fan out across all 6 datasets
    with EQUAL budget/weight each.

Fan-out only adds cheap FAISS/BM25 calls at the retrieval stage -- the
expensive parts (MultiQueryRetriever's LLM call, CrossEncoderReranker)
still run exactly ONCE per question either way, since build_hybrid_retriever
returns a single merged retriever regardless of which mode it built.
"""

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq

import re

from dataset_config import DATASET_ALIASES

ALL_DATASETS = sorted(set(DATASET_ALIASES.values()))


def _bm25_preprocess(text: str):
    """BM25Retriever's default preprocessing is a bare text.split() --
    no lowercasing, no punctuation stripping (confirmed directly:
    default_preprocessing_func(text) = text.split()). Real, confirmed
    problem this causes: record_to_text's consistent "Field: value."
    formatting (e.g. "Item: Kurta. Domain: Clothing.") produces tokens
    like 'Item:' and 'Clothing.' -- capitalized, glued to punctuation --
    which share ZERO vocabulary with a natural-language query's 'item'/
    'clothing' tokens under the default tokenizer. This isn't specific to
    any one dataset; every dataset's raw records use this same
    "Field: value." pattern, so BM25 was effectively non-functional
    against real record content and was instead favoring report-chunk
    PDF prose (which doesn't have this same rigid structure) by default.
    Confirmed with a direct before/after test: this exact fix took a
    clothing query's top-3 results from 100% wrong-source report chunks
    to 100% correct raw records."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()

# LangChain 1.0 stripped EnsembleRetriever, ContextualCompressionRetriever,
# MultiQueryRetriever, and CrossEncoderReranker out of the core `langchain`
# package entirely into a new standalone `langchain-classic` package. Try
# the new location first (matches LangChain 1.0+ installs), fall back to
# the pre-1.0 `langchain.retrievers` location for older installs, so this
# keeps working regardless of which version pip grabbed.
try:
    from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
except ImportError:
    from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker

# HuggingFaceEmbeddings / HuggingFaceCrossEncoder have been migrated into a
# standalone langchain_huggingface package in newer LangChain releases,
# same relocation pattern that broke chunking.py's text splitter import --
# try the new package first, fall back to the older langchain_community
# location so this keeps working regardless of install version.
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_huggingface import HuggingFaceCrossEncoder
except ImportError:
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

INDEX_DIR = "./faiss_store"

# Was BAAI/bge-small-en-v1.5 (English-only -- "en" in the model name).
# Confirmed real problem: Durga QA's source data is in Indonesian
# ("Dewi apa Durga?" / "Durga adalah dewi Hindu."), so an English-only
# embedding model has no shared meaning-space with it at all -- an
# English query like "how many arms does Durga have" was retrieving zero
# real Durga QA content, only the (also English) EDA report PDF.
#
# Switched to a multilingual model -- confirmed via its model card to
# explicitly cover Indonesian ('id' in its language list, alongside 50
# others) -- trained on parallel data across those languages, so English
# queries and Indonesian documents land in a shared embedding space.
# Chose the MiniLM variant over the larger bge-m3 specifically for CPU
# cost: 118M params vs bge-m3's ~568M, meaningfully faster to re-embed
# the full ~79K-document corpus on a CPU-only machine, while still
# covering the one language gap that actually exists in this project.
#
# NOTE: this only changes retrieval (FAISS). encoders.py's TextEncoder
# (bert-base-uncased, used for MoE fusion) deliberately stays English-only
# -- a known, accepted tradeoff: MoE re-scoring will still be weaker on
# Durga QA specifically, but retrieval will now actually find its content
# in the first place, which is the bigger of the two gaps.
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def build_faiss_index(chunks, index_dir: str = INDEX_DIR, batch_size: int = 500):
    """Embeds and indexes in batches with visible progress. The original
    single FAISS.from_documents(all_chunks) call gives zero terminal output
    until every single document is embedded -- indistinguishable from being
    frozen once you're past ~50K documents on a CPU. Batching prints a
    running count so you can see it's actually working, and
    encode_kwargs={'show_progress_bar': True} turns on sentence-transformers'
    own per-batch progress bar as an extra signal of life."""
    import time

    # NOTE: don't pass show_progress_bar via encode_kwargs -- the
    # langchain_huggingface wrapper already passes that keyword internally
    # from its own `show_progress` constructor arg, so setting it in
    # encode_kwargs too causes "got multiple values for keyword argument
    # 'show_progress_bar'". Use the wrapper's dedicated param instead.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        show_progress=True,
    )

    total = len(chunks)
    print(f"[faiss] embedding {total} documents in batches of {batch_size} "
          f"(this can take a while on CPU for 10K+ documents -- watch the "
          f"counts below to confirm it's progressing, not frozen)")

    vectordb = None
    start = time.time()
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        if vectordb is None:
            vectordb = FAISS.from_documents(batch, embeddings)
        else:
            vectordb.add_documents(batch)
        done = min(i + batch_size, total)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        print(f"[faiss]   {done}/{total} embedded "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining at current rate)")

    vectordb.save_local(index_dir)
    print(f"[faiss] index saved to {index_dir}")
    return vectordb


def load_faiss_index(index_dir: str = INDEX_DIR):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def _build_single_dataset_hybrid(vectordb, chunks, k: int, dataset_filter: str):
    """The dense+BM25 EnsembleRetriever for exactly ONE dataset. Used both
    for an explicit hard --dataset filter and as one branch of the fan-out
    below -- same underlying building block either way."""
    dense_kwargs = {"k": k, "filter": {"dataset": dataset_filter}}
    dense_retriever = vectordb.as_retriever(search_kwargs=dense_kwargs)

    bm25_chunks = [c for c in chunks if c.metadata.get("dataset") == dataset_filter]
    if not bm25_chunks:
        # nothing indexed for this dataset -- dense-only rather than
        # letting BM25Retriever error out on an empty document list
        return dense_retriever
    bm25 = BM25Retriever.from_documents(bm25_chunks, preprocess_func=_bm25_preprocess)
    bm25.k = k

    return EnsembleRetriever(retrievers=[dense_retriever, bm25], weights=[0.6, 0.4])


def build_hybrid_retriever(vectordb, chunks, k: int = 8, dataset_filter: str = None,
                            boost_dataset: str = None, boost_factor: float = 2.0):
    """See module docstring for the three modes (hard filter / boosted
    fan-out / equal fan-out). Returns a single retriever either way, so
    everything downstream (query expansion, reranking) is unaffected by
    which mode built it."""
    if dataset_filter:
        return _build_single_dataset_hybrid(vectordb, chunks, k, dataset_filter)

    # Each dataset gets the FULL k as its baseline budget, not k // 2 --
    # FAISS/BM25 retrieval is cheap, and a thin per-dataset budget is
    # exactly the wrong place to save compute when the actual goal is "the
    # correct answer surfaces regardless of which of the 6 datasets it's
    # actually in." A weak per-dataset budget risks the true match never
    # even reaching the candidate pool the reranker/MoE gets to see.
    baseline_k = k
    sub_retrievers = []
    weights = []
    datasets_included = []
    for ds in ALL_DATASETS:
        if not any(c.metadata.get("dataset") == ds for c in chunks):
            continue  # nothing indexed for this dataset -- nothing to search
        is_boosted = ds == boost_dataset
        this_k = int(baseline_k * boost_factor) if is_boosted else baseline_k
        sub_retrievers.append(_build_single_dataset_hybrid(vectordb, chunks, this_k, ds))
        weights.append(boost_factor if is_boosted else 1.0)
        datasets_included.append(ds)

    if not sub_retrievers:
        # nothing indexed at all -- fall back to a flat unfiltered search
        # rather than erroring on an empty EnsembleRetriever
        return vectordb.as_retriever(search_kwargs={"k": k})

    total = sum(weights)
    weights = [w / total for w in weights]
    print(f"[retrieval] fan-out across {datasets_included} "
          f"(boosted: {boost_dataset if boost_dataset in datasets_included else 'none'})")
    return EnsembleRetriever(retrievers=sub_retrievers, weights=weights)


def build_query_expansion_retriever(base_retriever, expansion_model: str = "openai/gpt-oss-20b"):
    # NOTE: llama-3.1-8b-instant was deprecated by Groq on June 17, 2026.
    # openai/gpt-oss-20b is Groq's own recommended replacement for it
    # (confirmed via console.groq.com/docs/deprecations). If this breaks
    # again in the future, check that page directly -- Groq's free-tier
    # model catalog churns fairly often (their own docs note models get
    # swapped out roughly every few months), so pinning one model name in
    # code will periodically need this same kind of update regardless of
    # which specific model is chosen.
    llm = ChatGroq(model=expansion_model, temperature=0)
    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)


def build_reranking_retriever(expanded_retriever, top_n: int = 6):
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=expanded_retriever)


def build_full_retriever(vectordb, chunks, dataset_filter: str = None,
                          boost_dataset: str = None, boost_factor: float = 2.0):
    """Full stack: hybrid (fair fan-out / boosted fan-out / hard scope,
    see build_hybrid_retriever) -> query expansion -> cross-encoder rerank.
    This is what feeds the MoE fusion + extractive reader downstream."""
    hybrid = build_hybrid_retriever(vectordb, chunks, dataset_filter=dataset_filter,
                                     boost_dataset=boost_dataset, boost_factor=boost_factor)
    expanded = build_query_expansion_retriever(hybrid)
    reranked = build_reranking_retriever(expanded)
    return reranked