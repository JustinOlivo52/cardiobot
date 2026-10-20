import math
import re
from collections import Counter
from functools import lru_cache
import chromadb
from openai import OpenAI
from config import OPENAI_API_KEY, CHROMA_PERSIST_DIR, CHROMA_COLLECTION, EMBED_MODEL
from tools.retriever import assign_pages
from utils.logger import get_logger, CardioRetrievalError

logger = get_logger(__name__)

# One retry and a short timeout: if OpenAI is down, get to the keyword fallback quickly
client = OpenAI(api_key=OPENAI_API_KEY, timeout=15, max_retries=1)

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding

def build_vector_store(chunks: list[str], source_name: str):
    try:
        chroma = get_chroma_client()
        collection = chroma.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

        existing = collection.get(where={"source": source_name})
        already_done = len(existing['ids'])
        if already_done > 0:
            logger.info(f"Resuming from chunk {already_done} for {source_name}")

        total = len(chunks)
        pages = assign_pages(chunks)
        logger.info(f"Embedding {total - already_done} remaining chunks from {source_name}...")

        batch_size = 20
        for i in range(already_done, total, batch_size):
            batch = chunks[i:i+batch_size]
            embeddings = [get_embedding(chunk) for chunk in batch]
            ids = [f"{source_name}_chunk_{i+j}" for j in range(len(batch))]
            metadatas = [
                {"source": source_name, "chunk_index": i+j, "page": pages[i+j] or "Unknown"}
                for j in range(len(batch))
            ]

            collection.add(
                documents=batch,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            logger.info(f"  [{i+len(batch)}/{total}] chunks embedded")

        logger.info(f"✅ Done! {total} chunks stored for {source_name}")
        return collection

    except Exception as e:
        raise CardioRetrievalError(f"Failed to build vector store: {e}")

def query_vector_store(query: str, n_results: int = 3) -> list[dict]:
    """Semantic search via OpenAI embeddings, falling back to keyword search
    if OpenAI is unavailable (quota exhausted, bad key, outage)."""
    try:
        return _semantic_search(query, n_results)
    except Exception as e:
        logger.warning(f"OpenAI embedding search failed, using keyword fallback: {e}")

    try:
        return _keyword_search(query, n_results)
    except Exception as e:
        raise CardioRetrievalError(f"Query failed: {e}")

def _semantic_search(query: str, n_results: int) -> list[dict]:
    chroma = get_chroma_client()
    collection = chroma.get_collection(CHROMA_COLLECTION)

    query_embedding = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    page_map = _load_page_map()
    hits = []
    for chunk_id, doc, meta, dist in zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ):
        hits.append({
            "text": doc,
            "source": meta['source'],
            "page": page_map.get(chunk_id),
            "score": round(1 - dist, 3)
        })

    return hits

# ── Page lookup ───────────────────────────────────────────────────
# Chunks ingested after page metadata was added carry it directly; older
# ones get it derived by walking each source's chunks in order.
@lru_cache(maxsize=1)
def _load_all_chunks():
    collection = get_chroma_client().get_collection(CHROMA_COLLECTION)
    data = collection.get(include=["documents", "metadatas"])
    return data["ids"], data["documents"], data["metadatas"]

@lru_cache(maxsize=1)
def _load_page_map() -> dict:
    ids, docs, metas = _load_all_chunks()
    by_source = {}
    for i in sorted(range(len(ids)), key=lambda i: metas[i].get("chunk_index", 0)):
        by_source.setdefault(metas[i]["source"], []).append(i)

    page_map = {}
    for indexes in by_source.values():
        derived = assign_pages([docs[i] for i in indexes])
        for i, page in zip(indexes, derived):
            stored = metas[i].get("page")
            page_map[ids[i]] = stored if stored and stored != "Unknown" else page
    return page_map

# ── Keyword fallback (BM25 over the stored chunks, no API needed) ──
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())

@lru_cache(maxsize=1)
def _load_keyword_index():
    ids, docs, metas = _load_all_chunks()
    term_counts = [Counter(_tokenize(d)) for d in docs]
    lengths = [sum(tc.values()) for tc in term_counts]
    doc_freq = Counter(t for tc in term_counts for t in tc)
    avg_len = sum(lengths) / max(len(lengths), 1)
    logger.info(f"Keyword index built over {len(docs)} chunks")
    return ids, docs, metas, term_counts, lengths, doc_freq, avg_len

def _keyword_search(query: str, n_results: int, k1: float = 1.5, b: float = 0.75) -> list[dict]:
    ids, docs, metas, term_counts, lengths, doc_freq, avg_len = _load_keyword_index()
    page_map = _load_page_map()
    n_docs = len(docs)
    terms = set(_tokenize(query))

    scores = []
    for i, tc in enumerate(term_counts):
        score = 0.0
        for t in terms:
            f = tc.get(t)
            if not f:
                continue
            idf = math.log(1 + (n_docs - doc_freq[t] + 0.5) / (doc_freq[t] + 0.5))
            score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * lengths[i] / avg_len))
        # Reference-list chunks match lots of keywords but aren't useful passages
        if docs[i].count("doi.org") >= 3:
            score *= 0.2
        scores.append(score)

    ranked = sorted(range(n_docs), key=lambda i: scores[i], reverse=True)[:n_results]
    top = scores[ranked[0]] if ranked and scores[ranked[0]] > 0 else 1.0

    # Scale relative to the best match, capped at 0.8 so keyword hits never
    # show as "High" confidence (those labels are tuned for embedding scores)
    return [{
        "text": docs[i],
        "source": metas[i]["source"],
        "page": page_map.get(ids[i]),
        "score": round(0.8 * scores[i] / top, 3)
    } for i in ranked if scores[i] > 0]
