from tools.embedder import query_vector_store
from utils.logger import get_logger

logger = get_logger(__name__)

def retrieve_context(query: str, n_results: int = 3) -> tuple[str, list[dict]]:
    """
    Search ChromaDB for relevant chunks and format them as context.
    Returns a tuple of (formatted_context_string, raw_hits)
    """
    hits = query_vector_store(query, n_results=n_results)

    if not hits:
        return "No relevant context found.", []

    context_parts = []
    for i, hit in enumerate(hits):
        context_parts.append(
            f"[Source {i+1}: {hit['source']} | Relevance: {hit['score']}]\n{hit['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)
    logger.info(f"Retrieved {len(hits)} chunks for query: {query[:50]}...")
    return context, hits
