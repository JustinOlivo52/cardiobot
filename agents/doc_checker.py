from tools.embedder import query_vector_store
from utils.logger import get_logger

logger = get_logger(__name__)

CONFIDENCE_THRESHOLDS = {
    "high":   0.85,
    "medium": 0.70,
    "low":    0.50
}

def get_confidence_label(score: float) -> str:
    if score >= CONFIDENCE_THRESHOLDS["high"]:
        return "🟢 High"
    elif score >= CONFIDENCE_THRESHOLDS["medium"]:
        return "🟡 Medium"
    elif score >= CONFIDENCE_THRESHOLDS["low"]:
        return "🔴 Low"
    else:
        return "⚫ Very Low"

def check_citations(query: str, n_results: int = 5) -> list[dict]:
    """
    Search ChromaDB and return detailed citation information
    for a given query.
    """
    hits = query_vector_store(query, n_results=n_results)

    citations = []
    for i, hit in enumerate(hits):
        # Extract page number from text if present
        page_num = "Unknown"
        text = hit["text"]
        if "[Page " in text:
            try:
                page_start = text.index("[Page ") + 6
                page_end = text.index("]", page_start)
                page_num = text[page_start:page_end]
            except ValueError:
                pass

        # Try to extract section heading from chunk
        lines = text.strip().split("\n")
        heading = "General Content"
        for line in lines[:5]:
            line = line.strip()
            if len(line) > 10 and len(line) < 100 and line[0].isupper():
                heading = line
                break

        citations.append({
            "rank": i + 1,
            "source": hit["source"],
            "page": page_num,
            "score": hit["score"],
            "confidence": get_confidence_label(hit["score"]),
            "heading": heading,
            "excerpt": text[:300] + "..." if len(text) > 300 else text
        })

    logger.info(f"Found {len(citations)} citations for: {query[:50]}")
    return citations
