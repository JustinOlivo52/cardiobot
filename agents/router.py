from utils.logger import get_logger

logger = get_logger(__name__)

import re

# Phrases that signal each tool. Kept narrow on purpose: general questions
# about drugs or rhythms should still get a guideline-grounded answer.
ROUTING_RULES = {
    "dosing": [
        "dose", "doses", "dosing", "how much", "mg", "mcg", "mg/kg",
        "units/kg", "kg", "calculate", "infusion rate"
    ],
    "ekg": [
        "upload", "this ekg", "this ecg", "my ekg", "my ecg", "strip",
        "read this", "interpret this", "ekg image", "ecg image"
    ],
    "citation": [
        "source", "sources", "citation", "cite", "reference", "show me where",
        "where does", "which page", "what page", "guideline says"
    ],
    "rag": [
        # Default — anything clinical that doesn't match above
    ]
}

def _matches(keyword: str, text: str) -> bool:
    # Whole-word match, so "kg" doesn't fire on "background"
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text) is not None

def route_query(query: str) -> str:
    """
    Determine which tool to use based on the user query.
    Returns: 'dosing', 'ekg', 'citation', or 'rag'
    """
    query_lower = query.lower()

    # Check dosing keywords
    for keyword in ROUTING_RULES["dosing"]:
        if _matches(keyword, query_lower):
            logger.info(f"Routing to: DOSING (matched '{keyword}')")
            return "dosing"

    # Check EKG keywords
    for keyword in ROUTING_RULES["ekg"]:
        if _matches(keyword, query_lower):
            logger.info(f"Routing to: EKG (matched '{keyword}')")
            return "ekg"

    # Check citation keywords
    for keyword in ROUTING_RULES["citation"]:
        if _matches(keyword, query_lower):
            logger.info(f"Routing to: CITATION (matched '{keyword}')")
            return "citation"

    # Default to RAG
    logger.info("Routing to: RAG (default)")
    return "rag"
