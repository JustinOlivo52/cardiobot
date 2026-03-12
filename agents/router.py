from utils.logger import get_logger

logger = get_logger(__name__)

# Keywords that signal each tool
ROUTING_RULES = {
    "dosing": [
        "dose", "dosing", "mg", "mcg", "units", "how much", "administer",
        "give", "medication", "drug", "heparin", "amiodarone", "lidocaine",
        "epinephrine", "metoprolol", "atropine", "adenosine", "nitroglycerin",
        "weight", "kg", "calculate"
    ],
    "ekg": [
        "ekg", "ecg", "rhythm", "strip", "interpret", "image", "upload",
        "waveform", "st elevation", "st depression", "afib", "heart block",
        "qrs", "p wave", "t wave", "arrhythmia"
    ],
    "citation": [
        "source", "citation", "reference", "guideline says", "where does",
        "page", "section", "evidence", "class i", "class ii", "level a",
        "level b", "cite", "proof", "show me where"
    ],
    "rag": [
        # Default — anything clinical that doesn't match above
    ]
}

def route_query(query: str) -> str:
    """
    Determine which tool to use based on the user query.
    Returns: 'dosing', 'ekg', 'citation', or 'rag'
    """
    query_lower = query.lower()

    # Check dosing keywords
    for keyword in ROUTING_RULES["dosing"]:
        if keyword in query_lower:
            logger.info(f"Routing to: DOSING (matched '{keyword}')")
            return "dosing"

    # Check EKG keywords
    for keyword in ROUTING_RULES["ekg"]:
        if keyword in query_lower:
            logger.info(f"Routing to: EKG (matched '{keyword}')")
            return "ekg"

    # Check citation keywords
    for keyword in ROUTING_RULES["citation"]:
        if keyword in query_lower:
            logger.info(f"Routing to: CITATION (matched '{keyword}')")
            return "citation"

    # Default to RAG
    logger.info("Routing to: RAG (default)")
    return "rag"
