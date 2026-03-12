from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from agents.router import route_query
from agents.claude_agent import ask_cardiobot
from agents.doc_checker import check_citations
from utils.logger import get_logger

logger = get_logger(__name__)

# ── State Definition ─────────────────────────────────────────────
class CardioState(TypedDict):
    query: str
    route: Optional[str]
    response: Optional[dict]
    conversation_history: Optional[list]
    error: Optional[str]

# ── Node Functions ───────────────────────────────────────────────
def router_node(state: CardioState) -> CardioState:
    """Decide which tool to use."""
    route = route_query(state["query"])
    return {**state, "route": route}

def rag_node(state: CardioState) -> CardioState:
    """Handle clinical Q&A with RAG + Claude."""
    try:
        result = ask_cardiobot(
            state["query"],
            conversation_history=state.get("conversation_history", [])
        )
        return {**state, "response": {"type": "rag", **result}}
    except Exception as e:
        return {**state, "error": str(e)}

def citation_node(state: CardioState) -> CardioState:
    """Handle citation search requests."""
    try:
        citations = check_citations(state["query"], n_results=5)
        return {**state, "response": {"type": "citation", "citations": citations}}
    except Exception as e:
        return {**state, "error": str(e)}

def dosing_node(state: CardioState) -> CardioState:
    """Handle drug dosing requests - prompts user for more info if needed."""
    # For the graph we return a prompt asking for weight
    # The actual calculation happens in the UI
    return {**state, "response": {
        "type": "dosing_redirect",
        "message": "💊 I detected a dosing question! Please use the **Drug Calculator** tab for accurate weight-based dosing calculations."
    }}

def ekg_node(state: CardioState) -> CardioState:
    """Handle EKG interpretation requests."""
    return {**state, "response": {
        "type": "ekg_redirect",
        "message": "📊 I detected an EKG question! Please use the **EKG Interpreter** tab to upload and analyze your EKG strip."
    }}

def route_decision(state: CardioState) -> str:
    """LangGraph conditional edge — returns which node to go to next."""
    return state.get("route", "rag")

# ── Build the Graph ──────────────────────────────────────────────
def build_cardiobot_graph():
    graph = StateGraph(CardioState)

    # Add nodes
    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("citation", citation_node)
    graph.add_node("dosing", dosing_node)
    graph.add_node("ekg", ekg_node)

    # Set entry point
    graph.set_entry_point("router")

    # Add conditional edges from router
    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag": "rag",
            "citation": "citation",
            "dosing": "dosing",
            "ekg": "ekg"
        }
    )

    # All nodes end after execution
    graph.add_edge("rag", END)
    graph.add_edge("citation", END)
    graph.add_edge("dosing", END)
    graph.add_edge("ekg", END)

    return graph.compile()

# Compile once at import time
cardiobot_graph = build_cardiobot_graph()

def run_cardiobot(query: str, conversation_history: list = None) -> dict:
    """
    Main entry point — run the full LangGraph pipeline.
    """
    initial_state = CardioState(
        query=query,
        route=None,
        response=None,
        conversation_history=conversation_history or [],
        error=None
    )

    logger.info(f"Running CardioBot graph for: {query[:50]}...")
    final_state = cardiobot_graph.invoke(initial_state)

    if final_state.get("error"):
        raise Exception(final_state["error"])

    return final_state["response"]
