from agents.claude_client import ask_claude
from agents.rag_agent import retrieve_context
from prompts.clinical_prompts import CARDIOBOT_SYSTEM_PROMPT, build_rag_prompt
from utils.logger import get_logger, CardioAPIError

logger = get_logger(__name__)

def ask_cardiobot(question: str, conversation_history: list[dict] = None) -> dict:
    """
    Full RAG pipeline with conversation memory.
    - Retrieves relevant context from ChromaDB
    - Builds prompt with context
    - Sends full conversation history to Claude
    """
    try:
        # Step 1: Retrieve relevant context
        context, hits = retrieve_context(question)

        # Step 2: Build the current user message with context injected
        user_prompt = build_rag_prompt(question, context)

        # Step 3: Build messages array with history + current question
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_prompt})

        # Step 4: Ask Claude
        logger.info(f"Sending to Claude with {len(messages)} messages in history")
        answer = ask_claude(CARDIOBOT_SYSTEM_PROMPT, messages, max_tokens=4096, effort="low")
        logger.info("Claude responded successfully")

        return {
            "answer": answer,
            "sources": [h["source"] for h in hits],
            "scores": [h["score"] for h in hits],
            "context": context
        }

    except Exception as e:
        raise CardioAPIError(f"CardioBot failed: {e}")
