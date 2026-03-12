import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from agents.rag_agent import retrieve_context
from prompts.clinical_prompts import CARDIOBOT_SYSTEM_PROMPT, build_rag_prompt
from utils.logger import get_logger, CardioAPIError

logger = get_logger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

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
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=CARDIOBOT_SYSTEM_PROMPT,
            messages=messages
        )

        answer = response.content[0].text
        logger.info("Claude responded successfully")

        return {
            "answer": answer,
            "sources": [h["source"] for h in hits],
            "scores": [h["score"] for h in hits],
            "context": context
        }

    except Exception as e:
        raise CardioAPIError(f"CardioBot failed: {e}")
