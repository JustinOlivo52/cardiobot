import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from agents.rag_agent import retrieve_context
from prompts.clinical_prompts import CARDIOBOT_SYSTEM_PROMPT, build_rag_prompt
from utils.logger import get_logger, CardioAPIError

logger = get_logger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def ask_cardiobot(question: str) -> dict:
    """
    Full RAG pipeline: retrieve context → build prompt → ask Claude.
    Returns a dict with answer, sources, and context.
    """
    try:
        # Step 1: Retrieve relevant context
        context, hits = retrieve_context(question)

        # Step 2: Build the prompt
        user_prompt = build_rag_prompt(question, context)

        # Step 3: Ask Claude
        logger.info(f"Sending question to Claude: {question[:50]}...")
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=CARDIOBOT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
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
