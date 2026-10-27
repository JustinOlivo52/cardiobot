from config import CLAUDE_MODEL, VISION_MODEL, EMBED_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

# Each check makes the smallest billable call the provider offers. Listing
# models isn't enough: it succeeds even when the account is out of credits.

def _check_openai():
    from tools.embedder import client
    client.embeddings.create(model=EMBED_MODEL, input="ping")

def _check_claude():
    from agents.claude_client import client
    client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": "ping"}]
    )

def _check_gemini():
    # count_tokens validates the key without spending the free tier's small
    # daily generation quota, which the EKG reader needs
    from agents.ekg_agent import client
    client.models.count_tokens(model=VISION_MODEL, contents="ping")

CHECKS = {
    "Claude": ("Consult, Q&A, EKG, dosing backup", _check_claude),
    "OpenAI": ("Semantic search, dosing", _check_openai),
    "Gemini": ("EKG backup (key check only)", _check_gemini),
}

def run_health_checks() -> dict:
    """Returns {provider: (ok, detail)} for each AI provider."""
    from utils.logger import friendly_error
    results = {}
    for name, (used_for, check) in CHECKS.items():
        try:
            check()
            results[name] = (True, used_for)
        except Exception as e:
            reason = friendly_error(e, name).split("because ", 1)[-1].split(".")[0]
            results[name] = (False, reason)
    return results
