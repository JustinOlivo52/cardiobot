import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, API_TIMEOUT_SECONDS, API_MAX_RETRIES
from utils.logger import get_logger, CardioAPIError

logger = get_logger(__name__)

# The SDK retries connection errors, 429s, and 5xx with backoff; the timeout
# keeps a hung request from leaving the UI spinning for the default 10 minutes
client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    timeout=API_TIMEOUT_SECONDS,
    max_retries=API_MAX_RETRIES
)

def ask_claude(system: str, messages: list[dict], max_tokens: int = 4096, effort: str = "low") -> str:
    """
    Send a request to Claude and return the text of the reply.
    - Thinking counts toward max_tokens, so leave headroom beyond the answer length
    - Server-side fallback reroutes the request if a safety classifier declines it
    """
    response = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_config={"effort": effort},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default"
    )

    if response.stop_reason == "refusal":
        raise CardioAPIError("Claude declined this request")

    # Read by block type: the response can start with a thinking block
    text = "".join(block.text for block in response.content if block.type == "text")
    if response.stop_reason == "max_tokens":
        logger.warning("Claude response hit max_tokens and may be cut off")
    return text
