import logging, sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    return logger

class CardioAPIError(Exception): pass
class CardioRetrievalError(Exception): pass
class CardioImageError(Exception): pass
class CardioPDFError(Exception): pass

def friendly_error(e: Exception, feature: str) -> str:
    """Turn a raw provider error into a message that's safe to show in the UI."""
    logging.getLogger("cardiobot.errors").error(f"{feature} failed: {e}")
    msg = str(e).lower()
    if any(k in msg for k in ("quota", "rate limit", "429", "usage limit", "credit balance", "resource_exhausted")):
        reason = "the AI provider's usage limit was reached"
    elif any(k in msg for k in ("api key", "401", "authentication", "permission")):
        reason = "an API key was rejected"
    elif any(k in msg for k in ("connection", "timeout", "timed out", "503", "502", "overloaded")):
        reason = "the AI provider is temporarily unreachable"
    else:
        reason = "an unexpected error occurred"
    return f"{feature} is temporarily unavailable because {reason}. Please try again in a moment."
