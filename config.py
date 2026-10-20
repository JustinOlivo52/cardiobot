import os

try:
    from dotenv import load_dotenv
    load_dotenv()  # Local runs read keys from .env; Streamlit Cloud uses st.secrets
except ImportError:
    pass

try:
    import streamlit as st
    def get_secret(key: str) -> str:
        try:
            return st.secrets[key]
        except Exception:
            return os.getenv(key, "")
except Exception:
    def get_secret(key: str) -> str:
        return os.getenv(key, "")

# ── API Keys ─────────────────────────────────────────────────────
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")
GOOGLE_API_KEY    = get_secret("GOOGLE_API_KEY")
OPENAI_API_KEY    = get_secret("OPENAI_API_KEY")

# ── Model Names ───────────────────────────────────────────────────
CLAUDE_MODEL      = "claude-sonnet-5-5"
GPT4_MODEL        = "gpt-4o"
VISION_MODEL      = "gemini-2.5-flash"
EMBED_MODEL       = "text-embedding-3-small"

# ── API Reliability ───────────────────────────────────────────────
API_TIMEOUT_SECONDS = 60   # Fail fast instead of hanging the UI
API_MAX_RETRIES     = 2    # Retries transient errors (network, 429, 5xx)

# ── App Settings ──────────────────────────────────────────────────
APP_TITLE         = "CardioBot"
APP_SUBTITLE      = "AI-Powered Clinical Cardiology Assistant"
DISCLAIMER        = "CardioBot is for educational purposes only and is not a substitute for clinical judgment."

# ── ChromaDB Settings ─────────────────────────────────────────────
CHROMA_PERSIST_DIR  = "./chroma_db"
CHROMA_COLLECTION   = "cardiobot_docs"
