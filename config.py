import os

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
CLAUDE_MODEL      = "claude-sonnet-4-5"
GPT4_MODEL        = "gpt-4o"
VISION_MODEL      = "gemini-2.5-flash"

# ── App Settings ──────────────────────────────────────────────────
APP_TITLE         = "CardioBot"
APP_SUBTITLE      = "AI-Powered Clinical Cardiology Assistant"
DISCLAIMER        = "CardioBot is for educational purposes only and is not a substitute for clinical judgment."

# ── ChromaDB Settings ─────────────────────────────────────────────
CHROMA_PERSIST_DIR  = "./chroma_db"
CHROMA_COLLECTION   = "cardiobot_docs"

def check_required_keys() -> dict:
    return {
        "anthropic": bool(ANTHROPIC_API_KEY),
        "gemini":    bool(GOOGLE_API_KEY),
        "openai":    bool(OPENAI_API_KEY),
        "ollama":    False
    }
