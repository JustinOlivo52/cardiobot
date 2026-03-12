import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "llama3")

GEMINI_EMBED_MODEL  = "text-embedding-004"
GEMINI_VISION_MODEL = "gemini-1.5-flash"
CLAUDE_MODEL        = "claude-3-5-sonnet-20241022"
GPT4_MODEL          = "gpt-4o"

RETRIEVAL_TOP_K     = int(os.getenv("RETRIEVAL_TOP_K", "3"))
RETRIEVAL_MIN_SCORE = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.3"))
CHUNK_SIZE          = 1000
CHUNK_OVERLAP       = 200

CHROMA_PERSIST_DIR  = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION   = "cardiobot_docs"

APP_TITLE     = "🫀 CardioBot"
APP_SUBTITLE  = "AI-Powered Clinical Cardiology Assistant"
DISCLAIMER    = (
    "⚠️ For educational use only. "
    "CardioBot is not a substitute for clinical judgment. "
    "Always confirm AI interpretations with a supervising physician."
)

def check_required_keys():
    return {
        "gemini":  bool(GOOGLE_API_KEY),
        "claude":  bool(ANTHROPIC_API_KEY),
        "openai":  bool(OPENAI_API_KEY),
        "ollama":  False,
    }

def get_missing_required_keys():
    missing = []
    if not GOOGLE_API_KEY:     missing.append("GOOGLE_API_KEY")
    if not ANTHROPIC_API_KEY:  missing.append("ANTHROPIC_API_KEY")
    if not OPENAI_API_KEY:     missing.append("OPENAI_API_KEY")
    return missing
