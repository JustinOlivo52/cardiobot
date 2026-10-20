import os
import shutil
import pytest

# Fake keys: tests must never depend on (or spend) real API credits
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")
os.environ.setdefault("GOOGLE_API_KEY", "test")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture(scope="session")
def vector_store(tmp_path_factory):
    """A throwaway copy of the bundled ChromaDB, so tests never modify the real one."""
    import tools.embedder as embedder
    copy = tmp_path_factory.mktemp("chroma") / "chroma_db"
    shutil.copytree(os.path.join(REPO_ROOT, "chroma_db"), copy)
    embedder.CHROMA_PERSIST_DIR = str(copy)
    embedder._load_all_chunks.cache_clear()
    embedder._load_page_map.cache_clear()
    embedder._load_keyword_index.cache_clear()
    return embedder

class Unavailable(Exception):
    pass

@pytest.fixture
def openai_down(monkeypatch):
    """Simulate OpenAI being out of quota for both embeddings and chat."""
    import tools.embedder as embedder
    import agents.dosing_agent as dosing_agent

    def fail(*args, **kwargs):
        raise Unavailable("Error code: 429 - insufficient_quota")

    monkeypatch.setattr(embedder.client.embeddings, "create", fail)
    monkeypatch.setattr(dosing_agent.client.chat.completions, "create", fail)
