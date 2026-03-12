import os
import time
import chromadb
from google import genai
from config import GOOGLE_API_KEY, CHROMA_PERSIST_DIR, CHROMA_COLLECTION
from utils.logger import get_logger, CardioRetrievalError

logger = get_logger(__name__)

EMBED_MODEL = "gemini-embedding-001"

client = genai.Client(api_key=GOOGLE_API_KEY)

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

def get_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )
    return result.embeddings[0].values

def build_vector_store(chunks: list[str], source_name: str):
    try:
        chroma = get_chroma_client()
        collection = chroma.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

        existing = collection.get(where={"source": source_name})
        already_done = len(existing['ids'])
        if already_done > 0:
            logger.info(f"Resuming from chunk {already_done} for {source_name}")

        total = len(chunks)
        logger.info(f"Embedding {total - already_done} remaining chunks from {source_name}...")

        batch_size = 5
        for i in range(already_done, total, batch_size):
            batch = chunks[i:i+batch_size]
            embeddings = [get_embedding(chunk) for chunk in batch]
            ids = [f"{source_name}_chunk_{i+j}" for j in range(len(batch))]
            metadatas = [{"source": source_name, "chunk_index": i+j} for j in range(len(batch))]

            collection.add(
                documents=batch,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            logger.info(f"  [{i+len(batch)}/{total}] chunks embedded")
            time.sleep(0.7)

        logger.info(f"✅ Done! {total} chunks stored for {source_name}")
        return collection

    except Exception as e:
        raise CardioRetrievalError(f"Failed to build vector store: {e}")

def query_vector_store(query: str, n_results: int = 3) -> list[dict]:
    try:
        chroma = get_chroma_client()
        collection = chroma.get_collection(CHROMA_COLLECTION)

        query_embedding = client.models.embed_content(
            model=EMBED_MODEL,
            contents=query
        ).embeddings[0].values

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        hits = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            hits.append({
                "text": doc,
                "source": meta['source'],
                "score": round(1 - dist, 3)
            })

        return hits

    except Exception as e:
        raise CardioRetrievalError(f"Query failed: {e}")
