"""
Run this script to load PDFs into ChromaDB.
Usage: python ingest.py
"""
import os
from tools.retriever import load_pdf, chunk_text
from tools.embedder import build_vector_store
from utils.logger import get_logger

logger = get_logger("ingest")

PDF_DIR = "data/sample_docs"

def ingest_all():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        logger.warning(f"No PDFs found in {PDF_DIR}. Add your PDFs there first!")
        return

    logger.info(f"Found {len(pdf_files)} PDFs: {pdf_files}")

    for pdf_file in pdf_files:
        path = os.path.join(PDF_DIR, pdf_file)
        source_name = pdf_file.replace(".pdf", "").replace(" ", "_")

        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {pdf_file}")

        text = load_pdf(path)
        chunks = chunk_text(text)
        build_vector_store(chunks, source_name)

    logger.info("\n🎉 All PDFs ingested! ChromaDB is ready.")

if __name__ == "__main__":
    ingest_all()
