from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import get_logger, CardioPDFError

logger = get_logger(__name__)

def load_pdf(path: str) -> str:
    """Extract all text from a PDF file."""
    try:
        reader = PdfReader(path)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n[Page {i+1}]\n{page_text}"
        logger.info(f"Loaded {len(reader.pages)} pages from {path}")
        return text
    except Exception as e:
        raise CardioPDFError(f"Failed to load PDF {path}: {e}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    logger.info(f"Split into {len(chunks)} chunks")
    return chunks
