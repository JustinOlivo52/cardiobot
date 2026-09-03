"""
Chunk-index -> guideline-page mapping for retrieval ground truth.

Chunks were created by prepending a "[Page N]" marker to each PDF page and
then splitting the joined text into 1000-char chunks, so many chunks carry
no marker at all (they were cut mid-page). This module reconstructs each
chunk's page coverage by walking chunks in chunk_index order and carrying
the last seen marker forward.

The map is committed at evaluation/datasets/page_map.json so retrieval
evals and dataset validation need neither ChromaDB nor an API key. It
reads the Chroma sqlite file directly with stdlib sqlite3. Rebuild after
re-ingesting documents:

    python -m evaluation.page_map
"""
import json
import sqlite3
from pathlib import Path

from evaluation.checks import PAGE_MARKER_RE

REPO_ROOT = Path(__file__).resolve().parents[1]
CHROMA_SQLITE = REPO_ROOT / "chroma_db" / "chroma.sqlite3"
PAGE_MAP_PATH = Path(__file__).resolve().parent / "datasets" / "page_map.json"


def build_page_map(sqlite_path: Path = CHROMA_SQLITE) -> dict[str, list[int]]:
    """Map each chunk_index (as str, JSON-friendly) to the pages it covers."""
    con = sqlite3.connect(sqlite_path)
    try:
        rows = con.execute(
            """
            SELECT idx.int_value, doc.string_value
            FROM embedding_metadata idx
            JOIN embedding_metadata doc ON doc.id = idx.id AND doc.key = 'chroma:document'
            WHERE idx.key = 'chunk_index'
            ORDER BY idx.int_value
            """
        ).fetchall()
    finally:
        con.close()

    page_map: dict[str, list[int]] = {}
    carried_page = None
    for chunk_index, text in rows:
        markers = [int(m) for m in PAGE_MARKER_RE.findall(text or "")]
        pages = set(markers)
        if carried_page is not None:
            pages.add(carried_page)
        page_map[str(chunk_index)] = sorted(pages)
        if markers:
            carried_page = max(markers)
    return page_map


def load_page_map(path: Path = PAGE_MAP_PATH) -> dict[str, list[int]]:
    return json.loads(path.read_text())


def pages_for_chunk_text(text: str, page_map: dict[str, list[int]], chunk_index=None) -> set[int]:
    """Pages a retrieved hit covers: exact via chunk_index, else inline markers."""
    if chunk_index is not None and str(chunk_index) in page_map:
        return set(page_map[str(chunk_index)])
    return {int(m) for m in PAGE_MARKER_RE.findall(text or "")}


if __name__ == "__main__":
    page_map = build_page_map()
    PAGE_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGE_MAP_PATH.write_text(json.dumps(page_map, indent=0))
    all_pages = {p for pages in page_map.values() for p in pages}
    print(f"Wrote {PAGE_MAP_PATH}: {len(page_map)} chunks covering pages "
          f"{min(all_pages)}-{max(all_pages)}")
