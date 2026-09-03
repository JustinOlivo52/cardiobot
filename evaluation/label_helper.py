"""
Ground-truth labeling helper.

Two modes, both printing chunk_index, covered pages, similarity, and an
excerpt so expected_pages labels can be verified against the actual PDF:

  python -m evaluation.label_helper "door-to-balloon time"
      Embeds the query and shows the top-10 retrieved chunks
      (requires OPENAI_API_KEY; costs a fraction of a cent).

  python -m evaluation.label_helper --grep "door-to-balloon"
      Free keyword search over all 947 chunk texts via the Chroma sqlite,
      no API key needed. Use this to find relevant pages the retriever
      MISSES, which mitigates labeling-from-retrieval bias.

Always cross-check the page numbers against the source PDF before
committing them to qa_cases.jsonl.
"""
import argparse
import re
import sqlite3

from evaluation.page_map import CHROMA_SQLITE, load_page_map, pages_for_chunk_text


def _print_hit(chunk_index, pages, score, text):
    excerpt = re.sub(r"\s+", " ", text or "")[:400]
    score_str = f"score {score:.3f}" if score is not None else "keyword match"
    print(f"--- chunk {chunk_index} | pages {sorted(pages)} | {score_str}")
    print(f"    {excerpt}\n")


def grep_chunks(terms: list[str], limit: int = 10):
    page_map = load_page_map()
    con = sqlite3.connect(CHROMA_SQLITE)
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

    shown = 0
    for chunk_index, text in rows:
        lowered = (text or "").lower()
        if all(t.lower() in lowered for t in terms):
            _print_hit(chunk_index, page_map.get(str(chunk_index), []), None, text)
            shown += 1
            if shown >= limit:
                break
    if shown == 0:
        print(f"No chunk contains all of: {terms}")


def retrieve_chunks(query: str, limit: int = 10):
    from tools.embedder import query_vector_store  # lazy: needs OPENAI_API_KEY

    page_map = load_page_map()
    hits = query_vector_store(query, n_results=limit)
    for hit in hits:
        pages = pages_for_chunk_text(hit["text"], page_map, hit.get("chunk_index"))
        _print_hit(hit.get("chunk_index", "?"), pages, hit["score"], hit["text"])


def main():
    parser = argparse.ArgumentParser(description="Inspect chunks to label retrieval ground truth")
    parser.add_argument("query", nargs="+", help="Search query (or terms with --grep)")
    parser.add_argument("--grep", action="store_true",
                        help="Free keyword search instead of embedding retrieval")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if args.grep:
        grep_chunks(args.query, args.limit)
    else:
        retrieve_chunks(" ".join(args.query), args.limit)


if __name__ == "__main__":
    main()
