"""
Retrieval quality suite: hit_rate@k, page_recall@k, and MRR against
page-level ground truth (expected_pages in qa_cases.jsonl).

Costs only one query embedding per case (~fractions of a cent for the
full dataset). k=3 is the headline metric because ask_cardiobot feeds
exactly the top 3 chunks to Claude; the k-curve up to 10 shows whether
misses are ranking problems (found at k=10) or corpus problems (never
found).
"""
import json
from pathlib import Path

from evaluation.page_map import load_page_map, pages_for_chunk_text

K_VALUES = [1, 3, 5, 10]
DATASETS = Path(__file__).resolve().parent / "datasets"


def load_qa_cases() -> list[dict]:
    lines = (DATASETS / "qa_cases.jsonl").read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def evaluate_case(case: dict, hits: list[dict], page_map: dict) -> dict:
    expected = set(case["expected_pages"])
    hit_pages = [pages_for_chunk_text(h["text"], page_map, h.get("chunk_index")) for h in hits]

    relevant_ranks = [i + 1 for i, pages in enumerate(hit_pages) if pages & expected]
    mrr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0

    metrics = {"mrr": mrr}
    for k in K_VALUES:
        top_k_pages = set().union(*hit_pages[:k]) if hit_pages[:k] else set()
        metrics[f"hit_rate@{k}"] = 1.0 if top_k_pages & expected else 0.0
        metrics[f"page_recall@{k}"] = len(top_k_pages & expected) / len(expected) if expected else 0.0

    return {
        "id": case["id"],
        "category": case["category"],
        "metrics": metrics,
        "pass": metrics["hit_rate@3"] == 1.0,
        "first_relevant_rank": relevant_ranks[0] if relevant_ranks else None,
        "top_scores": [h["score"] for h in hits[:3]],
    }


def run_suite(limit: int | None = None) -> dict:
    from tools.embedder import query_vector_store  # lazy: needs OPENAI_API_KEY

    page_map = load_page_map()
    cases = [c for c in load_qa_cases() if c["case_type"] == "standard"]
    if limit:
        cases = cases[:limit]

    results = []
    for case in cases:
        hits = query_vector_store(case["question"], n_results=max(K_VALUES))
        results.append(evaluate_case(case, hits, page_map))

    n = len(results) or 1
    aggregate = {"cases": len(results), "mrr": sum(r["metrics"]["mrr"] for r in results) / n}
    for k in K_VALUES:
        aggregate[f"hit_rate@{k}"] = sum(r["metrics"][f"hit_rate@{k}"] for r in results) / n
        aggregate[f"page_recall@{k}"] = sum(r["metrics"][f"page_recall@{k}"] for r in results) / n

    return {"suite": "retrieval", "cases": results, "aggregate": aggregate}
