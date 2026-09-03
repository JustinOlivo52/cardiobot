"""
CardioBot evaluation runner.

    python -m evaluation.run --suite dosing              # free, no API keys
    python -m evaluation.run --suite retrieval           # needs OPENAI_API_KEY
    python -m evaluation.run --suite qa --judge          # needs OPENAI + ANTHROPIC
    python -m evaluation.run --judge                     # everything

Writes evaluation/results/run_<timestamp>.{json,md}, diffs against the
previous run, and exits nonzero on any gate violation or regression.
LLM responses are cached by content hash; a rerun with nothing changed
costs only query embeddings.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# consult retrieves guideline context too, so it needs the embedding key
# and the vector store just like qa does.
SUITE_KEYS = {
    "dosing": [],
    "retrieval": ["OPENAI_API_KEY"],
    "qa": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
    "consult": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
}
CHROMA_SUITES = {"retrieval", "qa", "consult"}


def _check_prerequisites(suites: list[str], live_dosing: bool) -> list[str]:
    problems = []
    needed = {key for s in suites for key in SUITE_KEYS[s]}
    if live_dosing:
        needed.add("OPENAI_API_KEY")
    for key in sorted(needed):
        if not os.getenv(key):
            problems.append(f"missing environment variable {key} "
                            f"(needed by: {', '.join(s for s in suites if key in SUITE_KEYS[s]) or 'live dosing'})")
    if any(s in CHROMA_SUITES for s in suites) and not (REPO_ROOT / "chroma_db").exists():
        problems.append("chroma_db/ not found; run ingest.py first")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CardioBot evaluation suites")
    parser.add_argument("--suite", choices=[*SUITE_KEYS, "all"], default="all")
    parser.add_argument("--judge", action="store_true",
                        help="Grade qa/consult answers with the Claude judge (paid, cached)")
    parser.add_argument("--limit", type=int, default=None, help="Cap cases per suite")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass cache reads (still refreshes the cache)")
    parser.add_argument("--live-dosing", action="store_true",
                        help="Also check GPT dosing narratives against calculated doses")
    args = parser.parse_args()

    # CHROMA_PERSIST_DIR is relative, so run from the repo root.
    os.chdir(REPO_ROOT)

    suites_to_run = list(SUITE_KEYS) if args.suite == "all" else [args.suite]
    problems = _check_prerequisites(suites_to_run, args.live_dosing)
    if problems:
        for p in problems:
            print(f"ERROR: {p}", file=sys.stderr)
        return 2

    use_cache = not args.no_cache
    results = []
    for name in suites_to_run:
        print(f"Running suite: {name} ...")
        if name == "dosing":
            from evaluation.dosing_eval import run_suite
            results.append(run_suite(limit=args.limit, live=args.live_dosing))
        elif name == "retrieval":
            from evaluation.retrieval_eval import run_suite
            results.append(run_suite(limit=args.limit))
        elif name == "qa":
            from evaluation.qa_eval import run_suite
            results.append(run_suite(limit=args.limit, use_cache=use_cache, judge=args.judge))
        elif name == "consult":
            from evaluation.consult_eval import run_suite
            results.append(run_suite(limit=args.limit, use_cache=use_cache, judge=args.judge))
        agg = results[-1]["aggregate"]
        summary = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in agg.items())
        print(f"  {summary}")

    cache_hits = sum(
        1 for suite in results for case in suite["cases"] if case.get("cache_hit")
    )

    from evaluation.report import write_report
    config = {"suite": args.suite, "judge": args.judge, "limit": args.limit,
              "no_cache": args.no_cache, "live_dosing": args.live_dosing}
    exit_code, md_path = write_report(results, config, cache_hits)

    print(f"\nReport: {md_path}")
    print("PASS" if exit_code == 0 else "FAIL (see gates/regressions in report)")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
