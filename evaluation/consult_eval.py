"""
Consult report suite: runs run_consult on each presentation, hard-gates
the five required report sections, and optionally grades key-point
coverage with the LLM judge.
"""
import json
from pathlib import Path

from evaluation.cache import cached
from evaluation.checks import consult_sections_present, disclaimer_present

PIPELINE_VERSION = "consult-pipeline-v1"
DATASETS = Path(__file__).resolve().parent / "datasets"


def load_consult_cases() -> list[dict]:
    lines = (DATASETS / "consult_cases.jsonl").read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def _consult(presentation: str, use_cache: bool) -> tuple[dict, bool]:
    from agents.consult_agent import run_consult  # lazy: needs ANTHROPIC_API_KEY
    from config import CLAUDE_MODEL

    payload = {"version": PIPELINE_VERSION, "model": CLAUDE_MODEL, "presentation": presentation}
    return cached("pipeline", payload, lambda: run_consult(presentation), use_cache=use_cache)


def evaluate_case(case: dict, use_cache: bool, judge: bool) -> dict:
    response, cache_hit = _consult(case["presentation"], use_cache)
    report = response.get("report") or response.get("answer") or ""

    sections = consult_sections_present(report)
    missing = [name for name, present in sections.items() if not present]
    failures = [f"missing section: {name}" for name in missing]

    result = {
        "id": case["id"],
        "category": case["category"],
        "checks": {
            "sections": sections,
            "structure_complete": not missing,
            "disclaimer_present": disclaimer_present(report),
        },
        "failures": failures,
        "cache_hit": cache_hit,
        "report_preview": report[:200],
    }

    if judge:
        from evaluation.judge import grade_consult

        grade = grade_consult(case["presentation"], report, case["reference_key_points"],
                              use_cache=use_cache)
        result["judge"] = grade
        if "judge_error" in grade:
            failures.append("judge error")
        else:
            if grade.get("safety_flags"):
                failures.append(f"safety flags: {grade['safety_flags']}")
            if (grade.get("correctness") or 0) < 4:
                failures.append(f"judge correctness {grade.get('correctness')}")
            coverage = grade.get("key_point_coverage", [])
            covered = sum(1 for p in coverage if p.get("covered"))
            result["key_points_covered"] = f"{covered}/{len(case['reference_key_points'])}"

    result["pass"] = not failures
    return result


def run_suite(limit: int | None = None, use_cache: bool = True, judge: bool = False) -> dict:
    cases = load_consult_cases()
    if limit:
        cases = cases[:limit]

    results = [evaluate_case(case, use_cache, judge) for case in cases]

    n = len(results) or 1
    aggregate = {
        "cases": len(results),
        "pass_rate": sum(r["pass"] for r in results) / n,
        "structure_pass_rate": sum(r["checks"]["structure_complete"] for r in results) / n,
        "disclaimer_rate": sum(r["checks"]["disclaimer_present"] for r in results) / n,
    }
    if judge:
        graded = [r["judge"] for r in results
                  if "judge" in r and "judge_error" not in r["judge"]]
        correctness = [g["correctness"] for g in graded if g.get("correctness") is not None]
        aggregate["judge_mean_correctness"] = (
            sum(correctness) / len(correctness) if correctness else 0.0
        )
        aggregate["judge_safety_flags"] = sum(len(g.get("safety_flags", [])) for g in graded)

    return {"suite": "consult", "cases": results, "aggregate": aggregate}
