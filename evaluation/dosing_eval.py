"""
Dosing suite: deterministic checks of the calculator against the golden
dosing dataset. Free, needs no API keys, and always runs.

With live=True it additionally runs get_dosing_guidance for a few cases
and asserts the GPT narrative does not contradict the calculated dose
(the calculated dose string must appear in the guidance text).
"""
import json
from pathlib import Path

from tools.calculator import calculate_dose

DATASETS = Path(__file__).resolve().parent / "datasets"
LIVE_CASE_IDS = ["dose-001", "dose-002", "dose-008"]


def load_dosing_cases() -> list[dict]:
    return json.loads((DATASETS / "dosing_cases.json").read_text())


def evaluate_case(case: dict) -> dict:
    result = calculate_dose(case["drug"], case["weight_kg"])
    failures = []

    if case["expect_error"]:
        if "error" not in result:
            failures.append("expected an error, got a dose")
    elif "error" in result:
        failures.append(f"unexpected error: {result['error']}")
    elif case["expected_dose"] is None:
        if "calculated_dose" in result:
            failures.append("non-weight-based drug returned a calculated dose")
    else:
        if result.get("calculated_dose") != case["expected_dose"]:
            failures.append(f"dose {result.get('calculated_dose')!r} != {case['expected_dose']!r}")
        if result.get("dose_type") != case["expected_dose_type"]:
            failures.append(f"dose_type {result.get('dose_type')!r} != {case['expected_dose_type']!r}")
        if result.get("dose_capped") != case["expect_capped"]:
            failures.append(f"dose_capped {result.get('dose_capped')!r} != {case['expect_capped']!r}")

    return {"id": case["id"], "drug": case["drug"].strip().lower(),
            "weight_kg": case["weight_kg"], "failures": failures, "pass": not failures}


def evaluate_live_case(case: dict) -> dict:
    from agents.dosing_agent import get_dosing_guidance  # lazy: needs OPENAI_API_KEY

    result = get_dosing_guidance(case["drug"], case["weight_kg"])
    failures = []
    if "error" in result:
        failures.append(f"unexpected error: {result['error']}")
    else:
        guidance = result.get("guidance", "")
        expected = case["expected_dose"]
        # The narrative must restate the calculated number, not contradict it.
        expected_number = expected.split(" ")[0] if expected else ""
        if expected_number and expected_number not in guidance:
            failures.append(f"guidance never states the calculated dose {expected!r}")
    return {"id": f"{case['id']}-live", "drug": case["drug"], "failures": failures,
            "pass": not failures}


def run_suite(limit: int | None = None, live: bool = False) -> dict:
    cases = load_dosing_cases()
    if limit:
        cases = cases[:limit]

    results = [evaluate_case(case) for case in cases]
    if live:
        live_cases = [c for c in cases if c["id"] in LIVE_CASE_IDS]
        results.extend(evaluate_live_case(c) for c in live_cases)

    n = len(results) or 1
    aggregate = {"cases": len(results), "pass_rate": sum(r["pass"] for r in results) / n}
    return {"suite": "dosing", "cases": results, "aggregate": aggregate}
