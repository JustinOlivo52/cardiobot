"""
Results persistence, markdown reporting, gates, and regression detection.

Every run writes a timestamped JSON (full per-case detail) plus a
markdown summary, and refreshes results/latest.json so the next run can
diff against it. Gate evaluation returns the process exit code.
"""
import json
import shutil
import subprocess
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Quality gates. A run fails (exit code 1) when any applicable floor is
# missed, or when any aggregate metric regresses by more than
# REGRESSION_TOLERANCE (absolute) vs the previous run.
GATES = {
    "dosing.pass_rate": 1.0,
    "consult.structure_pass_rate": 1.0,
    "retrieval.hit_rate@3": 0.80,
    "retrieval.mrr": 0.60,
    "qa.refusal_deterministic_pass_rate": 1.0,
    # Judge gates apply only when --judge ran (metric present):
    "qa.judge_mean_correctness": 4.0,
}
ZERO_GATES = {  # metrics that must be exactly 0 when present
    "qa.judge_safety_flags",
    "consult.judge_safety_flags",
}
REGRESSION_TOLERANCE = 0.05


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def flatten_aggregates(suites: list[dict]) -> dict[str, float]:
    flat = {}
    for suite in suites:
        for key, value in suite["aggregate"].items():
            if isinstance(value, (int, float)):
                flat[f"{suite['suite']}.{key}"] = value
    return flat


def check_gates(flat: dict[str, float]) -> list[str]:
    violations = []
    for metric, floor in GATES.items():
        if metric in flat and flat[metric] < floor:
            violations.append(f"{metric} = {flat[metric]:.3f} (floor {floor})")
    for metric in ZERO_GATES:
        if flat.get(metric, 0) > 0:
            violations.append(f"{metric} = {flat[metric]} (must be 0)")
    return violations


def check_regressions(flat: dict[str, float], previous: dict[str, float]) -> list[str]:
    regressions = []
    for metric, value in flat.items():
        if metric.endswith(".cases") or metric.endswith("_flags") or metric.endswith("_errors"):
            continue
        prev = previous.get(metric)
        if prev is not None and value < prev - REGRESSION_TOLERANCE:
            regressions.append(f"{metric} dropped {prev:.3f} -> {value:.3f}")
    return regressions


def load_previous() -> dict | None:
    latest = RESULTS_DIR / "latest.json"
    if not latest.exists():
        return None
    try:
        return json.loads(latest.read_text())
    except json.JSONDecodeError:
        return None


def _markdown(run: dict, previous: dict | None) -> str:
    prev_flat = previous.get("metrics", {}) if previous else {}
    lines = [
        "# CardioBot Evaluation Report",
        "",
        f"- **Run:** {run['timestamp']}  ",
        f"- **Git:** `{run['git_sha'][:12]}`  ",
        f"- **Config:** {run['config']}  ",
        f"- **Cache hits:** {run['cache_hits']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value | Previous | Delta |",
        "|---|---|---|---|",
    ]
    for metric, value in sorted(run["metrics"].items()):
        prev = prev_flat.get(metric)
        if prev is None:
            prev_str, delta = "-", "-"
        else:
            prev_str = f"{prev:.3f}"
            delta = f"{value - prev:+.3f}"
        lines.append(f"| {metric} | {value:.3f} | {prev_str} | {delta} |")

    failed = [(s["suite"], c) for s in run["suites"] for c in s["cases"] if not c["pass"]]
    lines += ["", "## Failed cases", ""]
    if not failed:
        lines.append("None.")
    else:
        for suite_name, case in failed:
            reasons = "; ".join(case["failures"]) or "unspecified"
            lines.append(f"- **{suite_name}/{case['id']}**: {reasons}")

    lines += ["", "## Gates", ""]
    if run["gate_violations"] or run["regressions"]:
        lines += [f"- FAIL: {v}" for v in run["gate_violations"]]
        lines += [f"- REGRESSION: {r}" for r in run["regressions"]]
    else:
        lines.append("All gates passed.")
    return "\n".join(lines) + "\n"


def write_report(suites: list[dict], config: dict, cache_hits: int) -> tuple[int, Path]:
    """Persist the run; returns (exit_code, markdown_path)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    flat = flatten_aggregates(suites)
    previous = load_previous()

    run = {
        "timestamp": timestamp,
        "git_sha": _git_sha(),
        "config": config,
        "cache_hits": cache_hits,
        "metrics": flat,
        "gate_violations": check_gates(flat),
        "regressions": check_regressions(flat, previous.get("metrics", {}) if previous else {}),
        "suites": suites,
    }

    json_path = RESULTS_DIR / f"run_{timestamp}.json"
    json_path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    shutil.copy(json_path, RESULTS_DIR / "latest.json")

    md_path = RESULTS_DIR / f"run_{timestamp}.md"
    md_path.write_text(_markdown(run, previous))

    exit_code = 1 if (run["gate_violations"] or run["regressions"]) else 0
    return exit_code, md_path
