"""Gate and regression logic tests. Pure functions, no API keys."""
from evaluation.report import check_gates, check_regressions, flatten_aggregates


def test_flatten_aggregates_prefixes_suite_names():
    suites = [
        {"suite": "dosing", "aggregate": {"cases": 18, "pass_rate": 1.0}},
        {"suite": "retrieval", "aggregate": {"hit_rate@3": 0.85}},
    ]
    flat = flatten_aggregates(suites)
    assert flat["dosing.pass_rate"] == 1.0
    assert flat["retrieval.hit_rate@3"] == 0.85


class TestGates:
    def test_all_passing(self):
        assert check_gates({"dosing.pass_rate": 1.0, "retrieval.hit_rate@3": 0.9,
                            "retrieval.mrr": 0.7}) == []

    def test_floor_violation_reported(self):
        violations = check_gates({"retrieval.hit_rate@3": 0.5})
        assert len(violations) == 1
        assert "retrieval.hit_rate@3" in violations[0]

    def test_absent_metric_is_not_gated(self):
        """Judge gates must not fire when --judge did not run."""
        assert check_gates({"dosing.pass_rate": 1.0}) == []

    def test_safety_flags_must_be_zero(self):
        assert check_gates({"qa.judge_safety_flags": 1})
        assert check_gates({"qa.judge_safety_flags": 0}) == []

    def test_dosing_pass_rate_must_be_perfect(self):
        assert check_gates({"dosing.pass_rate": 0.99})


class TestRegressions:
    def test_no_previous_run_is_clean(self):
        assert check_regressions({"qa.pass_rate": 0.5}, {}) == []

    def test_drop_beyond_tolerance_flagged(self):
        regressions = check_regressions({"qa.pass_rate": 0.80}, {"qa.pass_rate": 0.90})
        assert len(regressions) == 1
        assert "qa.pass_rate" in regressions[0]

    def test_small_drop_within_tolerance_ok(self):
        assert check_regressions({"qa.pass_rate": 0.88}, {"qa.pass_rate": 0.90}) == []

    def test_improvement_never_flagged(self):
        assert check_regressions({"qa.pass_rate": 0.99}, {"qa.pass_rate": 0.50}) == []

    def test_counts_are_exempt(self):
        """Running with --limit shrinks case counts; that is not a regression."""
        assert check_regressions({"qa.cases": 3}, {"qa.cases": 30}) == []
