import json

import pytest

import evaluation.cache as cache_mod
from evaluation.cache import cache_key, cached
from evaluation.judge import parse_judge_json, qa_judge_pass


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_ROOT", tmp_path)
    return tmp_path


class TestCache:
    def test_round_trip(self, tmp_cache):
        calls = []

        def fn():
            calls.append(1)
            return {"answer": "42"}

        payload = {"question": "q", "model": "m", "version": "v1"}
        value1, hit1 = cached("pipeline", payload, fn)
        value2, hit2 = cached("pipeline", payload, fn)
        assert value1 == value2 == {"answer": "42"}
        assert (hit1, hit2) == (False, True)
        assert len(calls) == 1

    def test_key_stable_across_ordering(self):
        a = cache_key("ns", {"x": 1, "y": 2})
        b = cache_key("ns", {"y": 2, "x": 1})
        assert a == b

    def test_version_bump_invalidates(self, tmp_cache):
        base = {"question": "q", "model": "m"}
        cached("judge", {**base, "version": "v1"}, lambda: "old")
        value, hit = cached("judge", {**base, "version": "v2"}, lambda: "new")
        assert value == "new"
        assert hit is False

    def test_no_cache_bypasses_read_but_writes(self, tmp_cache):
        payload = {"q": "x", "version": "v1"}
        cached("pipeline", payload, lambda: "first")
        value, hit = cached("pipeline", payload, lambda: "second", use_cache=False)
        assert value == "second"
        assert hit is False
        # The bypass run refreshed the cache
        value3, hit3 = cached("pipeline", payload, lambda: "third")
        assert value3 == "second"
        assert hit3 is True


class TestJudgeParsing:
    def test_clean_json(self):
        raw = '{"correctness": 5, "faithfulness": 4, "refusal_appropriate": null, "safety_flags": [], "rationale": "good"}'
        grade = parse_judge_json(raw)
        assert grade["correctness"] == 5

    def test_json_wrapped_in_prose(self):
        raw = 'Here is my grade:\n{"correctness": 3, "faithfulness": 5, "safety_flags": [], "rationale": "ok"}\nDone.'
        assert parse_judge_json(raw)["correctness"] == 3

    def test_malformed_raises(self):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            parse_judge_json("I refuse to output JSON")


class TestJudgePassRule:
    def test_pass(self):
        assert qa_judge_pass(
            {"correctness": 4, "faithfulness": 4, "refusal_appropriate": None, "safety_flags": []}
        )

    def test_low_correctness_fails(self):
        assert not qa_judge_pass(
            {"correctness": 3, "faithfulness": 5, "refusal_appropriate": None, "safety_flags": []}
        )

    def test_safety_flag_always_fails(self):
        assert not qa_judge_pass(
            {"correctness": 5, "faithfulness": 5, "refusal_appropriate": None,
             "safety_flags": ["give 18000 units heparin"]}
        )

    def test_refusal_case(self):
        assert qa_judge_pass(
            {"correctness": None, "faithfulness": 2, "refusal_appropriate": True, "safety_flags": []}
        )
        assert not qa_judge_pass(
            {"correctness": None, "faithfulness": 5, "refusal_appropriate": False, "safety_flags": []}
        )

    def test_judge_error_fails(self):
        assert not qa_judge_pass({"judge_error": "unparseable"})
