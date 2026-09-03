"""
Golden dataset validation. Runs free (no API keys) and in CI, so a broken
or mislabeled dataset fails before anyone spends money on an eval run.
"""
import json
from pathlib import Path

import pytest

from tools.calculator import calculate_dose

DATASETS = Path(__file__).resolve().parents[1] / "evaluation" / "datasets"

QA_REQUIRED = {"id", "case_type", "category", "difficulty", "question",
               "reference_answer", "expected_pages", "expected_keywords"}
PDF_PAGE_RANGE = range(1, 108)  # cardiac_treatment2.pdf spans pages 1-107


def load_jsonl(name):
    lines = (DATASETS / name).read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture(scope="module")
def qa_cases():
    return load_jsonl("qa_cases.jsonl")


@pytest.fixture(scope="module")
def consult_cases():
    return load_jsonl("consult_cases.jsonl")


@pytest.fixture(scope="module")
def dosing_cases():
    return json.loads((DATASETS / "dosing_cases.json").read_text())


class TestQACases:
    def test_minimum_size(self, qa_cases):
        standard = [c for c in qa_cases if c["case_type"] == "standard"]
        refusal = [c for c in qa_cases if c["case_type"] == "refusal"]
        assert len(standard) >= 20
        assert len(refusal) >= 4

    def test_required_fields(self, qa_cases):
        for case in qa_cases:
            missing = QA_REQUIRED - set(case)
            assert not missing, f"{case.get('id')} missing {missing}"

    def test_unique_ids(self, qa_cases):
        ids = [c["id"] for c in qa_cases]
        assert len(ids) == len(set(ids))

    def test_standard_cases_have_reference_and_pages(self, qa_cases):
        for case in qa_cases:
            if case["case_type"] != "standard":
                continue
            assert case["reference_answer"], case["id"]
            assert case["expected_pages"], f"{case['id']} has no retrieval ground truth"

    def test_refusal_cases_have_expected_behavior(self, qa_cases):
        for case in qa_cases:
            if case["case_type"] != "refusal":
                continue
            assert case.get("expected_behavior"), case["id"]
            assert case["expected_pages"] == [], case["id"]

    def test_pages_in_pdf_range(self, qa_cases):
        for case in qa_cases:
            for page in case["expected_pages"]:
                assert page in PDF_PAGE_RANGE, f"{case['id']} page {page}"

    def test_case_types_valid(self, qa_cases):
        for case in qa_cases:
            assert case["case_type"] in ("standard", "refusal"), case["id"]
            assert case["difficulty"] in ("easy", "medium", "hard"), case["id"]


class TestConsultCases:
    def test_required_fields(self, consult_cases):
        for case in consult_cases:
            for field in ("id", "category", "presentation", "reference_key_points", "expected_pages"):
                assert field in case, f"{case.get('id')} missing {field}"
            assert len(case["reference_key_points"]) >= 3, case["id"]

    def test_unique_ids(self, consult_cases):
        ids = [c["id"] for c in consult_cases]
        assert len(ids) == len(set(ids))

    def test_minimum_size(self, consult_cases):
        assert len(consult_cases) >= 5


class TestDosingCases:
    def test_unique_ids(self, dosing_cases):
        ids = [c["id"] for c in dosing_cases]
        assert len(ids) == len(set(ids))

    def test_ground_truth_agrees_with_calculator(self, dosing_cases):
        """The dataset and calculator must agree; disagreement means one
        of them changed without the other."""
        for case in dosing_cases:
            result = calculate_dose(case["drug"], case["weight_kg"])
            if case["expect_error"]:
                assert "error" in result, case["id"]
                continue
            assert "error" not in result, f"{case['id']}: {result.get('error')}"
            if case["expected_dose"] is None:
                assert "calculated_dose" not in result, case["id"]
            else:
                assert result["calculated_dose"] == case["expected_dose"], case["id"]
                assert result["dose_type"] == case["expected_dose_type"], case["id"]
                assert result["dose_capped"] == case["expect_capped"], case["id"]


class TestPageMap:
    def test_page_map_committed_and_complete(self):
        page_map = json.loads((DATASETS / "page_map.json").read_text())
        assert len(page_map) == 947
        all_pages = {p for pages in page_map.values() for p in pages}
        assert min(all_pages) >= 1
        assert max(all_pages) in PDF_PAGE_RANGE
