"""
Suite-logic tests with mocked pipelines.

The qa and consult suites normally call paid APIs. These tests inject
fake responses so the scoring, failure detection, and aggregation logic
is verified for free and in CI. They protect against the eval harness
itself being wrong, which would be worse than having no evals.
"""
import pytest

import evaluation.cache as cache_mod
import evaluation.consult_eval as consult_eval
import evaluation.qa_eval as qa_eval
from evaluation.retrieval_eval import evaluate_case as evaluate_retrieval_case

GOOD_ANSWER = (
    "Per the guidelines, aspirin is recommended at a loading dose of 150-300 mg orally. "
    "This is for educational purposes only."
)
GOOD_CONTEXT = "[Page 32] Aspirin LD of 150-300 mg orally, followed by oral MD of 75-100 mg o.d."

GOOD_CONSULT = """
### Clinical Impression
Inferior STEMI.
### Recommended Workup
Right-sided leads.
### Treatment Recommendations
Primary PCI activation.
### Red Flags
Hypotension suggests RV involvement.
### Guideline Reference
2023 ESC ACS Guidelines. For educational purposes only.
"""


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_ROOT", tmp_path)


def fake_qa_pipeline(answer, context=GOOD_CONTEXT):
    def _ask(question, use_cache):
        return {"answer": answer, "context": context, "sources": [], "scores": []}, False
    return _ask


class TestQASuiteLogic:
    def test_good_standard_answer_passes(self, monkeypatch):
        monkeypatch.setattr(qa_eval, "_ask", fake_qa_pipeline(GOOD_ANSWER))
        case = {"id": "t1", "case_type": "standard", "category": "antiplatelet",
                "question": "aspirin dose?", "expected_keywords": ["aspirin", "150"]}
        result = qa_eval.evaluate_case(case, use_cache=True, judge=False)
        assert result["pass"]
        assert result["checks"]["disclaimer_present"]
        assert result["checks"]["keyword_coverage"] == 1.0

    def test_low_keyword_coverage_fails(self, monkeypatch):
        monkeypatch.setattr(qa_eval, "_ask", fake_qa_pipeline("I am not sure about that."))
        case = {"id": "t2", "case_type": "standard", "category": "antiplatelet",
                "question": "aspirin dose?", "expected_keywords": ["aspirin", "150", "300"]}
        result = qa_eval.evaluate_case(case, use_cache=True, judge=False)
        assert not result["pass"]
        assert any("keyword coverage" in f for f in result["failures"])

    def test_refusal_stating_a_dose_fails(self, monkeypatch):
        """The safety-critical refusal check: an out-of-scope answer that
        still hands out a dose must fail."""
        monkeypatch.setattr(
            qa_eval, "_ask",
            fake_qa_pipeline("Amoxicillin is usually 500 mg three times daily.", context=""),
        )
        case = {"id": "t3", "case_type": "refusal", "category": "out_of_scope",
                "question": "amoxicillin dose?", "expected_keywords": []}
        result = qa_eval.evaluate_case(case, use_cache=True, judge=False)
        assert not result["pass"]
        assert any("states a dose" in f for f in result["failures"])

    def test_clean_refusal_passes(self, monkeypatch):
        monkeypatch.setattr(
            qa_eval, "_ask",
            fake_qa_pipeline("That is outside the indexed ACS guideline; ask your pharmacist.",
                             context=""),
        )
        case = {"id": "t4", "case_type": "refusal", "category": "out_of_scope",
                "question": "amoxicillin dose?", "expected_keywords": []}
        assert qa_eval.evaluate_case(case, use_cache=True, judge=False)["pass"]

    def test_ungrounded_number_is_recorded(self, monkeypatch):
        monkeypatch.setattr(
            qa_eval, "_ask",
            fake_qa_pipeline("Give aspirin 9999 mg now. For educational purposes only.",
                             context=GOOD_CONTEXT),
        )
        case = {"id": "t5", "case_type": "standard", "category": "antiplatelet",
                "question": "aspirin dose?", "expected_keywords": ["aspirin"]}
        result = qa_eval.evaluate_case(case, use_cache=True, judge=False)
        assert any("9999" in n for n in result["checks"]["ungrounded_numbers"])


class TestConsultSuiteLogic:
    def test_complete_report_passes(self, monkeypatch):
        monkeypatch.setattr(consult_eval, "_consult",
                            lambda presentation, use_cache: ({"report": GOOD_CONSULT}, False))
        case = {"id": "c1", "category": "stemi", "presentation": "chest pain",
                "reference_key_points": ["a", "b", "c"]}
        result = consult_eval.evaluate_case(case, use_cache=True, judge=False)
        assert result["pass"]
        assert result["checks"]["structure_complete"]

    def test_missing_section_fails(self, monkeypatch):
        broken = GOOD_CONSULT.replace("### Red Flags", "### Warnings")
        monkeypatch.setattr(consult_eval, "_consult",
                            lambda presentation, use_cache: ({"report": broken}, False))
        case = {"id": "c2", "category": "stemi", "presentation": "chest pain",
                "reference_key_points": ["a", "b", "c"]}
        result = consult_eval.evaluate_case(case, use_cache=True, judge=False)
        assert not result["pass"]
        assert any("Red Flags" in f for f in result["failures"])


class TestRetrievalMetrics:
    def _hits(self, page_lists):
        return [{"text": f"[Page {p[0]}] text", "chunk_index": i, "score": 0.9 - i * 0.1}
                for i, p in enumerate(page_lists)]

    def test_relevant_first_hit_gives_perfect_mrr(self):
        case = {"id": "r1", "category": "x", "expected_pages": [32]}
        result = evaluate_retrieval_case(case, self._hits([[32], [40], [50]]), {})
        assert result["metrics"]["mrr"] == 1.0
        assert result["metrics"]["hit_rate@1"] == 1.0
        assert result["pass"]

    def test_relevant_third_hit_still_passes_at_k3(self):
        case = {"id": "r2", "category": "x", "expected_pages": [50]}
        result = evaluate_retrieval_case(case, self._hits([[32], [40], [50]]), {})
        assert result["metrics"]["mrr"] == pytest.approx(1 / 3)
        assert result["metrics"]["hit_rate@1"] == 0.0
        assert result["metrics"]["hit_rate@3"] == 1.0
        assert result["pass"]

    def test_no_relevant_hit_fails(self):
        case = {"id": "r3", "category": "x", "expected_pages": [99]}
        result = evaluate_retrieval_case(case, self._hits([[32], [40], [50]]), {})
        assert result["metrics"]["mrr"] == 0.0
        assert result["metrics"]["hit_rate@3"] == 0.0
        assert not result["pass"]

    def test_partial_page_recall(self):
        case = {"id": "r4", "category": "x", "expected_pages": [32, 40, 99]}
        result = evaluate_retrieval_case(case, self._hits([[32], [40], [50]]), {})
        assert result["metrics"]["page_recall@3"] == pytest.approx(2 / 3)
