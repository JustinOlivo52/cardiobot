from evaluation.checks import (
    consult_sections_present,
    contains_dose_numbers,
    disclaimer_present,
    extract_pages,
    keyword_coverage,
    ungrounded_numbers,
)

WELL_FORMED_CONSULT = """
### 🔍 Clinical Impression
Likely inferior STEMI.

### 🧪 Recommended Workup
12-lead ECG with right-sided leads, high-sensitivity troponin.

### 💊 Treatment Recommendations
Aspirin loading dose, primary PCI activation.

### 🚩 Red Flags
Hypotension with clear lungs suggests RV involvement.

### 📖 Guideline Reference
2023 ESC ACS Guidelines.

DISCLAIMER: For educational purposes only.
"""


class TestExtractPages:
    def test_single_marker(self):
        assert extract_pages("[Page 42]\nsome guideline text") == {42}

    def test_multiple_markers(self):
        text = "end of one page [Page 12]\nnew page text [Page 13]\nmore"
        assert extract_pages(text) == {12, 13}

    def test_no_marker(self):
        assert extract_pages("chunk split mid-page with no marker") == set()

    def test_empty_and_none(self):
        assert extract_pages("") == set()
        assert extract_pages(None) == set()


class TestConsultSections:
    def test_all_sections_found(self):
        result = consult_sections_present(WELL_FORMED_CONSULT)
        assert all(result.values())
        assert len(result) == 5

    def test_missing_section_detected(self):
        report = WELL_FORMED_CONSULT.replace("Red Flags", "Warnings")
        result = consult_sections_present(report)
        assert result["Red Flags"] is False
        assert result["Clinical Impression"] is True

    def test_case_insensitive(self):
        assert consult_sections_present("clinical impression: ok")["Clinical Impression"]


class TestDisclaimer:
    def test_present(self):
        assert disclaimer_present("This is for educational purposes only.")
        assert disclaimer_present("Not a substitute for clinical judgment")

    def test_absent(self):
        assert not disclaimer_present("Give aspirin 300mg now.")


class TestKeywordCoverage:
    def test_full_coverage(self):
        assert keyword_coverage("Aspirin plus ticagrelor", ["aspirin", "ticagrelor"]) == 1.0

    def test_partial(self):
        assert keyword_coverage("Aspirin alone", ["aspirin", "ticagrelor"]) == 0.5

    def test_empty_keywords_is_full(self):
        assert keyword_coverage("anything", []) == 1.0


class TestUngroundedNumbers:
    def test_grounded_claims_not_flagged(self):
        context = "[Page 40] UFH 70-100 U/kg i.v. bolus. Aspirin 150-300 mg loading."
        answer = "Give aspirin 300 mg loading dose."
        assert ungrounded_numbers(answer, context) == []

    def test_fabricated_claim_flagged(self):
        context = "[Page 40] Aspirin loading dose is recommended."
        answer = "Give vancomycin 1250 mg immediately."
        flagged = ungrounded_numbers(answer, context)
        assert any("1250" in f for f in flagged)

    def test_no_numbers_no_flags(self):
        assert ungrounded_numbers("Refer to cardiology.", "any context") == []


class TestContainsDoseNumbers:
    def test_dose_detected(self):
        assert contains_dose_numbers("Amoxicillin 500 mg three times daily")
        assert contains_dose_numbers("give 5000 units of heparin")

    def test_clean_refusal(self):
        assert not contains_dose_numbers(
            "I can't advise on antibiotic dosing; my knowledge covers the 2023 ESC ACS guidelines."
        )

    def test_timings_are_not_doses(self):
        assert not contains_dose_numbers("Door-to-balloon time should be under 90 minutes.")
