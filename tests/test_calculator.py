"""
Safety lock-in tests for the dosing calculator.

These run with no API keys and no network. Two of these cases document
real bugs that were fixed:
- heparin had no max-dose cap (300kg input used to yield 18,000 units)
- epinephrine's weight-based value is an infusion rate (mcg/min), but it
  was labeled and unit-tagged as a bolus
"""
import pytest

from tools.calculator import CARDIAC_DRUGS, calculate_dose, get_available_drugs, get_unit

NON_WEIGHT_BASED = ["amiodarone", "metoprolol", "nitroglycerin", "atropine", "adenosine"]


class TestWeightBasedDosing:
    def test_heparin_standard_weight_uncapped(self):
        result = calculate_dose("heparin", 70)
        assert result["calculated_dose"] == "4200.0 units"
        assert result["dose_capped"] is False
        assert "cap_note" not in result

    def test_heparin_extreme_weight_is_capped(self):
        """The original bug: 60 u/kg x 300 kg returned 18,000 units."""
        result = calculate_dose("heparin", 300)
        assert result["calculated_dose"] == "5000.0 units"
        assert result["dose_capped"] is True
        assert "exceeds" in result["cap_note"]

    def test_heparin_cap_boundary(self):
        # 60 * 83.33 = 4999.8 -> just under the cap
        under = calculate_dose("heparin", 83.33)
        assert under["calculated_dose"] == "4999.8 units"
        assert under["dose_capped"] is False
        # 60 * 83.34 = 5000.4 -> capped to exactly 5000
        over = calculate_dose("heparin", 83.34)
        assert over["calculated_dose"] == "5000.0 units"
        assert over["dose_capped"] is True

    def test_epinephrine_is_infusion_not_bolus(self):
        """The original bug: an infusion rate was surfaced under a 'Bolus' label."""
        result = calculate_dose("epinephrine", 70)
        assert result["calculated_dose"] == "7.0 mcg/min"
        assert result["dose_type"] == "infusion"

    def test_lidocaine_bolus(self):
        result = calculate_dose("lidocaine", 80)
        assert result["calculated_dose"] == "120.0 mg"
        assert result["dose_type"] == "bolus"

    def test_patient_weight_echoed(self):
        result = calculate_dose("heparin", 72.5)
        assert result["patient_weight"] == "72.5 kg"


class TestNonWeightBasedDrugs:
    @pytest.mark.parametrize("drug", NON_WEIGHT_BASED)
    def test_no_calculated_dose(self, drug):
        result = calculate_dose(drug, 70)
        assert "error" not in result
        assert "calculated_dose" not in result
        assert result["weight_based"] is False


class TestInputValidation:
    def test_unknown_drug_returns_error(self):
        result = calculate_dose("amoxicillin", 70)
        assert "error" in result
        assert "amoxicillin" in result["error"]

    def test_name_normalization(self):
        result = calculate_dose("  HEPARIN ", 70)
        assert result["calculated_dose"] == "4200.0 units"

    @pytest.mark.parametrize("weight", [0, -5, -0.1])
    def test_non_positive_weight_returns_error(self, weight):
        result = calculate_dose("heparin", weight)
        assert "error" in result

    def test_available_drugs_sorted_complete(self):
        drugs = get_available_drugs()
        assert drugs == sorted(CARDIAC_DRUGS.keys())
        assert len(drugs) == 8


class TestDataIntegrity:
    def test_every_weight_based_drug_declares_dose_type(self):
        for name, info in CARDIAC_DRUGS.items():
            if info.get("weight_based"):
                assert info.get("dose_type") in ("bolus", "infusion"), name
                assert info.get("dose_per_kg", 0) > 0, name

    def test_units_defined_for_weight_based_drugs(self):
        assert get_unit("heparin") == "units"
        assert get_unit("epinephrine") == "mcg/min"
        assert get_unit("lidocaine") == "mg"
