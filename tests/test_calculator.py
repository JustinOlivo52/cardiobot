import pytest
from tools.calculator import calculate_dose, get_available_drugs

def test_heparin_bolus_is_weight_based():
    assert calculate_dose("heparin", 70)["calculated_dose"] == "4200.0 units"

def test_heparin_bolus_is_capped_at_max():
    result = calculate_dose("heparin", 100)
    assert result["calculated_dose"] == "5000.0 units (max dose)"

def test_epinephrine_is_labeled_as_infusion():
    result = calculate_dose("epinephrine", 70)
    assert result["calc_label"] == "Calculated Infusion"
    assert result["calculated_dose"] == "7.0 mcg/min"

def test_lidocaine_bolus():
    result = calculate_dose("lidocaine", 80)
    assert result["calc_label"] == "Calculated Bolus"
    assert result["calculated_dose"] == "120.0 mg"

def test_fixed_dose_drug_has_no_calculated_dose():
    assert "calculated_dose" not in calculate_dose("amiodarone", 70)

def test_drug_lookup_ignores_case_and_spaces():
    assert "error" not in calculate_dose("  Heparin ", 70)

def test_unknown_drug_returns_error():
    assert "error" in calculate_dose("tylenol", 70)

@pytest.mark.parametrize("drug", get_available_drugs())
def test_every_drug_has_core_fields(drug):
    result = calculate_dose(drug, 70)
    for field in ("indication", "iv_bolus", "infusion", "notes"):
        assert result.get(field), f"{drug} missing {field}"
