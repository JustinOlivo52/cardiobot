from utils.logger import get_logger

logger = get_logger(__name__)

CARDIAC_DRUGS = {
    "amiodarone": {
        "indication": "VF/pulseless VT (cardiac arrest)",
        "iv_bolus": "300mg IV push, may repeat 150mg",
        "infusion": "1mg/min x 6hrs, then 0.5mg/min x 18hrs",
        "weight_based": False,
        "notes": "Dilute in D5W. Monitor for hypotension."
    },
    "lidocaine": {
        "indication": "VF/pulseless VT (alternative to amiodarone)",
        "iv_bolus": "1-1.5 mg/kg IV push",
        "infusion": "1-4 mg/min",
        "weight_based": True,
        "dose_per_kg": 1.5,
        "notes": "Max 3mg/kg total. Monitor for CNS toxicity."
    },
    "epinephrine": {
        "indication": "Cardiac arrest / anaphylaxis",
        "iv_bolus": "1mg IV every 3-5 min (cardiac arrest)",
        "infusion": "0.1-0.5 mcg/kg/min (cardiogenic shock)",
        "weight_based": True,
        "dose_per_kg": 0.1,
        "notes": "High alert medication. Verify concentration before administration."
    },
    "heparin": {
        "indication": "STEMI / NSTEMI anticoagulation",
        "iv_bolus": "60-70 units/kg IV (max 5000 units)",
        "infusion": "12-15 units/kg/hr (max 1000 units/hr)",
        "weight_based": True,
        "dose_per_kg": 60,
        "notes": "Follow institutional heparin protocol. Monitor aPTT."
    },
    "metoprolol": {
        "indication": "Rate control / ACS",
        "iv_bolus": "5mg IV every 5 min x 3 doses",
        "infusion": "N/A",
        "weight_based": False,
        "notes": "Hold for HR<60, SBP<100, or signs of heart failure."
    },
    "nitroglycerin": {
        "indication": "ACS / acute pulmonary edema",
        "iv_bolus": "400mcg SL every 5 min x 3",
        "infusion": "5-200 mcg/min IV",
        "weight_based": False,
        "notes": "Hold for SBP<90. Contraindicated with PDE5 inhibitors."
    },
    "atropine": {
        "indication": "Symptomatic bradycardia",
        "iv_bolus": "0.5mg IV every 3-5 min (max 3mg)",
        "infusion": "N/A",
        "weight_based": False,
        "notes": "Not effective for high-degree AV block."
    },
    "adenosine": {
        "indication": "SVT conversion",
        "iv_bolus": "6mg rapid IV push, then 12mg x 2 if needed",
        "infusion": "N/A",
        "weight_based": False,
        "notes": "Give as fast IV push followed by rapid saline flush. Warn patient of brief chest discomfort."
    }
}

def get_available_drugs() -> list[str]:
    return sorted(CARDIAC_DRUGS.keys())

def calculate_dose(drug: str, weight_kg: float) -> dict:
    """Calculate weight-based dose for a cardiac drug."""
    drug = drug.lower().strip()
    if drug not in CARDIAC_DRUGS:
        return {"error": f"Drug '{drug}' not in database"}

    info = CARDIAC_DRUGS[drug].copy()

    if info.get("weight_based") and weight_kg > 0:
        dose_per_kg = info.get("dose_per_kg", 0)
        calculated_dose = dose_per_kg * weight_kg
        info["calculated_bolus"] = f"{calculated_dose:.1f} {get_unit(drug)}"
        info["patient_weight"] = f"{weight_kg} kg"

    return info

def get_unit(drug: str) -> str:
    units = {
        "lidocaine": "mg",
        "epinephrine": "mcg/min (infusion)",
        "heparin": "units",
    }
    return units.get(drug, "mg")
