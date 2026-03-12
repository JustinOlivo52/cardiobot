from openai import OpenAI
from config import OPENAI_API_KEY, GPT4_MODEL
from tools.calculator import calculate_dose, get_available_drugs
from utils.logger import get_logger, CardioAPIError

logger = get_logger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)

DOSING_SYSTEM_PROMPT = """You are a clinical pharmacist specializing in cardiac medications.

Given a drug name, patient weight, and pre-calculated dose information, provide:
1. Confirmation of the calculated dose
2. How to prepare/administer the medication
3. Key monitoring parameters
4. Critical contraindications to check
5. Any weight-based adjustments for special populations (elderly, renal impairment)

Be precise with numbers. Always flag high-alert medications.
Format your response in clear clinical sections.

DISCLAIMER: Always verify doses with pharmacy and institutional protocols before administration."""

def get_dosing_guidance(drug: str, weight_kg: float, clinical_context: str = "") -> dict:
    """
    Calculate dose and get GPT-4 clinical guidance.
    """
    try:
        # Step 1: Calculate dose from our drug database
        dose_info = calculate_dose(drug, weight_kg)

        if "error" in dose_info:
            available = ", ".join(get_available_drugs())
            return {
                "error": dose_info["error"],
                "available_drugs": available
            }

        # Step 2: Build prompt for GPT-4
        prompt = f"""
Drug: {drug.upper()}
Patient Weight: {weight_kg} kg
Indication: {dose_info.get('indication', 'Not specified')}
Clinical Context: {clinical_context or 'Standard dosing request'}

Pre-calculated dose information:
- IV Bolus: {dose_info.get('iv_bolus', 'N/A')}
- Infusion: {dose_info.get('infusion', 'N/A')}
- Weight-based calculated dose: {dose_info.get('calculated_bolus', 'Fixed dose - not weight based')}
- Notes: {dose_info.get('notes', 'None')}

Please provide complete clinical dosing guidance for this patient.
"""

        # Step 3: Ask GPT-4
        logger.info(f"Requesting GPT-4 dosing guidance for {drug} ({weight_kg}kg)")
        response = client.chat.completions.create(
            model=GPT4_MODEL,
            max_tokens=800,
            messages=[
                {"role": "system", "content": DOSING_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )

        guidance = response.choices[0].message.content
        logger.info("GPT-4 dosing guidance received")

        return {
            "drug": drug.upper(),
            "weight_kg": weight_kg,
            "indication": dose_info.get("indication"),
            "calculated_bolus": dose_info.get("calculated_bolus", "Fixed dose"),
            "iv_bolus": dose_info.get("iv_bolus"),
            "infusion": dose_info.get("infusion"),
            "guidance": guidance,
            "status": "success"
        }

    except Exception as e:
        raise CardioAPIError(f"Dosing calculation failed: {e}")
