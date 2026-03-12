from google import genai
from google.genai import types
from config import GOOGLE_API_KEY
from tools.image_tool import load_and_encode_image
from utils.logger import get_logger, CardioImageError

logger = get_logger(__name__)

VISION_MODEL = "gemini-2.5-flash"

client = genai.Client(api_key=GOOGLE_API_KEY)

EKG_SYSTEM_PROMPT = """You are an expert cardiologist analyzing an ECG/EKG strip.

Structure your response in exactly two sections in this order:

## 🫀 Clinical Interpretation
State your overall clinical impression in 1-2 sentences maximum.
Example: "Clinical Interpretation: STEMI — Inferior MI pattern with ST elevation in leads II, III, and aVF."
Be direct and concise. This is what the ER physician reads first.

## 📋 Clinical Breakdown
Then provide the systematic analysis:
1. **Rate** — Heart rate in bpm
2. **Rhythm** — Regular or irregular, rhythm name
3. **P Waves** — Present? Normal? PR interval?
4. **QRS Complex** — Width, morphology, bundle branch block?
5. **ST Segment** — Elevation or depression, which leads?
6. **T Waves** — Normal, inverted, peaked, or biphasic?
7. **Recommended Action** — Immediate next clinical step

Flag any life-threatening findings in the breakdown with ⚠️

DISCLAIMER: This is an AI interpretation for educational purposes only. Always have ECGs reviewed by a qualified physician before clinical decisions."""

def interpret_ekg(image_source) -> dict:
    try:
        logger.info("Loading and encoding EKG image...")
        encoded_image, media_type = load_and_encode_image(image_source)

        logger.info(f"Sending EKG to Gemini Vision ({VISION_MODEL})...")
        response = client.models.generate_content(
            model=VISION_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type=media_type,
                                data=encoded_image
                            )
                        ),
                        types.Part(text=EKG_SYSTEM_PROMPT)
                    ]
                )
            ]
        )

        interpretation = response.text
        logger.info("Gemini Vision EKG interpretation complete")

        return {
            "interpretation": interpretation,
            "model": VISION_MODEL,
            "status": "success"
        }

    except CardioImageError:
        raise
    except Exception as e:
        raise CardioImageError(f"EKG interpretation failed: {e}")
