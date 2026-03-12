from google import genai
from google.genai import types
from config import GOOGLE_API_KEY
from tools.image_tool import load_and_encode_image
from utils.logger import get_logger, CardioImageError

logger = get_logger(__name__)

VISION_MODEL = "gemini-2.5-flash"

client = genai.Client(api_key=GOOGLE_API_KEY)

EKG_SYSTEM_PROMPT = """You are an expert cardiologist analyzing an ECG/EKG strip.

Analyze the ECG systematically using this structure:
1. **Rate** — Calculate heart rate (bpm)
2. **Rhythm** — Regular or irregular? Identify the rhythm
3. **P Waves** — Present? Normal morphology? PR interval?
4. **QRS Complex** — Width? Morphology? Any bundle branch block?
5. **ST Segment** — Elevation or depression? Which leads?
6. **T Waves** — Normal, inverted, peaked, or biphasic?
7. **Impression** — Your overall clinical interpretation
8. **Recommended Action** — What should be done clinically?

Be specific about lead findings. Flag any life-threatening findings immediately.

DISCLAIMER: This is an AI interpretation for educational purposes only. 
Always have ECGs reviewed by a qualified physician before clinical decisions."""

def interpret_ekg(image_source) -> dict:
    """
    Send an EKG image to Gemini Vision for interpretation.
    Returns a dict with interpretation and metadata.
    """
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
