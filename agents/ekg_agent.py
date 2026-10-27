from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, VISION_MODEL, CLAUDE_MODEL, EKG_PROVIDERS
from agents.claude_client import ask_claude
from tools.image_tool import prepare_for_vision
from utils.logger import get_logger, CardioImageError

logger = get_logger(__name__)

# No retries: Gemini's usual failure is a free-tier quota error, which a retry
# can't fix. Failing fast hands the image to the next provider sooner.
client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options=types.HttpOptions(
        timeout=45 * 1000,  # milliseconds
        retry_options=types.HttpRetryOptions(attempts=1)
    )
)

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

def _read_with_claude(encoded_image: str, media_type: str) -> str:
    return ask_claude(
        EKG_SYSTEM_PROMPT,
        [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": encoded_image}
                },
                {"type": "text", "text": "Interpret this EKG."}
            ]
        }],
        max_tokens=4096,
        effort="medium"
    )

def _read_with_gemini(encoded_image: str, media_type: str) -> str:
    response = client.models.generate_content(
        model=VISION_MODEL,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(inline_data=types.Blob(mime_type=media_type, data=encoded_image)),
                    types.Part(text=EKG_SYSTEM_PROMPT)
                ]
            )
        ]
    )
    return response.text

READERS = {
    "claude": (CLAUDE_MODEL, _read_with_claude),
    "gemini": (VISION_MODEL, _read_with_gemini),
}

def interpret_ekg(image_source) -> dict:
    """Try each vision provider in EKG_PROVIDERS order until one succeeds."""
    logger.info("Preparing EKG image...")
    encoded_image, media_type = prepare_for_vision(image_source)

    errors = []
    for provider in EKG_PROVIDERS:
        model, read = READERS[provider]
        try:
            logger.info(f"Sending EKG to {model}...")
            interpretation = read(encoded_image, media_type)
            if not interpretation or not interpretation.strip():
                raise CardioImageError(f"{model} returned an empty interpretation")
            logger.info(f"EKG interpretation complete ({model})")
            return {
                "interpretation": interpretation,
                "model": model,
                "status": "success"
            }
        except Exception as e:
            logger.warning(f"{model} EKG read failed, trying next provider: {e}")
            errors.append(f"{model}: {e}")

    raise CardioImageError("EKG interpretation failed. " + " | ".join(errors))
