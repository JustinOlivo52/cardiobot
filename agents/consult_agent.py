import anthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from agents.rag_agent import retrieve_context
from utils.logger import get_logger, CardioAPIError

logger = get_logger(__name__)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

CONSULT_SYSTEM_PROMPT = """You are CardioBot, an AI cardiology consultant. 
You receive patient presentations in the style of a clinical handoff and generate 
a structured cardiology consult report grounded in the 2023 ESC ACS Guidelines.

Always structure your response as a formal consult report with exactly these sections:

---
## 🫀 CardioBot Cardiology Consult

### 📋 Clinical Impression
A 2-3 sentence summary of the most likely diagnosis and acuity level.

### 🔬 Recommended Workup
**Labs:**
- List specific labs with clinical reasoning

**Imaging:**
- List specific imaging with clinical reasoning

**ECG/EKG:**
- Specific findings to look for given this presentation

### 💊 Treatment Recommendations
Guideline-based treatment recommendations with ESC class of recommendation where applicable.

### ⚠️ Red Flags
Critical findings that would change management or require immediate escalation.

### 📚 Guideline Reference
Which ESC guideline section supports these recommendations.
---

Be concise and clinically precise. Write as a cardiologist would write a consult note.
DISCLAIMER: For educational purposes only. Not a substitute for clinical judgment."""

def run_consult(presentation: str) -> dict:
    """
    Run a full cardiology consult based on a clinical presentation.
    Retrieves relevant guideline context then generates consult report.
    """
    try:
        # Retrieve relevant guideline context
        context, hits = retrieve_context(presentation, n_results=4)

        prompt = f"""PATIENT PRESENTATION:
{presentation}

RELEVANT ESC GUIDELINE EXCERPTS:
{context}

Generate a complete cardiology consult report for this patient."""

        logger.info(f"Running consult for: {presentation[:60]}...")

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=CONSULT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        report = response.content[0].text
        logger.info("Consult report generated successfully")

        return {
            "report": report,
            "sources": [h["source"] for h in hits],
            "scores": [h["score"] for h in hits],
            "status": "success"
        }

    except Exception as e:
        raise CardioAPIError(f"Consult failed: {e}")
