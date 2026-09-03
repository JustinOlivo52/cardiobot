# 🫀 CardioBot — AI-Powered Clinical Cardiology Assistant

**🔴 Live Demo: [cardiodoc.streamlit.app](https://cardiodoc.streamlit.app/)**

> A multi-LLM agentic RAG system built for clinical cardiology support, powered by Claude, Gemini, and GPT-4.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)
![Claude](https://img.shields.io/badge/Claude-Sonnet-purple)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏥 What is CardioBot?

CardioBot is a clinical AI assistant that helps healthcare professionals and students navigate cardiology guidelines, interpret EKGs, and calculate cardiac drug doses — all powered by a multi-LLM architecture and grounded in the 2023 ESC ACS Guidelines.

Built by an Emergency Room Technician transitioning into health tech AI, CardioBot combines real clinical knowledge with modern AI engineering.

---

## 🎯 The Problem It Solves

In a busy ER, clinicians need fast, reliable answers to questions like:
- *"What's the ESC recommendation for NSTEMI antiplatelet therapy?"*
- *"What's the heparin dose for this 95kg patient?"*
- *"Is this EKG showing a STEMI?"*

CardioBot answers all three — with citations, weight-based calculations, and AI-powered EKG interpretation.

---

## 🛠️ Architecture
```
User Input
    │
    ▼
LangGraph Router ──────────────────────────────────────────┐
    │                                                       │
    ├── Clinical Q&A ──► ChromaDB (OpenAI Embeddings)      │
    │                        │                             │
    │                        ▼                             │
    │                   Claude Sonnet ◄── ESC Guidelines   │
    │                                                       │
    ├── EKG Interpreter ──► Gemini Vision                  │
    │                                                       │
    ├── Drug Calculator ──► GPT-4 + Weight-Based Logic     │
    │                                                       │
    └── Citation Checker ──► ChromaDB Direct Search        │
                                                           │
    ◄──────────────────────────────────────────────────────┘
Streamlit UI (Dark Theme, Multi-Tab)
```

### Models & Their Roles
| Model | Provider | Role |
|---|---|---|
| Claude Sonnet | Anthropic | Clinical reasoning + answer generation |
| Gemini 2.5 Flash | Google | EKG/ECG image interpretation |
| GPT-4 | OpenAI | Drug dosing guidance |
| text-embedding-3-small | OpenAI | Document embeddings |

### Tech Stack
| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| Vector Database | ChromaDB |
| UI | Streamlit |
| PDF Processing | pypdf + LangChain |
| Memory | Conversation Buffer |

---

## ✨ Features

### 💬 Clinical Q&A
- RAG pipeline grounded in 2023 ESC ACS Guidelines
- 947 indexed passages with semantic search
- Source citations with relevance scores
- Multi-turn conversation memory

### 📊 EKG Interpreter
- Upload any EKG image (JPG, PNG, WebP)
- Gemini Vision analyzes waveform patterns
- Clinical format: **Impression first, breakdown second**
- Systematic read: Rate → Rhythm → P Waves → QRS → ST → T Waves

### 💊 Drug Dosing Calculator
- 8 core cardiac medications
- Weight-based dose calculation
- GPT-4 clinical guidance with monitoring parameters
- Contraindication flagging

### 📚 Citation Checker
- Direct semantic search of ESC guidelines
- Confidence scoring (High/Medium/Low)
- Passage excerpts with source tracking

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12+
- API keys for Anthropic, Google, and OpenAI

### Installation
```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/cardiobot.git
cd cardiobot

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your API keys to .env
```

### Ingest the Guidelines
```bash
python ingest.py
```

### Run CardioBot
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
cardiobot/
├── app.py                  # Streamlit UI
├── config.py               # Environment + model config
├── ingest.py               # PDF ingestion pipeline
├── agents/
│   ├── claude_agent.py     # RAG + Claude answer generation
│   ├── ekg_agent.py        # Gemini Vision EKG interpreter
│   ├── dosing_agent.py     # GPT-4 drug dosing
│   ├── doc_checker.py      # Citation search
│   ├── rag_agent.py        # ChromaDB retrieval
│   ├── router.py           # LangGraph query router
│   └── graph.py            # LangGraph workflow
├── tools/
│   ├── retriever.py        # PDF loader + chunker
│   ├── embedder.py         # OpenAI embeddings + ChromaDB
│   ├── calculator.py       # Drug dose calculations
│   └── image_tool.py       # Image encoding utility
├── memory/
│   └── conversation.py     # Conversation buffer
├── prompts/
│   └── clinical_prompts.py # System prompts
├── utils/
│   └── logger.py           # Structured logging + exceptions
├── evaluation/
│   ├── run.py              # Eval CLI runner (gates + exit codes)
│   ├── checks.py           # Deterministic checks (pure functions)
│   ├── judge.py            # Claude LLM-as-judge grading
│   ├── cache.py            # Content-hash response cache
│   ├── report.py           # JSON/markdown reports + regression diff
│   ├── page_map.py         # Chunk → guideline page mapping
│   ├── label_helper.py     # Ground-truth labeling CLI
│   ├── *_eval.py           # retrieval / qa / consult / dosing suites
│   └── datasets/           # Golden datasets + page map
├── tests/                  # pytest suite (no API keys required)
└── data/
    └── sample_docs/        # PDF guidelines
```

---

## ⚕️ Clinical Data Sources

- **2023 ESC Guidelines for the Management of Acute Coronary Syndromes**
  - 107 pages, 947 indexed passages
  - Covers: ACS, STEMI, NSTEMI, antiplatelet therapy, anticoagulation, reperfusion

---

## 🔑 Environment Variables
```bash
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

---

## 🧪 Evaluation

CardioBot is evaluated in three layers, cheapest first, so most of the suite runs free and in CI.

### What is measured

| Layer | Metrics | Cost |
|---|---|---|
| **Deterministic** | Dose math + max-dose caps, consult report structure, refusal behavior, disclaimer rate, ungrounded-number tripwire | Free, no API keys |
| **Retrieval** | `hit_rate@k`, `page_recall@k`, MRR against page-level ground truth (k = 1, 3, 5, 10) | Query embeddings only |
| **LLM-as-judge** | Correctness vs. reference answer, faithfulness to retrieved context, refusal appropriateness, safety flags | Opt-in `--judge`, cached |

**`hit_rate@3` is the headline retrieval metric** — the Q&A pipeline feeds exactly the top 3 chunks to Claude, so k=3 is what actually reaches the model. The curve out to k=10 separates ranking problems (found at 10, missed at 3) from corpus problems (never found).

The judge runs Claude at `temperature=0` and returns structured JSON (two 1-5 scores, safety flags, and a rationale). A case passes only with correctness ≥ 4, faithfulness ≥ 4, and zero safety flags.

### Datasets

| File | Size | Contents |
|---|---|---|
| `evaluation/datasets/qa_cases.jsonl` | 32 | 26 standard ACS questions across 7 categories + 6 refusal/out-of-scope cases |
| `evaluation/datasets/consult_cases.jsonl` | 7 | Patient presentations with reference key points |
| `evaluation/datasets/dosing_cases.json` | 18 | All 8 drugs, edge weights, cap boundaries, invalid inputs |

Reference answers were written against the 2023 ESC ACS guidelines, and `expected_pages` labels were verified against the source PDF using `evaluation/label_helper.py`.

### Running it

```bash
pytest -q                                    # 84 tests, no API keys needed
python -m evaluation.run --suite dosing      # free safety suite
python -m evaluation.run --suite retrieval   # embeddings only
python -m evaluation.run --judge             # full run, all suites
```

Every run writes a timestamped JSON and markdown report to `evaluation/results/`, diffs against the previous run, and **exits nonzero** on any gate violation or regression. Gates live in a visible `GATES` dict in `evaluation/report.py`: dosing 100%, consult structure 100%, `hit_rate@3` ≥ 0.80, MRR ≥ 0.60, refusals 100%, judge correctness ≥ 4.0, zero safety flags. Any metric dropping more than 5 points from the previous run fails the run even if it clears its floor.

LLM responses are cached by content hash (`evaluation/.cache/`), so a first full run costs roughly $1.50-3.00 and reruns with nothing changed cost close to nothing.

### Ground truth honesty

Retrieval labels are page-level, not chunk-level, because chunks were split across page boundaries at ingest. `label_helper.py --grep` does a free keyword scan of all 947 chunks specifically so relevant pages the retriever *misses* can be labeled too — otherwise the ground truth would only ever contain what the retriever already finds, and the scores would flatter themselves.

### Known limitations

- Single-turn only. Conversation history is stored raw while the API receives the RAG-augmented prompt, so multi-turn evals would need to replicate that asymmetry.
- The EKG vision path has no evals yet; that needs a labeled image set.
- The `ungrounded_numbers` check matches on normalized number-and-unit strings, so rephrasings can produce false positives. It is a signal to inspect, not an automatic failure.

---

## ⚠️ Disclaimer

CardioBot is designed for **educational purposes only**. It is not a substitute for clinical judgment, and should never be used as the sole basis for medical decisions. Always consult current guidelines and qualified healthcare professionals.

---

## 👨‍💻 About the Developer

Built by **Justin Olivo** — Emergency Room Technician transitioning into Health Tech AI.

This project combines 10+ years of clinical ER experience with modern AI engineering to build tools that solve real problems in clinical environments.

- 🔗 [LinkedIn](https://linkedin.com/in/justin-olivo52)
- 🐙 [GitHub](https://github.com/JustinOlivo52)

---

## 🗺️ Roadmap

### ✅ Completed
- [x] Multi-agent LangGraph architecture with semantic router
- [x] Hybrid RAG pipeline with ChromaDB + OpenAI embeddings
- [x] EKG interpretation via Gemini Vision
- [x] Weight-based drug dosing calculator
- [x] Citation checker with confidence scoring
- [x] Evaluation suite: retrieval metrics, deterministic safety checks, LLM-as-judge grading
- [x] Drug dose safety limiters (max-dose caps)
- [x] Automated test suite (84 tests) + GitHub Actions CI
- [x] Deployed to Streamlit Cloud

### 🔜 In Progress
- [ ] Async streaming responses for real-time agent output

### 📋 Planned
- [ ] Add AHA and ACC guideline PDFs
- [ ] Re-ingest with page number metadata for precise citations
- [ ] Contraindication flagging beyond free-text notes
- [ ] EKG interpretation evals (needs a labeled EKG image set)
- [ ] Multi-turn conversation evals
- [ ] Pediatric weight-based dosing adjustments
- [ ] Migrate to FastAPI backend for production scaling


