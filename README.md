# 🫀 CardioBot — AI-Powered Clinical Cardiology Assistant

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
- 895 indexed passages with semantic search
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
└── data/
    └── sample_docs/        # PDF guidelines
```

---

## ⚕️ Clinical Data Sources

- **2023 ESC Guidelines for the Management of Acute Coronary Syndromes**
  - 107 pages, 895 indexed passages
  - Covers: ACS, STEMI, NSTEMI, antiplatelet therapy, anticoagulation, reperfusion

---

## 🔑 Environment Variables
```bash
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

---

## ⚠️ Disclaimer

CardioBot is designed for **educational purposes only**. It is not a substitute for clinical judgment, and should never be used as the sole basis for medical decisions. Always consult current guidelines and qualified healthcare professionals.

---

## 👨‍💻 About the Developer

Built by **Justin Olivo** — Emergency Room Technician transitioning into Health Tech AI.

This project combines 5+ years of clinical ER experience with modern AI engineering to build tools that solve real problems in clinical environments.

- 🔗 [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
- 🐙 [GitHub](https://github.com/YOUR_USERNAME)

---

## 🗺️ Roadmap

- [ ] Add AHA and ACC guideline PDFs
- [ ] Re-ingest with page number metadata
- [ ] Drug dose safety limiters
- [ ] Pediatric weight-based adjustments
- [ ] Deploy to Streamlit Cloud
- [ ] Add evaluation benchmarks

