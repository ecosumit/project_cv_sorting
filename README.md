# CV Shortlister (Local, Ollama, Streamlit)

End‑to‑end, local CV sorting & scoring system that parses resumes (PDF/DOCX/TXT), extracts signals, and ranks candidates against a Job Description (JD). It complies with the capstone spec “CV Sorting using LLMs”: uses **Ollama w/ llama3:8b**, **LangChain** orchestration, **pyresparser**/**open‑resume** style extraction (with resilient fallback), semantic embeddings via **Ollama**, a transparent scoring rubric, and a **Streamlit** UI.


## Features
- Upload multiple CVs + one JD; get an **explainable ranking** with per‑criterion reasons.
- Hybrid scoring:
  - **Must‑haves gate** (years, required skills)
  - **Keyword coverage** (critical & desired)
  - **Semantic similarity** (JD↔CV chunks via Ollama embeddings)
  - **LLM rubric** with structured JSON output (llama3:8b via Ollama)
  - **Bonus/Malus** (leadership, publications, job hops proxy)
- Uses **LangChain** to template prompts and parse JSON reliably.
- Extraction via **pyresparser** when available; otherwise, robust fallback (regex + patterns).
- **Local‑first**: models run via Ollama; no external API calls.
- Export CSV/JSON; inspect per‑candidate explanations in UI.

> This project follows the provided guideline document exactly (models, steps, evaluation, repo hygiene).

## Architecture
```
cv-shortlister-ollama/
├─ app.py                # Streamlit UI
├─ config.yaml           # Scoring config & rubric
├─ src/
│  ├─ parsing.py         # Resume & JD parsing (pyresparser + fallback; PDF/DOCX/TXT)
│  ├─ jd_parser.py       # JD requirement extraction with LLM (LangChain + Ollama)
│  ├─ scoring.py         # Must-haves, keywords, semantic, rubric, bonus/malus
│  ├─ ranker.py          # End-to-end ranking pipeline
│  ├─ prompts.py         # Prompt templates
│  └─ ui_components.py   # Streamlit helpers
├─ data/
│  ├─ sample_jd.txt
│  └─ sample_cvs/        # Put sample PDFs/DOCX/TXT here
├─ evaluation/
│  └─ evaluate.py        # Offline eval: precision/recall/MAP given labels
├─ scripts/
│  ├─ setup_ollama.sh    # Pull models
│  └─ docker/Dockerfile  # Optional container
├─ requirements.txt
└─ README.md
```

## Prereqs
- Python 3.10+
- [Ollama](https://ollama.com/) installed & running locally
- (Windows) Visual C++ Build Tools may be required for some deps
- (Linux/Mac) standard build tools

## Install
```bash
git clone <your-repo-url>.git cv-shortlister-ollama
cd cv-shortlister-ollama

# Python deps
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# (Optional) spaCy model for pyresparser accuracy
python -m spacy download en_core_web_sm || true

# Ollama models
./scripts/setup_ollama.sh
```

## Run
```bash
streamlit run app.py
```
Open the local URL printed by Streamlit. Upload a JD (.txt) and a set of CVs (.pdf/.docx/.txt), then click **Score**.

## Config
See `config.yaml` to tune weights, must‑haves, and rubric.

## Evaluation (optional)
Put your ground‑truth labels in a CSV and run `evaluation/evaluate.py` to compute accuracy/precision/recall/MAP.

![alt text](image.png)
