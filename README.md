# PROJECT SAMARTHAN — 60% Functional Engineering POC

A judge-facing functional proof-of-concept for SIH26094: AI-assisted, continuous distress monitoring and human-led safety support for victims/complainants.

## What is actually functional

- FastAPI web application + SQLite persistence
- Real scikit-learn text classification pipeline (TF-IDF + Logistic Regression)
- Four engineering risk bands: Stable / Elevated / High / Critical
- Dynamic score combining model probabilities, personal-baseline deviation, engagement and response latency
- Optional uploaded WAV analysis using standard-library audio statistics (RMS, zero-crossing rate, duration)
- Longitudinal check-in history and trend chart
- Explainability: top model terms and contributing signal cards
- Human review queue with prioritisation
- Case view with proposed interventions and follow-up state
- Mock NHAA/e-Courts/PFMS adapters showing how authorised integrations would plug in
- DTMF-style silent duress simulation (`#`)
- Multilingual UI labels for English/Hindi/Marathi/Bengali/Tamil/Telugu/Kannada/Malayalam + additional Bhashini language placeholders
- Validation Lab with held-out synthetic engineering data, accuracy, macro precision/recall/F1 and confusion matrix
- Audit log for AI inference and human actions
- Demo seed data so the dashboard is not empty
- Automated unit tests for scoring and model behaviour

## What is NOT claimed

This is not a clinically validated mental-health model, not a production government integration, and not a suicide/deception/credibility detector. Training/validation data included here are synthetic engineering examples. Real deployment would require authorised datasets, clinical/domain validation, security assessment, DPIA/privacy controls, government API contracts, multilingual evaluation, human-in-the-loop governance and field pilots.

## Judge-safe wording

> “Samarthan is demonstrated as a functional engineering POC with a real ML inference pipeline. The prototype covers intake, multimodal signal extraction, personal-baseline monitoring, explainable dynamic risk scoring, prioritisation, human intervention, follow-up and audit. Current model metrics are engineering validation on synthetic data; production and clinical validation are planned pilot stages.”

## Windows + Python 3.14

Python 3.14 is supported by the current dependency ranges when compatible wheels are available. The app is deliberately kept free of packages that require a separate native audio/ML stack.

### 1. Open Command Prompt in this folder

```cmd
py -3.14 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train the local model

```cmd
python scripts\train_model.py
```

### 3. Start Samarthan

```cmd
python -m uvicorn app.main:app --reload
```

Open: http://127.0.0.1:8000

### 4. Run tests

In another terminal with the venv activated:

```cmd
python -m unittest discover -s tests -v
```

## Recommended judge demo (3–5 minutes)

1. **Dashboard** — show the closed-loop queue and system state.
2. **New Check-in** — click “Load High-Risk Demo”, then analyse.
3. Show **dynamic score + model probabilities + personal baseline + contributing signals**.
4. Open **Case** → show timeline, case milestones and proposed human action.
5. Click **Acknowledge / Assign Counsellor / Escalate** and show the audit entry.
6. Return to **Queue** — demonstrate prioritisation.
7. Open **Validation Lab** — show held-out engineering metrics and confusion matrix.
8. Open **Integrations** — demonstrate NHAA/e-Courts/PFMS adapters are architected as replaceable interfaces.

## Project structure

```text
Samarthan_60pct_POC/
  app/main.py                 # API, scoring engine, persistence, UI routes
  app/templates/              # dashboard/check-in/case/queue/validation/audit/integrations
  app/static/style.css
  scripts/train_model.py      # synthetic engineering dataset + real ML training
  models/                     # generated model artefacts
  data/                       # generated SQLite database
  tests/                      # automated tests
```
