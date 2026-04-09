---
title: HR Automation OpenEnv
emoji: "🤖"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# HR Automation OpenEnv

HR Automation OpenEnv is a real-world hiring workflow environment for evaluating agents on candidate screening, fit scoring, and shortlist decisions. It is designed for Scaler OpenEnv Phase-1 checks and for Docker deployment on Hugging Face Spaces.

## Environment Description

The environment simulates common backend and recruiting operations:

- reading a live hiring brief for a role
- reviewing one or more candidate profiles
- extracting relevant evidence from resumes
- scoring fit against role requirements
- making a shortlist, interview, or rejection decision

The environment is deterministic, so the same policy produces the same scores every run.

## OpenEnv API

The FastAPI server exposes these endpoints:

- `GET /` for health checks
- `POST /reset` to start an episode for `easy`, `medium`, or `hard`
- `POST /step` to submit an action
- `GET /state` to inspect the current structured environment state

## Action Space

- `extract_skills`: submit extracted evidence from candidate resumes
- `score_candidate`: submit a calibrated fit score or ranking
- `finalize_decision`: submit a shortlist, reject, or best-candidate decision
- `analyze`: one-shot baseline action that completes the whole task

## Observation Space

`GET /state` returns typed state with:

- `job`
- `candidates`
- `objective`
- `step_count`
- `history`
- `milestones`
- `expected_action`

`/reset` and `/step` return the OpenEnv transition format:

```json
{
  "observation": "string summary",
  "reward": 0.0,
  "done": false,
  "info": {}
}
```

## Tasks

- `easy`: screen one backend engineer and decide whether to schedule an interview
- `medium`: evaluate one analytics candidate and make a calibrated shortlist decision
- `hard`: rank multiple product candidates and choose the strongest candidate

Each task includes an agent grader and partial reward signals:

- skill extraction contributes up to `0.35`
- score calibration or ranking quality contributes up to `0.35`
- final decision correctness contributes up to `0.30`

Final rewards stay in the range `0.0` to `1.0`.

## Setup

### Local Python

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Open the app in your browser at:

```text
http://127.0.0.1:7860/
```

or:

```text
http://localhost:7860/
```

### Docker

```bash
docker build -t hr-automation-openenv .
docker run -p 7860:7860 hr-automation-openenv
```

## Baseline Inference

The baseline policy is deterministic and reproducible. It:

- reads `API_BASE_URL`
- reads `MODEL_NAME`
- reads `HF_TOKEN`
- initializes the OpenAI Python client
- calls `/reset`, `/state`, and `/step`
- runs `easy`, `medium`, and `hard`

Run it with:

```bash
API_BASE_URL=http://127.0.0.1:7860 MODEL_NAME=gpt-4o-mini HF_TOKEN=dummy python inference.py
```

Example logs:

```text
[START] task=easy
[STEP] action=analyze
[END] score=1.00
```

## Hugging Face Spaces

This repository is ready for Docker-based Spaces deployment:

1. Create a new Space with SDK set to `Docker`
2. Push this repository as-is
3. Spaces will build the included `Dockerfile`
4. The app listens on port `7860`

## Files Required By Phase-1

The project root contains:

- `app.py`
- `inference.py`
- `openenv.yaml`
- `Dockerfile`
- `requirements.txt`
