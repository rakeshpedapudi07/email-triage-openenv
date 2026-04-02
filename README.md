# 📧 Email Triage OpenEnv

> An RL-style environment for intelligent email triage — OpenEnv Round 1 Submission

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Round%201-blue)](https://openenv.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![HuggingFace](https://img.shields.io/badge/HF%20Spaces-Deployable-yellow)](https://huggingface.co/spaces)

---

## 🧩 Problem Description

Email overload is a real productivity problem. This environment trains agents to triage emails intelligently — the same workflow a skilled human assistant uses:

1. **Classify** the email type (work, support, spam, billing, etc.)
2. **Prioritize** it correctly (low → urgent)
3. **Respond** when appropriate (not to spam!)
4. **Archive** to complete the task

Agents that get this right earn maximum reward. Wrong classifications, unnecessary responses, and out-of-order actions are penalized — just like in real life.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Email Triage OpenEnv                   │
│                                                          │
│  ┌──────────┐    POST /reset     ┌─────────────────┐    │
│  │  Agent   │ ────────────────▶  │  EmailTriageEnv │    │
│  │(LLM/RL)  │ ◀────────────────  │  (FSM Engine)   │    │
│  │          │    observation      │                 │    │
│  │          │                    │  Phase FSM:     │    │
│  │          │    POST /step       │  classify       │    │
│  │          │ ────────────────▶  │  ↓ priority     │    │
│  │          │ ◀────────────────  │  ↓ response?    │    │
│  │          │  reward/done/next  │  ↓ archive      │    │
│  └──────────┘                    └─────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  FastAPI Server  │  Pydantic Models  │  3 Tasks  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Environment info |
| `/health` | GET | Health check |
| `/tasks` | GET | List all tasks |
| `/reset` | POST | Start a new episode |
| `/step` | POST | Take an action |
| `/state` | GET | Get current state |
| `/docs` | GET | Swagger UI |

---

## 🎮 Action Space

Actions must be taken **in order**:

```
classify_email → set_priority → generate_response (optional) → archive_email
```

| `action_type` | Required Param | Valid Values |
|---|---|---|
| `classify_email` | `category` | `work`, `personal`, `spam`, `newsletter`, `support`, `billing`, `other` |
| `set_priority` | `priority` | `low`, `medium`, `high`, `urgent` |
| `generate_response` | `response_text` | Any string (2000 char max) |
| `archive_email` | *(none)* | — |

---

## 👁️ Observation Space

Each episode presents an email:

```json
{
  "subject": "Urgent: Q4 Budget Review Meeting Tomorrow",
  "sender": "manager@company.com",
  "body": "Hi Team, the meeting has been moved to...",
  "received_at": "2024-01-15T09:00:00Z",
  "has_attachment": true,
  "metadata": {
    "sender_domain": "company.com",
    "is_internal": true,
    "spam_score": 0.1
  }
}
```

---

## 🏆 Reward Shaping

| Action | Condition | Reward |
|--------|-----------|--------|
| `classify_email` | Correct category | **+0.30** |
| `classify_email` | Wrong category | **-0.15** |
| `set_priority` | Correct priority | **+0.25** |
| `set_priority` | Wrong priority | **-0.10** |
| `generate_response` | Appropriate + quality | **+0.20** |
| `generate_response` | Unnecessary (spam) | **-0.10** |
| `archive_email` | Episode completion | **+0.25** |
| Any action | Out of order | **-0.05** |

**Max possible score: 1.0** (all correct, no penalties)

Final score is normalized: `score = max(0, total_reward) / 1.0`

---

## 📋 Tasks

| Task ID | Difficulty | Description |
|---------|-----------|-------------|
| `easy_triage` | 🟢 Easy | Clear work email — obvious category/priority |
| `medium_triage` | 🟡 Medium | Angry customer billing email — needs urgency detection |
| `hard_triage` | 🔴 Hard | Sophisticated phishing email — must resist responding |

---

## 🚀 Setup & Run

### Option 1: Local Python

```bash
# Clone / set up
git clone <your-repo>
cd email-triage-openenv

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --reload --port 8000

# Visit docs
open http://localhost:8000/docs
```

### Option 2: Docker

```bash
# Build
docker build -t email-triage-openenv .

# Run
docker run -p 8000:8000 email-triage-openenv

# Visit docs
open http://localhost:8000/docs
```

### Option 3: HuggingFace Spaces

1. Create a new Space with SDK: **Docker**
2. Upload all files (including `Dockerfile`)
3. Set `app_port: 8000` in Space settings
4. Space auto-builds and deploys

---

## 🤖 Running the Inference Agent

```bash
# Set your LLM credentials
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="your_hf_token_here"
export ENV_BASE_URL="http://localhost:8000"

# Run easy task
python inference.py --task easy_triage

# Run all tasks
python inference.py --all --verbose
```

### Log Output Format (OpenEnv Standard)

```
[START] task=easy_triage env=email-triage-openenv model=mistralai/Mistral-7B-Instruct-v0.3
[STEP] step=1 action=classify_email(work) reward=0.30 done=false error=null
[STEP] step=2 action=set_priority(high) reward=0.25 done=false error=null
[STEP] step=3 action=generate_response reward=0.20 done=false error=null
[STEP] step=4 action=archive_email reward=0.25 done=true error=null
[END] success=true steps=4 rewards=0.30,0.25,0.20,0.25
```

---

## 🧪 Example API Usage

### 1. Reset environment

```bash
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_triage"}'
```

### 2. Classify email

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_triage", "action_type": "classify_email", "category": "work"}'
```

### 3. Set priority

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_triage", "action_type": "set_priority", "priority": "high"}'
```

### 4. Generate response

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_triage", "action_type": "generate_response", "response_text": "Thank you, I will attend the meeting and review the budget beforehand."}'
```

### 5. Archive

```bash
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy_triage", "action_type": "archive_email"}'
```

---

## 📐 OpenEnv Compliance Checklist

- [x] Typed Pydantic models (v2)
- [x] `/reset`, `/step`, `/state`, `/tasks` endpoints
- [x] `openenv.yaml` present and valid
- [x] Tasks include graders with weights
- [x] Scores normalized between 0.0–1.0
- [x] `inference.py` with OpenAI client
- [x] Strict `[START]/[STEP]/[END]` log format
- [x] Dockerfile (production-ready, multi-stage)
- [x] HuggingFace Spaces compatible
- [x] Swagger UI with examples at `/docs`
- [x] `/health` endpoint for deployment
- [x] Enum case-normalization (accepts UPPERCASE too)
- [x] Clear error messages with `next_expected_action` hints

---

## 🗂️ Project Structure

```
email-triage-openenv/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, routes
│   ├── models.py        # Pydantic models, enums
│   ├── environment.py   # RL environment FSM logic
│   └── tasks.py         # Task definitions + graders
├── inference.py         # LLM agent baseline
├── openenv.yaml         # OpenEnv specification
├── Dockerfile           # Production Docker image
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 📄 License

MIT License — see LICENSE file.

---

*Built for OpenEnv Round 1 Hackathon 🏆*
