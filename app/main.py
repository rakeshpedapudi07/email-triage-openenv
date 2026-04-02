"""
Email Triage OpenEnv — FastAPI Backend
OpenEnv Round 1 Compliant
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import time
import os
from pathlib import Path

from app.models import (
    ResetRequest, ResetResponse,
    StepRequest, StepResponse,
    StateResponse, TaskListResponse,
    EnvInfoResponse,
)
from app.environment import EmailTriageEnv
from app.tasks import TASKS
from app.logger import get_recent_logs

# ── App ────────────────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(
    title="Email Triage OpenEnv",
    description="""
## 📧 Email Triage — OpenEnv Round 1

An RL-style environment where an agent triages emails by:
1. **Classifying** the email category
2. **Assigning** a priority level
3. **Generating** a response (if needed)
4. **Archiving** the email

### Action Flow
```
classify_email → set_priority → generate_response (optional) → archive_email
```

### Valid `action_type` values
| Value | Description |
|---|---|
| `classify_email` | Classify email into a category |
| `set_priority` | Assign priority (low/medium/high/urgent) |
| `generate_response` | Generate a reply (optional step) |
| `archive_email` | Archive and complete the episode |

### Valid `category` values
`work`, `personal`, `spam`, `newsletter`, `support`, `billing`, `other`

### Valid `priority` values
`low`, `medium`, `high`, `urgent`
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env store (per-session via task_id)
_envs: dict[str, EmailTriageEnv] = {}


def _get_or_create_env(task_id: str) -> EmailTriageEnv:
    if task_id not in _envs:
        if task_id not in TASKS:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found. Available: {list(TASKS.keys())}")
        _envs[task_id] = EmailTriageEnv(task_id, TASKS[task_id])
    return _envs[task_id]


# ── Static & Dashboard ────────────────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/dashboard", response_class=FileResponse, tags=["Info"])
def dashboard():
    """Live interactive agent playground dashboard."""
    dash = STATIC_DIR / "dashboard.html"
    if not dash.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(str(dash))


@app.get("/logs", tags=["Debugging"])
def get_logs(task_id: str = None, limit: int = 50):
    """Retrieve recent structured episode logs."""
    return {"logs": get_recent_logs(task_id=task_id, limit=limit)}


# ── Root ───────────────────────────────────────────────────────────────────────
@app.get("/", response_model=EnvInfoResponse, tags=["Info"])
def root():
    """Environment info and quick-start guide."""
    return EnvInfoResponse(
        name="Email Triage OpenEnv",
        version="1.0.0",
        description="RL environment for intelligent email triage",
        tasks=list(TASKS.keys()),
        action_types=["classify_email", "set_priority", "generate_response", "archive_email"],
        categories=["work", "personal", "spam", "newsletter", "support", "billing", "other"],
        priorities=["low", "medium", "high", "urgent"],
        action_flow=["classify_email", "set_priority", "generate_response (optional)", "archive_email"],
        docs_url="/docs",
    )


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Info"])
def health():
    """Health check for deployment platforms."""
    return {"status": "ok", "timestamp": time.time()}


# ── Tasks ──────────────────────────────────────────────────────────────────────
@app.get("/tasks", response_model=TaskListResponse, tags=["Environment"])
def list_tasks():
    """List all available tasks with metadata."""
    return TaskListResponse(
        tasks=[
            {
                "task_id": tid,
                "name": t["name"],
                "difficulty": t["difficulty"],
                "description": t["description"],
                "max_steps": t["max_steps"],
            }
            for tid, t in TASKS.items()
        ]
    )


# ── Reset ──────────────────────────────────────────────────────────────────────
@app.post("/reset", response_model=ResetResponse, tags=["Environment"])
def reset(req: ResetRequest):
    """
    Reset the environment for a given task.

    Returns the initial observation (email to triage).
    """
    task_id = req.task_id
    if task_id not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found. Available tasks: {list(TASKS.keys())}"
        )
    env = EmailTriageEnv(task_id, TASKS[task_id])
    _envs[task_id] = env
    obs = env.reset()
    return ResetResponse(
        task_id=task_id,
        observation=obs,
        info={"message": "Environment reset. Start with action_type='classify_email'"},
    )


# ── Step ───────────────────────────────────────────────────────────────────────
@app.post("/step", response_model=StepResponse, tags=["Environment"])
def step(req: StepRequest):
    """
    Take one step in the environment.

    ### Action Flow (must follow this order):
    1. `classify_email` — with `category` param
    2. `set_priority` — with `priority` param
    3. `generate_response` — with `response_text` param *(optional)*
    4. `archive_email` — finalizes episode

    ### Example (classify step):
    ```json
    {
      "task_id": "easy_triage",
      "action_type": "classify_email",
      "category": "work"
    }
    ```
    """
    env = _get_or_create_env(req.task_id)
    if not env.started:
        raise HTTPException(
            status_code=400,
            detail="Environment not reset. Call POST /reset first."
        )

    result = env.step(req)

    if "error" in result and result["error"]:
        # Return error in step response (not HTTP error) — agent can retry
        return StepResponse(**result)

    return StepResponse(**result)


# ── State ──────────────────────────────────────────────────────────────────────
@app.get("/state", response_model=StateResponse, tags=["Environment"])
def state(task_id: str):
    """
    Get the current environment state for a task.

    Query param: `task_id`
    """
    env = _get_or_create_env(task_id)
    return StateResponse(**env.get_state())


# ── Custom 422 handler ─────────────────────────────────────────────────────────
@app.exception_handler(422)
async def validation_exception_handler(request: Request, exc):
    body = await request.body()
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors() if hasattr(exc, "errors") else str(exc),
            "hint": "Valid action_type values: classify_email | set_priority | generate_response | archive_email",
            "body_received": body.decode() if body else None,
        },
    )
