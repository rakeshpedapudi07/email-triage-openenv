"""
Structured logger for Email Triage OpenEnv.
Writes OpenEnv-format logs to ./logs/ directory.
"""

from __future__ import annotations
import os
import time
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

LOG_DIR = Path(os.environ.get("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_file(task_id: str) -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"{task_id}_{date}.jsonl"


def log_event(task_id: str, event: dict):
    """Append a structured event to the task's JSONL log file."""
    record = {"ts": _ts(), "task_id": task_id, **event}
    with _lock:
        with open(_log_file(task_id), "a") as f:
            f.write(json.dumps(record) + "\n")


def log_start(task_id: str, model: str = ""):
    line = f"[START] task={task_id} env=email-triage-openenv model={model}"
    print(line, flush=True)
    log_event(task_id, {"type": "start", "model": model, "raw": line})


def log_step(task_id: str, step: int, action: str, reward: float,
             done: bool, error: Optional[str] = None):
    err_str = error if error else "null"
    line = (f"[STEP] step={step} action={action} reward={reward:.2f} "
            f"done={str(done).lower()} error={err_str}")
    print(line, flush=True)
    log_event(task_id, {
        "type": "step", "step": step, "action": action,
        "reward": reward, "done": done, "error": error, "raw": line,
    })


def log_end(task_id: str, success: bool, steps: int, rewards: list[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    line = f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}"
    print(line, flush=True)
    log_event(task_id, {
        "type": "end", "success": success, "steps": steps,
        "rewards": rewards, "total_reward": round(sum(rewards), 4),
        "raw": line,
    })


def get_recent_logs(task_id: Optional[str] = None, limit: int = 50) -> list[dict]:
    """Read recent log events across all task log files."""
    files = sorted(LOG_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if task_id:
        files = [f for f in files if f.name.startswith(task_id)]

    events = []
    for f in files[:5]:  # read latest 5 files
        try:
            lines = f.read_text().strip().splitlines()
            for line in reversed(lines):
                events.append(json.loads(line))
                if len(events) >= limit:
                    break
        except Exception:
            pass
        if len(events) >= limit:
            break
    return events[:limit]
