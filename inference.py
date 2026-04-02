#!/usr/bin/env python3
"""
inference.py — Email Triage OpenEnv Baseline Agent

Uses OpenAI-compatible client to interact with the environment.
Follows EXACT OpenEnv log format.

Environment variables:
  API_BASE_URL   — LLM API base URL (e.g. https://api-inference.huggingface.co/v1)
  MODEL_NAME     — Model to use (e.g. mistralai/Mistral-7B-Instruct-v0.3)
  HF_TOKEN       — HuggingFace API token (used as OpenAI API key)
  ENV_BASE_URL   — Email Triage env URL (default: http://localhost:8000)

Usage:
  python inference.py
  python inference.py --task medium_triage
  python inference.py --task hard_triage --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

import requests
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────
API_BASE_URL  = os.environ.get("API_BASE_URL",  "https://api-inference.huggingface.co/v1")
MODEL_NAME    = os.environ.get("MODEL_NAME",    "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN      = os.environ.get("HF_TOKEN",      "")
ENV_BASE_URL  = os.environ.get("ENV_BASE_URL",  "http://localhost:8000")
ENV_NAME      = "email-triage-openenv"

# ── OpenAI Client ──────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy",  # HF uses token as api_key
)

# ── Logger (strict OpenEnv format) ────────────────────────────────────────────
_log_lines: list[str] = []

def log(msg: str):
    print(msg, flush=True)
    _log_lines.append(msg)

def log_start(task: str):
    log(f"[START] task={task} env={ENV_NAME} model={MODEL_NAME}")

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err_str = error if error else "null"
    log(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err_str}")

def log_end(success: bool, steps: int, rewards: list[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    log(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}")


# ── Env API Helpers ────────────────────────────────────────────────────────────
def env_reset(task_id: str) -> dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=10)
    r.raise_for_status()
    return r.json()

def env_step(payload: dict) -> dict[str, Any]:
    r = requests.post(f"{ENV_BASE_URL}/step", json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

def env_state(task_id: str) -> dict[str, Any]:
    r = requests.get(f"{ENV_BASE_URL}/state", params={"task_id": task_id}, timeout=10)
    r.raise_for_status()
    return r.json()


# ── LLM Helpers ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert email triage agent. Given an email, you must:
1. Classify it into exactly one category: work, personal, spam, newsletter, support, billing, other
2. Assign a priority: low, medium, high, urgent
3. Decide if a response is needed (not for spam or newsletters)
4. Archive the email to complete the episode

Always respond with a valid JSON object only. No explanation outside JSON."""

def llm_classify(email: dict) -> str:
    """Ask LLM to classify the email."""
    prompt = f"""Analyze this email and respond with JSON only:
{{
  "category": "<work|personal|spam|newsletter|support|billing|other>",
  "reasoning": "<brief reason>"
}}

Email:
Subject: {email.get('subject', '')}
From: {email.get('sender', '')}
Body: {email.get('body', '')[:500]}
"""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=150,
        temperature=0.1,
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown fences if present
    text = text.replace("```json", "").replace("```", "").strip()
    data = json.loads(text)
    return data.get("category", "other")


def llm_prioritize(email: dict, category: str) -> str:
    """Ask LLM to set priority."""
    prompt = f"""Given this email classified as '{category}', set the priority.
Respond with JSON only:
{{
  "priority": "<low|medium|high|urgent>",
  "reasoning": "<brief reason>"
}}

Subject: {email.get('subject', '')}
From: {email.get('sender', '')}
Category: {category}
"""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=100,
        temperature=0.1,
    )
    text = resp.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    data = json.loads(text)
    return data.get("priority", "medium")


def llm_should_respond(category: str) -> bool:
    """Determine if a response is needed based on category."""
    return category not in {"spam", "newsletter"}


def llm_generate_response(email: dict, category: str, priority: str) -> str:
    """Generate an appropriate email response."""
    prompt = f"""Write a professional email reply for this {category} email with {priority} priority.
Keep it concise (2-3 sentences). Respond with JSON only:
{{
  "response_text": "<your reply here>"
}}

Subject: {email.get('subject', '')}
From: {email.get('sender', '')}
"""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=200,
        temperature=0.3,
    )
    text = resp.choices[0].message.content.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    data = json.loads(text)
    return data.get("response_text", "Thank you for your email. We will follow up shortly.")


# ── Main Agent Loop ────────────────────────────────────────────────────────────
def run_episode(task_id: str, verbose: bool = False) -> bool:
    """Run a full triage episode. Returns True if successful."""
    log_start(task_id)

    # Reset
    try:
        reset_resp = env_reset(task_id)
    except Exception as e:
        log(f"[ERROR] Failed to reset: {e}")
        log_end(success=False, steps=0, rewards=[])
        return False

    obs = reset_resp["observation"]
    step_num = 0
    rewards: list[float] = []
    category = None
    priority = None

    # ── Step 1: Classify ───────────────────────────────────────────────────────
    try:
        category = llm_classify(obs)
        if verbose:
            print(f"  → LLM classified as: {category}")
    except Exception as e:
        category = "other"
        if verbose:
            print(f"  → LLM classify failed ({e}), defaulting to 'other'")

    step_num += 1
    result = env_step({
        "task_id": task_id,
        "action_type": "classify_email",
        "category": category,
    })
    reward = result.get("reward", 0.0)
    done = result.get("done", False)
    error = result.get("error")
    rewards.append(reward)
    log_step(step_num, f"classify_email({category})", reward, done, error)
    if done:
        log_end(success=not bool(error), steps=step_num, rewards=rewards)
        return not bool(error)

    # ── Step 2: Set Priority ───────────────────────────────────────────────────
    try:
        priority = llm_prioritize(obs, category)
        if verbose:
            print(f"  → LLM priority: {priority}")
    except Exception as e:
        priority = "medium"
        if verbose:
            print(f"  → LLM priority failed ({e}), defaulting to 'medium'")

    step_num += 1
    result = env_step({
        "task_id": task_id,
        "action_type": "set_priority",
        "priority": priority,
    })
    reward = result.get("reward", 0.0)
    done = result.get("done", False)
    error = result.get("error")
    rewards.append(reward)
    log_step(step_num, f"set_priority({priority})", reward, done, error)
    if done:
        log_end(success=not bool(error), steps=step_num, rewards=rewards)
        return not bool(error)

    # ── Step 3: Generate Response (optional) ───────────────────────────────────
    next_action = result.get("next_expected_action", "archive_email")
    if next_action == "generate_response" or llm_should_respond(category):
        try:
            response_text = llm_generate_response(obs, category, priority)
            if verbose:
                print(f"  → LLM response: {response_text[:80]}...")
        except Exception as e:
            response_text = "Thank you for your email. We will get back to you shortly."
            if verbose:
                print(f"  → LLM response failed ({e}), using default")

        step_num += 1
        result = env_step({
            "task_id": task_id,
            "action_type": "generate_response",
            "response_text": response_text,
        })
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        error = result.get("error")
        rewards.append(reward)
        log_step(step_num, "generate_response", reward, done, error)
        if done:
            log_end(success=not bool(error), steps=step_num, rewards=rewards)
            return not bool(error)

    # ── Step 4: Archive ────────────────────────────────────────────────────────
    step_num += 1
    result = env_step({
        "task_id": task_id,
        "action_type": "archive_email",
    })
    reward = result.get("reward", 0.0)
    done = result.get("done", False)
    error = result.get("error")
    rewards.append(reward)
    log_step(step_num, "archive_email", reward, done, error)

    success = done and not bool(error)
    log_end(success=success, steps=step_num, rewards=rewards)
    return success


# ── Entrypoint ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Email Triage OpenEnv Inference Agent")
    parser.add_argument("--task", default="easy_triage",
                        choices=["easy_triage", "medium_triage", "hard_triage"],
                        help="Task to run (default: easy_triage)")
    parser.add_argument("--all", action="store_true",
                        help="Run all tasks sequentially")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show LLM decision details")
    args = parser.parse_args()

    # Validate env server
    try:
        r = requests.get(f"{ENV_BASE_URL}/health", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Cannot reach environment server at {ENV_BASE_URL}: {e}")
        print("  → Start it with: uvicorn app.main:app --reload")
        sys.exit(1)

    tasks = ["easy_triage", "medium_triage", "hard_triage"] if args.all else [args.task]

    all_success = True
    for task_id in tasks:
        print(f"\n{'='*60}")
        print(f"Running task: {task_id}")
        print(f"{'='*60}")
        success = run_episode(task_id, verbose=args.verbose)
        if not success:
            all_success = False
        time.sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Overall result: {'✅ PASSED' if all_success else '❌ FAILED'}")
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
