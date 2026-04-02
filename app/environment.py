"""
EmailTriageEnv — Core RL Environment Logic

Episode flow:
  CLASSIFY → PRIORITY → [RESPONSE optional] → ARCHIVE → DONE

Reward shaping:
  +0.30  correct classification
  -0.15  wrong classification
  +0.25  correct priority
  -0.10  wrong priority
  +0.20  appropriate response generated
  -0.10  unnecessary response (spam/newsletter)
  +0.25  archive (completion bonus)
  -0.05  per out-of-order action attempt
"""

from __future__ import annotations
from typing import Any, Optional
from app.logger import log_start, log_step, log_end
from app.models import (
    StepRequest, EmailObservation,
    ActionType, EmailCategory, Priority,
)


# ── Phase FSM ──────────────────────────────────────────────────────────────────
PHASES = ["classify", "priority", "response_or_archive", "archive", "done"]

PHASE_EXPECTED_ACTIONS: dict[str, list[str]] = {
    "classify":            ["classify_email"],
    "priority":            ["set_priority"],
    "response_or_archive": ["generate_response", "archive_email"],
    "archive":             ["archive_email"],
    "done":                [],
}

# Category → correct priority mapping (ground truth per task)
PRIORITY_MAP: dict[str, str] = {
    "work":       "high",
    "support":    "urgent",
    "billing":    "urgent",
    "personal":   "medium",
    "newsletter": "low",
    "spam":       "low",
    "other":      "medium",
}

# Categories that should NOT get a response
RESPONSE_NOT_NEEDED = {"spam", "newsletter"}


class EmailTriageEnv:
    def __init__(self, task_id: str, task_cfg: dict):
        self.task_id = task_id
        self.task_cfg = task_cfg
        self.started = False

        # Episode state
        self._obs: Optional[EmailObservation] = None
        self._phase = "classify"
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._truncated = False

        # Tracked decisions
        self._classification: Optional[str] = None
        self._priority: Optional[str] = None
        self._has_response = False
        self._rewards: list[float] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(self) -> EmailObservation:
        email_data = self.task_cfg["email"]
        self._obs = EmailObservation(**email_data)
        self._phase = "classify"
        self._step_count = 0
        self._total_reward = 0.0
        self._done = False
        self._truncated = False
        self._classification = None
        self._priority = None
        self._has_response = False
        self._rewards = []
        self.started = True
        log_start(task_id=self.task_id)
        return self._obs

    def step(self, req: StepRequest) -> dict[str, Any]:
        if self._done:
            return self._step_result(
                reward=0.0, done=True,
                error="Episode is done. Call /reset to start a new episode.",
                next_action=None,
            )

        self._step_count += 1

        if self._step_count > self.task_cfg["max_steps"]:
            self._done = True
            self._truncated = True
            return self._step_result(
                reward=-0.1, done=True,
                error=f"Max steps ({self.task_cfg['max_steps']}) exceeded.",
                next_action=None,
            )
        action = req.action_type.value

        # ── Validate phase ─────────────────────────────────────────────────────
        valid_actions = PHASE_EXPECTED_ACTIONS.get(self._phase, [])
        if action not in valid_actions:
            penalty = -0.05
            self._total_reward += penalty
            self._rewards.append(penalty)
            return self._step_result(
                reward=penalty,
                done=False,
                error=(
                    f"Out-of-order action '{action}' in phase '{self._phase}'. "
                    f"Expected one of: {valid_actions}"
                ),
                next_action=valid_actions[0] if valid_actions else None,
            )

        # ── Dispatch ───────────────────────────────────────────────────────────
        if action == "classify_email":
            return self._handle_classify(req)
        elif action == "set_priority":
            return self._handle_priority(req)
        elif action == "generate_response":
            return self._handle_response(req)
        elif action == "archive_email":
            return self._handle_archive(req)

        return self._step_result(reward=0.0, done=False, error="Unknown action", next_action=None)

    def get_state(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_count": self._step_count,
            "done": self._done,
            "current_phase": self._phase,
            "classification": self._classification,
            "priority": self._priority,
            "has_response": self._has_response,
            "total_reward": round(self._total_reward, 4),
            "observation": self._obs,
        }

    # ── Action Handlers ────────────────────────────────────────────────────────

    def _handle_classify(self, req: StepRequest) -> dict[str, Any]:
        if req.category is None:
            return self._step_result(
                reward=-0.05, done=False,
                error="'category' is required for classify_email action. "
                      "Valid values: work, personal, spam, newsletter, support, billing, other",
                next_action="classify_email",
            )

        cat = req.category.value
        self._classification = cat
        correct_cat = self.task_cfg.get("correct_category")

        if correct_cat and cat == correct_cat:
            reward = 0.30
            msg = f"✅ Correct classification: '{cat}'"
        elif correct_cat:
            reward = -0.15
            msg = f"❌ Wrong classification: '{cat}' (email is: '{correct_cat}')"
        else:
            reward = 0.15  # partial credit when no ground truth
            msg = f"Classified as '{cat}'"

        self._total_reward += reward
        self._rewards.append(reward)
        self._phase = "priority"

        return self._step_result(
            reward=reward, done=False, error=None,
            next_action="set_priority",
            info={"message": msg, "classification": cat},
        )

    def _handle_priority(self, req: StepRequest) -> dict[str, Any]:
        if req.priority is None:
            return self._step_result(
                reward=-0.05, done=False,
                error="'priority' is required for set_priority action. "
                      "Valid values: low, medium, high, urgent",
                next_action="set_priority",
            )

        prio = req.priority.value
        self._priority = prio
        cat = self._classification or "other"
        expected_prio = self.task_cfg.get("correct_priority") or PRIORITY_MAP.get(cat, "medium")

        if prio == expected_prio:
            reward = 0.25
            msg = f"✅ Correct priority: '{prio}'"
        else:
            reward = -0.10
            msg = f"❌ Wrong priority: '{prio}' (expected: '{expected_prio}')"

        self._total_reward += reward
        self._rewards.append(reward)
        self._phase = "response_or_archive"

        # Determine if response is needed
        needs_response = cat not in RESPONSE_NOT_NEEDED
        return self._step_result(
            reward=reward, done=False, error=None,
            next_action="generate_response" if needs_response else "archive_email",
            info={
                "message": msg,
                "priority": prio,
                "response_recommended": needs_response,
            },
        )

    def _handle_response(self, req: StepRequest) -> dict[str, Any]:
        if req.response_text is None or not req.response_text.strip():
            return self._step_result(
                reward=-0.05, done=False,
                error="'response_text' is required for generate_response action.",
                next_action="generate_response",
            )

        cat = self._classification or "other"
        is_unnecessary = cat in RESPONSE_NOT_NEEDED

        if is_unnecessary:
            reward = -0.10
            msg = f"⚠️ Unnecessary response for '{cat}' emails."
        else:
            # Reward based on response quality (length heuristic)
            text = req.response_text.strip()
            if len(text) >= 20:
                reward = 0.20
                msg = "✅ Response generated."
            else:
                reward = 0.05
                msg = "⚠️ Response too short — consider a more detailed reply."

        self._has_response = True
        self._total_reward += reward
        self._rewards.append(reward)
        self._phase = "archive"

        return self._step_result(
            reward=reward, done=False, error=None,
            next_action="archive_email",
            info={"message": msg},
        )

    def _handle_archive(self, req: StepRequest) -> dict[str, Any]:
        # Completion reward
        reward = 0.25
        self._total_reward += reward
        self._rewards.append(reward)
        self._done = True
        self._phase = "done"

        # Grader: normalize total reward to 0–1
        score = self._compute_score()

        log_end(
            task_id=self.task_id,
            success=True,
            steps=self._step_count,
            rewards=self._rewards,
        )
        return self._step_result(
            reward=reward, done=True, error=None,
            next_action=None,
            info={
                "message": "✅ Email archived. Episode complete.",
                "score": score,
                "total_reward": round(self._total_reward, 4),
                "rewards_breakdown": self._rewards,
                "steps_taken": self._step_count,
            },
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _compute_score(self) -> float:
        """Normalize total reward to [0, 1] for OpenEnv grader."""
        max_possible = 0.30 + 0.25 + 0.20 + 0.25  # 1.0
        raw = self._total_reward
        score = max(0.0, min(1.0, raw / max_possible))
        return round(score, 4)

    def _step_result(
        self,
        reward: float,
        done: bool,
        error: Optional[str],
        next_action: Optional[str],
        info: Optional[dict] = None,
    ) -> dict[str, Any]:
        return {
            "observation": self._obs,
            "reward": round(reward, 4),
            "done": done,
            "truncated": self._truncated,
            "error": error,
            "next_expected_action": next_action,
            "info": {
                **(info or {}),
                "step": self._step_count,
                "phase": self._phase,
                "total_reward": round(self._total_reward, 4),
            },
        }
