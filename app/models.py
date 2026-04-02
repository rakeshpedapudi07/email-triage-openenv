"""
Pydantic models for Email Triage OpenEnv.
All enums use lowercase values to match API expectations.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """Valid action types — use exactly these lowercase string values."""
    classify_email    = "classify_email"
    set_priority      = "set_priority"
    generate_response = "generate_response"
    archive_email     = "archive_email"


class EmailCategory(str, Enum):
    """Valid email categories."""
    work       = "work"
    personal   = "personal"
    spam       = "spam"
    newsletter = "newsletter"
    support    = "support"
    billing    = "billing"
    other      = "other"


class Priority(str, Enum):
    """Valid priority levels."""
    low    = "low"
    medium = "medium"
    high   = "high"
    urgent = "urgent"


# ── Request Models ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = Field(
        ...,
        description="Task identifier",
        examples=["easy_triage"],
        json_schema_extra={"example": "easy_triage"},
    )

    model_config = {
        "json_schema_extra": {
            "example": {"task_id": "easy_triage"}
        }
    }


class StepRequest(BaseModel):
    task_id: str = Field(
        ...,
        description="Task identifier (must match reset task_id)",
        examples=["easy_triage"],
    )
    action_type: ActionType = Field(
        ...,
        description=(
            "Action to take. Must follow order: "
            "classify_email → set_priority → generate_response (opt) → archive_email"
        ),
        examples=["classify_email"],
    )
    category: Optional[EmailCategory] = Field(
        None,
        description="Required when action_type='classify_email'",
        examples=["work"],
    )
    priority: Optional[Priority] = Field(
        None,
        description="Required when action_type='set_priority'",
        examples=["high"],
    )
    response_text: Optional[str] = Field(
        None,
        description="Required when action_type='generate_response'",
        examples=["Thank you for your email. We will get back to you shortly."],
        max_length=2000,
    )

    @field_validator("action_type", mode="before")
    @classmethod
    def normalize_action_type(cls, v):
        """Accept both 'classify_email' and 'CLASSIFY_EMAIL' forms."""
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("priority", mode="before")
    @classmethod
    def normalize_priority(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Step 1 — Classify email",
                    "value": {
                        "task_id": "easy_triage",
                        "action_type": "classify_email",
                        "category": "work",
                    },
                },
                {
                    "summary": "Step 2 — Set priority",
                    "value": {
                        "task_id": "easy_triage",
                        "action_type": "set_priority",
                        "priority": "high",
                    },
                },
                {
                    "summary": "Step 3 — Generate response (optional)",
                    "value": {
                        "task_id": "easy_triage",
                        "action_type": "generate_response",
                        "response_text": "Thank you, I'll review your request shortly.",
                    },
                },
                {
                    "summary": "Step 4 — Archive email",
                    "value": {
                        "task_id": "easy_triage",
                        "action_type": "archive_email",
                    },
                },
            ]
        }
    }


# ── Response Models ────────────────────────────────────────────────────────────

class EmailObservation(BaseModel):
    subject: str
    sender: str
    body: str
    received_at: str
    has_attachment: bool = False
    metadata: dict[str, Any] = {}


class ResetResponse(BaseModel):
    task_id: str
    observation: EmailObservation
    info: dict[str, Any] = {}


class StepResponse(BaseModel):
    observation: Optional[EmailObservation] = None
    reward: float = Field(..., ge=-1.0, le=1.0, description="Reward for this step (0.0–1.0 positive)")
    done: bool = Field(..., description="True when episode is complete")
    truncated: bool = False
    info: dict[str, Any] = {}
    error: Optional[str] = Field(None, description="Error message if action was invalid")
    next_expected_action: Optional[str] = Field(None, description="Hint: what action to take next")


class StateResponse(BaseModel):
    task_id: str
    step_count: int
    done: bool
    current_phase: str
    classification: Optional[str] = None
    priority: Optional[str] = None
    has_response: bool = False
    total_reward: float
    observation: Optional[EmailObservation] = None


class TaskInfo(BaseModel):
    task_id: str
    name: str
    difficulty: str
    description: str
    max_steps: int


class TaskListResponse(BaseModel):
    tasks: list[TaskInfo]


class EnvInfoResponse(BaseModel):
    name: str
    version: str
    description: str
    tasks: list[str]
    action_types: list[str]
    categories: list[str]
    priorities: list[str]
    action_flow: list[str]
    docs_url: str
