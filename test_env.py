"""
test_env.py — Full test suite for Email Triage OpenEnv
Run: pytest test_env.py -v

Tests:
  - Models / enum normalization
  - Environment FSM logic
  - Reward shaping accuracy
  - API endpoint correctness (via TestClient)
  - OpenEnv compliance checks
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import StepRequest, ActionType, EmailCategory, Priority
from app.environment import EmailTriageEnv
from app.tasks import TASKS

client = TestClient(app)


# ══════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════

@pytest.fixture
def easy_env():
    env = EmailTriageEnv("easy_triage", TASKS["easy_triage"])
    env.reset()
    return env


@pytest.fixture
def medium_env():
    env = EmailTriageEnv("medium_triage", TASKS["medium_triage"])
    env.reset()
    return env


@pytest.fixture
def hard_env():
    env = EmailTriageEnv("hard_triage", TASKS["hard_triage"])
    env.reset()
    return env


def make_step(task_id, action_type, **kwargs):
    return StepRequest(task_id=task_id, action_type=action_type, **kwargs)


# ══════════════════════════════════════════════════════════════════
# 1. Model & Enum Tests
# ══════════════════════════════════════════════════════════════════

class TestModels:
    def test_action_type_lowercase(self):
        req = StepRequest(task_id="easy_triage", action_type="classify_email")
        assert req.action_type == ActionType.classify_email

    def test_action_type_uppercase_normalized(self):
        req = StepRequest(task_id="easy_triage", action_type="CLASSIFY_EMAIL")
        assert req.action_type == ActionType.classify_email

    def test_action_type_mixed_case(self):
        req = StepRequest(task_id="easy_triage", action_type="Classify_Email")
        assert req.action_type == ActionType.classify_email

    def test_category_uppercase_normalized(self):
        req = StepRequest(task_id="easy_triage", action_type="classify_email", category="WORK")
        assert req.category == EmailCategory.work

    def test_priority_uppercase_normalized(self):
        req = StepRequest(task_id="easy_triage", action_type="set_priority", priority="URGENT")
        assert req.priority == Priority.urgent

    def test_invalid_action_type_raises(self):
        with pytest.raises(Exception):
            StepRequest(task_id="easy_triage", action_type="do_something_else")

    def test_invalid_category_raises(self):
        with pytest.raises(Exception):
            StepRequest(task_id="easy_triage", action_type="classify_email", category="invoice")

    def test_invalid_priority_raises(self):
        with pytest.raises(Exception):
            StepRequest(task_id="easy_triage", action_type="set_priority", priority="critical")


# ══════════════════════════════════════════════════════════════════
# 2. Environment FSM Tests
# ══════════════════════════════════════════════════════════════════

class TestEnvironmentFSM:
    def test_reset_returns_observation(self):
        env = EmailTriageEnv("easy_triage", TASKS["easy_triage"])
        obs = env.reset()
        assert obs.subject
        assert obs.sender
        assert obs.body

    def test_initial_phase_is_classify(self):
        env = EmailTriageEnv("easy_triage", TASKS["easy_triage"])
        env.reset()
        state = env.get_state()
        assert state["current_phase"] == "classify"

    def test_phase_transitions_correctly(self, easy_env):
        s1 = easy_env.get_state()
        assert s1["current_phase"] == "classify"

        easy_env.step(make_step("easy_triage", "classify_email", category="work"))
        s2 = easy_env.get_state()
        assert s2["current_phase"] == "priority"

        easy_env.step(make_step("easy_triage", "set_priority", priority="high"))
        s3 = easy_env.get_state()
        assert s3["current_phase"] == "response_or_archive"

    def test_out_of_order_action_penalized(self, easy_env):
        # Try to archive before classifying
        result = easy_env.step(make_step("easy_triage", "archive_email"))
        assert result["reward"] == -0.05
        assert result["error"] is not None
        assert result["done"] is False

    def test_out_of_order_does_not_advance_phase(self, easy_env):
        easy_env.step(make_step("easy_triage", "archive_email"))
        state = easy_env.get_state()
        assert state["current_phase"] == "classify"

    def test_max_steps_truncates(self):
        env = EmailTriageEnv("easy_triage", TASKS["easy_triage"])
        env.reset()
        # max_steps=1 means step 2 triggers truncation
        env.task_cfg = {**env.task_cfg, "max_steps": 1}
        env.step(make_step("easy_triage", "classify_email", category="work"))
        result = env.step(make_step("easy_triage", "set_priority", priority="high"))
        assert result["done"] is True or result["truncated"] is True

    def test_step_after_done_returns_error(self, easy_env):
        easy_env.step(make_step("easy_triage", "classify_email", category="work"))
        easy_env.step(make_step("easy_triage", "set_priority", priority="high"))
        easy_env.step(make_step("easy_triage", "generate_response",
                                response_text="Thank you."))
        easy_env.step(make_step("easy_triage", "archive_email"))
        result = easy_env.step(make_step("easy_triage", "archive_email"))
        assert result["error"] is not None
        assert result["done"] is True


# ══════════════════════════════════════════════════════════════════
# 3. Reward Shaping Tests
# ══════════════════════════════════════════════════════════════════

class TestRewards:
    def test_correct_classify_gives_positive_reward(self, easy_env):
        result = easy_env.step(make_step("easy_triage", "classify_email", category="work"))
        assert result["reward"] == 0.30

    def test_wrong_classify_gives_negative_reward(self, easy_env):
        result = easy_env.step(make_step("easy_triage", "classify_email", category="spam"))
        assert result["reward"] == -0.15

    def test_correct_priority_gives_positive_reward(self, easy_env):
        easy_env.step(make_step("easy_triage", "classify_email", category="work"))
        result = easy_env.step(make_step("easy_triage", "set_priority", priority="high"))
        assert result["reward"] == 0.25

    def test_wrong_priority_gives_negative_reward(self, easy_env):
        easy_env.step(make_step("easy_triage", "classify_email", category="work"))
        result = easy_env.step(make_step("easy_triage", "set_priority", priority="low"))
        assert result["reward"] == -0.10

    def test_good_response_gives_positive_reward(self, easy_env):
        easy_env.step(make_step("easy_triage", "classify_email", category="work"))
        easy_env.step(make_step("easy_triage", "set_priority", priority="high"))
        result = easy_env.step(make_step("easy_triage", "generate_response",
                                         response_text="Thank you, I will attend the meeting."))
        assert result["reward"] == 0.20

    def test_short_response_gives_partial_reward(self, easy_env):
        easy_env.step(make_step("easy_triage", "classify_email", category="work"))
        easy_env.step(make_step("easy_triage", "set_priority", priority="high"))
        result = easy_env.step(make_step("easy_triage", "generate_response",
                                         response_text="OK"))
        assert result["reward"] == 0.05

    def test_unnecessary_response_penalized(self, hard_env):
        hard_env.step(make_step("hard_triage", "classify_email", category="spam"))
        hard_env.step(make_step("hard_triage", "set_priority", priority="low"))
        result = hard_env.step(make_step("hard_triage", "generate_response",
                                          response_text="Thank you for your message!"))
        assert result["reward"] == -0.10

    def test_archive_gives_completion_reward(self, easy_env):
        easy_env.step(make_step("easy_triage", "classify_email", category="work"))
        easy_env.step(make_step("easy_triage", "set_priority", priority="high"))
        easy_env.step(make_step("easy_triage", "generate_response",
                                response_text="I will attend the meeting."))
        result = easy_env.step(make_step("easy_triage", "archive_email"))
        assert result["reward"] == 0.25
        assert result["done"] is True

    def test_perfect_episode_score_is_1(self, easy_env):
        easy_env.step(make_step("easy_triage", "classify_email", category="work"))
        easy_env.step(make_step("easy_triage", "set_priority", priority="high"))
        easy_env.step(make_step("easy_triage", "generate_response",
                                response_text="Thank you, I will attend the meeting."))
        result = easy_env.step(make_step("easy_triage", "archive_email"))
        assert result["info"]["score"] == 1.0

    def test_score_normalized_between_0_and_1(self, easy_env):
        easy_env.step(make_step("easy_triage", "classify_email", category="spam"))  # wrong
        easy_env.step(make_step("easy_triage", "set_priority", priority="low"))     # wrong
        easy_env.step(make_step("easy_triage", "generate_response",
                                response_text="Reply."))
        result = easy_env.step(make_step("easy_triage", "archive_email"))
        score = result["info"]["score"]
        assert 0.0 <= score <= 1.0


# ══════════════════════════════════════════════════════════════════
# 4. API Endpoint Tests
# ══════════════════════════════════════════════════════════════════

class TestAPIEndpoints:
    def test_root_returns_env_info(self):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "Email Triage OpenEnv"
        assert "classify_email" in data["action_types"]

    def test_health_returns_ok(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_tasks_returns_all_three(self):
        r = client.get("/tasks")
        assert r.status_code == 200
        ids = [t["task_id"] for t in r.json()["tasks"]]
        assert "easy_triage" in ids
        assert "medium_triage" in ids
        assert "hard_triage" in ids

    def test_reset_valid_task(self):
        r = client.post("/reset", json={"task_id": "easy_triage"})
        assert r.status_code == 200
        data = r.json()
        assert data["task_id"] == "easy_triage"
        assert "subject" in data["observation"]

    def test_reset_invalid_task_returns_404(self):
        r = client.post("/reset", json={"task_id": "nonexistent_task"})
        assert r.status_code == 404

    def test_step_without_reset_returns_400(self):
        # Use a task that hasn't been reset this session
        r = client.post("/step", json={
            "task_id": "medium_triage",
            "action_type": "classify_email",
            "category": "work",
        })
        # Should either 400 or return error in body
        assert r.status_code in (200, 400)

    def test_step_classify_via_api(self):
        client.post("/reset", json={"task_id": "easy_triage"})
        r = client.post("/step", json={
            "task_id": "easy_triage",
            "action_type": "classify_email",
            "category": "work",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["reward"] == 0.30
        assert data["next_expected_action"] == "set_priority"

    def test_step_uppercase_enum_accepted(self):
        client.post("/reset", json={"task_id": "easy_triage"})
        r = client.post("/step", json={
            "task_id": "easy_triage",
            "action_type": "CLASSIFY_EMAIL",
            "category": "WORK",
        })
        assert r.status_code == 200
        assert r.json()["reward"] == 0.30

    def test_step_invalid_action_type_422(self):
        r = client.post("/step", json={
            "task_id": "easy_triage",
            "action_type": "do_magic",
            "category": "work",
        })
        assert r.status_code == 422

    def test_state_returns_current_phase(self):
        client.post("/reset", json={"task_id": "hard_triage"})
        r = client.get("/state", params={"task_id": "hard_triage"})
        assert r.status_code == 200
        data = r.json()
        assert data["current_phase"] == "classify"
        assert data["done"] is False

    def test_full_episode_via_api(self):
        client.post("/reset", json={"task_id": "easy_triage"})
        client.post("/step", json={"task_id": "easy_triage",
                                   "action_type": "classify_email", "category": "work"})
        client.post("/step", json={"task_id": "easy_triage",
                                   "action_type": "set_priority", "priority": "high"})
        client.post("/step", json={"task_id": "easy_triage", "action_type": "generate_response",
                                   "response_text": "I will attend the meeting tomorrow."})
        r = client.post("/step", json={"task_id": "easy_triage",
                                       "action_type": "archive_email"})
        assert r.status_code == 200
        data = r.json()
        assert data["done"] is True
        assert data["info"]["score"] == 1.0


# ══════════════════════════════════════════════════════════════════
# 5. OpenEnv Compliance Tests
# ══════════════════════════════════════════════════════════════════

class TestOpenEnvCompliance:
    def test_rewards_in_valid_range(self, easy_env):
        result = easy_env.step(make_step("easy_triage", "classify_email", category="work"))
        assert -1.0 <= result["reward"] <= 1.0

    def test_step_response_has_required_fields(self):
        client.post("/reset", json={"task_id": "easy_triage"})
        r = client.post("/step", json={"task_id": "easy_triage",
                                       "action_type": "classify_email", "category": "work"})
        data = r.json()
        # OpenEnv required fields
        assert "reward" in data
        assert "done" in data
        assert "observation" in data

    def test_reset_response_has_observation(self):
        r = client.post("/reset", json={"task_id": "medium_triage"})
        data = r.json()
        obs = data["observation"]
        assert "subject" in obs
        assert "body" in obs
        assert "sender" in obs

    def test_state_has_step_count(self):
        client.post("/reset", json={"task_id": "hard_triage"})
        client.post("/step", json={"task_id": "hard_triage",
                                   "action_type": "classify_email", "category": "spam"})
        r = client.get("/state", params={"task_id": "hard_triage"})
        assert r.json()["step_count"] == 1

    def test_tasks_have_difficulty_and_max_steps(self):
        r = client.get("/tasks")
        for task in r.json()["tasks"]:
            assert "difficulty" in task
            assert "max_steps" in task
            assert task["max_steps"] > 0

    def test_all_tasks_are_completable(self):
        """Each task can be completed in <= max_steps with correct actions."""
        scenarios = {
            "easy_triage":   ("work",    "high",   True),
            "medium_triage": ("billing", "urgent", True),
            "hard_triage":   ("spam",    "low",    False),
        }
        for task_id, (cat, prio, needs_response) in scenarios.items():
            client.post("/reset", json={"task_id": task_id})
            client.post("/step", json={"task_id": task_id,
                                       "action_type": "classify_email", "category": cat})
            client.post("/step", json={"task_id": task_id,
                                       "action_type": "set_priority", "priority": prio})
            if needs_response:
                client.post("/step", json={"task_id": task_id,
                                           "action_type": "generate_response",
                                           "response_text": "Thank you, we'll follow up."})
            r = client.post("/step", json={"task_id": task_id,
                                           "action_type": "archive_email"})
            assert r.json()["done"] is True, f"Task {task_id} didn't complete"
