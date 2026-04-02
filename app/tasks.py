"""
Task definitions for Email Triage OpenEnv.

Each task contains:
  - An email to triage
  - Ground truth labels (for grading)
  - Difficulty level
  - Max steps allowed
"""

from typing import Any

TASKS: dict[str, dict[str, Any]] = {

    # ── EASY ───────────────────────────────────────────────────────────────────
    "easy_triage": {
        "name": "Easy Email Triage",
        "difficulty": "easy",
        "description": (
            "Triage a straightforward work email. "
            "The category and priority are unambiguous."
        ),
        "max_steps": 8,
        "correct_category": "work",
        "correct_priority": "high",
        "response_required": True,
        "email": {
            "subject": "Urgent: Q4 Budget Review Meeting Tomorrow",
            "sender": "manager@company.com",
            "body": (
                "Hi Team,\n\n"
                "Please be advised that the Q4 budget review meeting has been moved to tomorrow "
                "at 10:00 AM in Conference Room B. Attendance is mandatory for all senior staff.\n\n"
                "Please review the attached budget spreadsheet before the meeting and come prepared "
                "with your department's expense projections.\n\n"
                "Best regards,\n"
                "Sarah Johnson\nDirector of Finance"
            ),
            "received_at": "2024-01-15T09:00:00Z",
            "has_attachment": True,
            "metadata": {
                "sender_domain": "company.com",
                "is_internal": True,
                "thread_count": 1,
            },
        },
        "grader": {
            "type": "exact_match",
            "weights": {
                "classification": 0.35,
                "priority": 0.30,
                "response": 0.15,
                "completion": 0.20,
            },
        },
    },

    # ── MEDIUM ─────────────────────────────────────────────────────────────────
    "medium_triage": {
        "name": "Medium Email Triage",
        "difficulty": "medium",
        "description": (
            "Triage a customer support email with billing concerns. "
            "Requires correctly identifying support vs billing category."
        ),
        "max_steps": 10,
        "correct_category": "billing",
        "correct_priority": "urgent",
        "response_required": True,
        "email": {
            "subject": "Re: Invoice #INV-2024-8821 — Incorrect Charge",
            "sender": "frustrated.customer@gmail.com",
            "body": (
                "Hello Support,\n\n"
                "I am writing regarding invoice #INV-2024-8821 dated January 10th. "
                "I was charged $499.99 for the Enterprise Plan, but I am subscribed to the "
                "Basic Plan at $49.99/month.\n\n"
                "This is the SECOND time this has happened. Last month I had to spend 2 hours "
                "on the phone to get a refund. I need this resolved TODAY or I will be disputing "
                "the charge with my bank and cancelling my subscription.\n\n"
                "Order reference: ORD-88217\n"
                "Account email: frustrated.customer@gmail.com\n\n"
                "This is unacceptable.\n"
                "— Michael Torres"
            ),
            "received_at": "2024-01-15T14:32:00Z",
            "has_attachment": False,
            "metadata": {
                "sender_domain": "gmail.com",
                "is_internal": False,
                "thread_count": 3,
                "customer_tier": "basic",
                "previous_tickets": 2,
            },
        },
        "grader": {
            "type": "exact_match",
            "weights": {
                "classification": 0.30,
                "priority": 0.35,
                "response": 0.20,
                "completion": 0.15,
            },
        },
    },

    # ── HARD ───────────────────────────────────────────────────────────────────
    "hard_triage": {
        "name": "Hard Email Triage",
        "difficulty": "hard",
        "description": (
            "Triage a deceptive phishing/spam email disguised as a newsletter. "
            "Agent must correctly identify spam and NOT generate a response."
        ),
        "max_steps": 12,
        "correct_category": "spam",
        "correct_priority": "low",
        "response_required": False,  # Responding to spam is penalized
        "email": {
            "subject": "🎉 Congratulations! You've been selected — Claim your $500 Amazon Gift Card",
            "sender": "rewards@amazon-gifts-promo.net",
            "body": (
                "Dear Valued Customer,\n\n"
                "You have been RANDOMLY SELECTED from millions of Amazon shoppers to receive "
                "a $500 Amazon Gift Card as part of our annual customer appreciation program!\n\n"
                "To claim your reward, simply:\n"
                "1. Click here: http://bit.ly/amz-gift-claim-now\n"
                "2. Enter your Amazon login credentials\n"
                "3. Verify your shipping address\n"
                "4. Your gift card will arrive within 24 hours!\n\n"
                "⚠️ This offer expires in 24 HOURS. Act now!\n\n"
                "This promotion is sponsored by Amazon Rewards Division.\n"
                "To unsubscribe: http://bit.ly/unsub-amz-rewards"
            ),
            "received_at": "2024-01-15T07:15:00Z",
            "has_attachment": False,
            "metadata": {
                "sender_domain": "amazon-gifts-promo.net",
                "is_internal": False,
                "spf_pass": False,
                "dkim_pass": False,
                "spam_score": 9.2,
                "contains_suspicious_links": True,
            },
        },
        "grader": {
            "type": "exact_match",
            "weights": {
                "classification": 0.40,  # Extra weight — spam detection is critical
                "priority": 0.25,
                "response": 0.15,        # Penalize if response generated for spam
                "completion": 0.20,
            },
        },
    },
}
