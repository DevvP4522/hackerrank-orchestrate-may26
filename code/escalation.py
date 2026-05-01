from __future__ import annotations

import re
from typing import Any, List

# High-risk / human intervention cases only.
# (Keep keyword-based + deterministic.)
_HIGH_RISK_KEYWORDS = (
    "fraud",
    "unauthorized",
    "stolen",
    "hacked",
    "scam",
    "identity theft",
    "identity stolen",
    "identity",
)

_UNAUTHORIZED_ACCESS_KEYWORDS = (
    "permission",
    "permissions",
    "access denied",
    "not authorized",
    "unauthorized access",
)

_ACCOUNT_ACCESS_KEYWORDS = (
    "cannot access account",
    "locked account",
    "account hacked",
)

_PAYMENT_DISPUTE_KEYWORDS = (
    "charged twice",
    "wrong charge",
    "refund not received",
)

# Escalate for invalid short inputs only (per requirements).
_MIN_WORDS_FOR_MEANINGFUL_INPUT = 3


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _word_count(text: str) -> int:
    # Requirement-aligned word count style.
    return len((text or "").strip().split())


def should_escalate(issue: str, retrieved_docs: List[Any]) -> bool:
    """
    Deterministic escalation rules only.

    Escalate ONLY when:
    - fraud / security / identity theft
    - payment disputes requiring human intervention
    - unauthorized access / permissions
    - issue is invalid AND word_count < 3 (approximated as: empty or <3 words)
    """
    text = _normalize_text(issue)

    # invalid-only short/empty: matches classifier's invalid rule when word_count < 3.
    if not text:
        return True
    if _word_count(text) < _MIN_WORDS_FOR_MEANINGFUL_INPUT:
        return True

    # fraud/security/identity theft
    if any(kw in text for kw in _HIGH_RISK_KEYWORDS):
        return True

    # unauthorized access / permissions
    if any(kw in text for kw in _UNAUTHORIZED_ACCESS_KEYWORDS):
        return True

    # account access sensitivity
    if any(kw in text for kw in _ACCOUNT_ACCESS_KEYWORDS):
        return True

    # payment disputes
    if any(kw in text for kw in _PAYMENT_DISPUTE_KEYWORDS):
        return True

    # Retrieval-confidence / empty docs must NOT drive escalation anymore.
    # (Status decision must rely on high_risk / invalid-short only.)
    return False
