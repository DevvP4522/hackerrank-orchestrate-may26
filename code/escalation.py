from __future__ import annotations

import re
from typing import Any, Dict, List


_HIGH_RISK_KEYWORDS = ("fraud", "unauthorized", "stolen", "hacked", "scam")
_ACCOUNT_ACCESS_KEYWORDS = ("cannot access account", "locked account", "account hacked")
_PAYMENT_DISPUTE_KEYWORDS = ("charged twice", "wrong charge", "refund not received")

# If more than one of the high-risk keywords appears, we escalate.
_MIN_HIGH_RISK_KEYWORDS_FOR_MULTI_RISK = 2

_LOW_RETRIEVAL_SCORE_THRESHOLD = 0.5
_MIN_WORDS_FOR_CLEAR_INPUT = 5


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text.lower()))


def _extract_top_score(retrieved_docs: List[Any]) -> float | None:
    """
    retrieved_docs is expected to be a list of dicts from retriever.py:
    { ..., "score": float }
    """
    if not retrieved_docs:
        return None

    first = retrieved_docs[0]
    if isinstance(first, dict):
        score = first.get("score")
        try:
            if score is None:
                return None
            return float(score)
        except (TypeError, ValueError):
            return None

    return None


def should_escalate(issue: str, retrieved_docs: List[Any]) -> bool:
    """
    Decide whether to escalate to a human.
    Deterministic rules only. No external calls.
    """
    text = _normalize_text(issue)

    # 5. UNCLAR / INVALID INPUT
    if not text:
        return True
    if _word_count(text) < _MIN_WORDS_FOR_CLEAR_INPUT:
        return True

    # 1. HIGH-RISK KEYWORDS
    high_risk_hits = sum(1 for kw in _HIGH_RISK_KEYWORDS if kw in text)
    if high_risk_hits >= 1:
        return True

    # 6. MULTI-RISK SIGNAL (trigger if multiple risk keywords appear)
    # Note: This will be redundant with rule #1 for single keyword hits.
    # Keeping both as requested: escalate on multiple risk keywords.
    if high_risk_hits >= _MIN_HIGH_RISK_KEYWORDS_FOR_MULTI_RISK:
        return True

    # 2. ACCOUNT ACCESS ISSUES (SENSITIVE)
    for kw in _ACCOUNT_ACCESS_KEYWORDS:
        if kw in text:
            return True

    # 3. PAYMENT DISPUTES
    for kw in _PAYMENT_DISPUTE_KEYWORDS:
        if kw in text:
            return True

    # 4. LOW RETRIEVAL CONFIDENCE
    if not retrieved_docs:
        return True

    top_score = _extract_top_score(retrieved_docs)
    if top_score is None:
        return True
    if top_score < _LOW_RETRIEVAL_SCORE_THRESHOLD:
        return True

    return False
