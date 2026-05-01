from __future__ import annotations

import re
from typing import Tuple


_REQUEST_TYPES = ("product_issue", "feature_request", "bug", "invalid")
_PRODUCT_AREAS = ("account", "billing", "fraud", "api", "assessment", "login", "payment", "security", "general")


_REQUEST_TYPE_KEYWORDS_PRODUCT_ISSUE = (
    "error",
    "not working",
    "failed",
    "issue",
)

_REQUEST_TYPE_KEYWORDS_BUG = (
    "bug",
    "crash",
    "unexpected",
)

_REQUEST_TYPE_KEYWORDS_FEATURE = (
    "feature",
    "add",
    "request",
    "suggest",
)

_INVALID_MIN_WORDS = 3


_PRODUCT_AREA_KEYWORDS = {
    "account": ("login", "password", "signin"),
    "billing": ("payment", "charged", "billing", "refund"),
    "fraud": ("fraud", "unauthorized", "suspicious"),
    "api": ("api", "endpoint", "integration"),
    "assessment": ("assessment", "test", "submission"),
}


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _word_count(text: str) -> int:
    # Keep it simple/deterministic: split on word boundaries.
    words = re.findall(r"\b\w+\b", text.lower())
    return len(words)


def _contains_any(haystack: str, needles: Tuple[str, ...]) -> bool:
    return any(n in haystack for n in needles)


def _fallback_by_overlap(text: str) -> str:
    """
    Lightweight NLP fallback: choose class by keyword overlap.
    Deterministic and uses only local heuristics.
    """
    # Token overlap for robustness (still keyword-driven).
    tokens = set(re.findall(r"\b\w+\b", text))
    # Also include bigrams/trigrams already handled in substring matching elsewhere; here just token overlap.
    def score_for(keywords: Tuple[str, ...]) -> int:
        score = 0
        for kw in keywords:
            # If keyword is multi-word, fall back to substring check since overlap won't catch it well.
            if " " in kw:
                score += 1 if kw in text else 0
                continue
            score += 1 if kw in tokens else 0
        return score

    product_issue_score = score_for(_REQUEST_TYPE_KEYWORDS_PRODUCT_ISSUE)
    bug_score = score_for(_REQUEST_TYPE_KEYWORDS_BUG)
    feature_score = score_for(_REQUEST_TYPE_KEYWORDS_FEATURE)

    best = max(
        [
            ("product_issue", product_issue_score),
            ("bug", bug_score),
            ("feature_request", feature_score),
        ],
        key=lambda x: x[1],
    )
    if best[1] <= 0:
        return "invalid"
    return best[0]


def _classify_request_type(text: str) -> str:
    # Primary rule-based classification.
    # If multiple matches occur, we use fixed precedence for determinism: bug > product_issue > feature_request > invalid.
    if _contains_any(text, _REQUEST_TYPE_KEYWORDS_BUG):
        return "bug"
    if _contains_any(text, _REQUEST_TYPE_KEYWORDS_PRODUCT_ISSUE):
        return "product_issue"
    if _contains_any(text, _REQUEST_TYPE_KEYWORDS_FEATURE):
        return "feature_request"

    # invalid: empty, nonsense, or unrelated
    if _word_count(text) < _INVALID_MIN_WORDS:
        return "invalid"

    # Lightweight NLP fallback: keyword overlap scoring.
    return _fallback_by_overlap(text)


def _classify_product_area(text: str) -> str:
    # Primary rule-based classification per requirement.
    for area, keywords in _PRODUCT_AREA_KEYWORDS.items():
        if _contains_any(text, keywords):
            # Note: "login" and "payment" and "security" exist in the category list but are not part of the provided
            # rules. We strictly follow the given ruleset and thus do not map those areas unless keywords match.
            return area
    return "general"


def classify_ticket(issue: str, subject: str) -> Tuple[str, str]:
    """
    Returns: (request_type, product_area)
    """
    combined = f"{subject or ''} {issue or ''}"
    text = _normalize_text(combined)

    request_type = _classify_request_type(text)
    product_area = _classify_product_area(text)

    if request_type not in _REQUEST_TYPES:
        request_type = "invalid"
    if product_area not in _PRODUCT_AREAS:
        product_area = "general"

    return request_type, product_area
