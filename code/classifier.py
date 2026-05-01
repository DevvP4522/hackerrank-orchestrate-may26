from __future__ import annotations

from typing import Tuple

_REQUEST_TYPES: Tuple[str, ...] = ("product_issue", "feature_request", "bug", "invalid")
_PRODUCT_AREAS: Tuple[str, ...] = (
    "account",
    "billing",
    "fraud",
    "api",
    "assessment",
    "login",
    "payment",
    "security",
    "general",
)

# Keyword-based request type classification (deterministic substring matching).
_BUG_KEYWORDS: Tuple[str, ...] = (
    "not working",
    "error",
    "failed",
    "down",
)

_PRODUCT_ISSUE_KEYWORDS: Tuple[str, ...] = (
    "can't",
    "cannot",
    "unable",
    "issue",
    "problem",
)

_FEATURE_REQUEST_KEYWORDS: Tuple[str, ...] = (
    "add",
    "request",
    "can you",
    "please allow",
)

_INVALID_WORD_COUNT_MIN = 3

_PRODUCT_AREA_KEYWORDS = {
    "account": ("login", "password", "signin"),
    "billing": ("payment", "charged", "billing", "refund"),
    "fraud": ("fraud", "unauthorized", "suspicious"),
    "api": ("api", "endpoint", "integration"),
    "assessment": ("assessment", "test", "submission"),
}


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _word_count(issue: str) -> int:
    # Requirement: word_count = len(issue.strip().split())
    return len((issue or "").strip().split())


def _contains_any(haystack: str, needles: Tuple[str, ...]) -> bool:
    return any(n in haystack for n in needles)


def _classify_request_type(combined_text: str) -> str:
    """
    Keyword-based triage.
    Assumes the caller already applied the invalid rules based on emptiness/word_count.
    """
    text = _normalize_text(combined_text)

    # Precedence: bug > product_issue > feature_request
    # (Deterministic and reduces invalid classification rate.)
    if _contains_any(text, _BUG_KEYWORDS):
        return "bug"
    if _contains_any(text, _PRODUCT_ISSUE_KEYWORDS):
        return "product_issue"
    if _contains_any(text, _FEATURE_REQUEST_KEYWORDS):
        return "feature_request"

    # If the input is meaningful but no keyword matched, default to product_issue.
    # This prevents over-aggressive invalid classification.
    return "product_issue"


def _classify_product_area(text: str) -> str:
    normalized = _normalize_text(text)
    for area, keywords in _PRODUCT_AREA_KEYWORDS.items():
        if _contains_any(normalized, keywords):
            return area
    return "general"


def classify_ticket(issue: str, subject: str) -> Tuple[str, str]:
    """
    Returns: (request_type, product_area)
    """
    issue_norm = _normalize_text(issue)
    combined = f"{subject or ''} {issue or ''}"
    combined_norm = _normalize_text(combined)

    # Invalid should only be used when:
    # - issue is empty
    # - issue is <3 words (short/unclear meaningfulness)
    wc = _word_count(issue)
    if not issue_norm:
        request_type = "invalid"
    elif wc < _INVALID_WORD_COUNT_MIN:
        request_type = "invalid"
    else:
        request_type = _classify_request_type(combined_norm)

    product_area = _classify_product_area(combined_norm)

    if request_type not in _REQUEST_TYPES:
        request_type = "invalid"
    if product_area not in _PRODUCT_AREAS:
        product_area = "general"

    return request_type, product_area
