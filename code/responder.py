from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

FALLBACK_MESSAGE = (
    "I'm sorry, but I don't have enough information to answer this request safely. "
    "Please contact support for further assistance."
)


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _tokenize_issue(issue: str) -> List[str]:
    """
    Extract stable keyword tokens from the issue to drive extractive selection.
    """
    normalized = _normalize_text(issue)
    # Keep alphanumerics/underscore, drop very short tokens.
    tokens = re.findall(r"\b[a-z0-9_]{2,}\b", normalized)
    # Deduplicate while preserving order.
    seen = set()
    out: List[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    # Also handle texts without punctuation by keeping as single "sentence".
    return [p.strip() for p in parts if p and p.strip()]


_ACTION_KEYWORDS = (
    "go to",
    "click",
    "select",
    "choose",
    "check",
    "ensure",
    "follow",
    "enter",
    "type",
    "submit",
    "log in",
    "login",
    "sign in",
    "reset",
    "password",
    "forgot",
    "update",
    "configure",
    "install",
    "enable",
    "disable",
    "verify",
)


def _sentence_keyword_overlap(sentence: str, tokens: Sequence[str]) -> int:
    s = _normalize_text(sentence)
    if not tokens or not s:
        return 0
    overlap = 0
    for tok in tokens:
        if tok in s:
            overlap += 1
    return overlap


def _sentence_action_score(sentence: str) -> int:
    s = _normalize_text(sentence)
    score = 0
    for kw in _ACTION_KEYWORDS:
        if kw in s:
            score += 1
    return score


def _dedupe_sentences(sentences: List[str]) -> List[str]:
    """
    Dedupe by normalized form to avoid repeated lines across docs.
    """
    seen = set()
    out: List[str] = []
    for s in sentences:
        ns = _normalize_text(s)
        if ns in seen:
            continue
        seen.add(ns)
        out.append(s)
    return out


def _is_relevant_sentence(sentence: str, issue_tokens: Sequence[str]) -> bool:
    s = _normalize_text(sentence)
    if not s:
        return False
    if _sentence_keyword_overlap(sentence, issue_tokens) > 0:
        return True
    # If no keyword overlap, still allow strongly instructional/action sentences.
    return _sentence_action_score(sentence) > 0


def _safe_truncate_words(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", (text or "").strip())
    if len(words) <= max_words:
        return (text or "").strip()
    return " ".join(words[:max_words]).strip()


def _extract_support_steps(issue: str, retrieved_docs: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Returns (steps_sentences, next_step_sentences).
    Both are drawn extractively from retrieved_docs.
    """
    issue_tokens = _tokenize_issue(issue)

    candidates: List[Tuple[float, str]] = []
    next_candidates: List[Tuple[float, str]] = []

    for doc in retrieved_docs:
        doc_text = ""
        doc_score = 0.0

        if isinstance(doc, dict):
            doc_text = doc.get("text", "") or ""
            try:
                doc_score = float(doc.get("score", 0.0))
            except Exception:
                doc_score = 0.0

        sentences = _split_sentences(doc_text)
        if not sentences:
            continue

        for sentence in sentences:
            s = sentence.strip()
            if not s:
                continue

            overlap = _sentence_keyword_overlap(s, issue_tokens)
            action = _sentence_action_score(s)

            # Prefer sentences that are instructional; also weight by doc relevance.
            # Keep it deterministic: no randomness.
            sentence_score = (overlap * 2.0) + action + (doc_score * 0.10)

            if _is_relevant_sentence(s, issue_tokens):
                candidates.append((sentence_score, s))

            # Next step: look for explicit contact/support style guidance.
            ns = _normalize_text(s)
            if any(x in ns for x in ("contact support", "contact", "support for", "reach out")):
                next_candidates.append((sentence_score, s))

    # Sort highest score first.
    candidates.sort(key=lambda x: x[0], reverse=True)
    next_candidates.sort(key=lambda x: x[0], reverse=True)

    steps = [s for _, s in candidates]
    steps = _dedupe_sentences(steps)

    # Keep steps concise.
    steps = steps[:5]

    next_step = [s for _, s in next_candidates]
    next_step = _dedupe_sentences(next_step)
    next_step = next_step[:2]

    return steps, next_step


def generate_response(issue: str, retrieved_docs: list) -> str:
    """
    Generate a safe, grounded, deterministic response using ONLY retrieved_docs content.

    retrieved_docs items are expected:
    {
      "text": "...",
      "source": "...",
      "product_area": "...",
      "score": float
    }
    """
    if not retrieved_docs:
        return FALLBACK_MESSAGE

    # Validate retrieved docs shape minimally.
    valid_docs: List[Dict] = []
    for d in retrieved_docs:
        if (
            isinstance(d, dict)
            and isinstance(d.get("text", ""), str)
            and d.get("text", "").strip()
        ):
            valid_docs.append(d)

    if not valid_docs:
        return FALLBACK_MESSAGE

    # Select top documents by score.
    def _doc_score(d: Dict) -> float:
        try:
            return float(d.get("score", 0.0))
        except Exception:
            return 0.0

    valid_docs.sort(key=_doc_score, reverse=True)
    top_docs = valid_docs[:3]

    steps, next_steps = _extract_support_steps(issue=issue or "", retrieved_docs=top_docs)

    # Safety checks: ensure response is not empty or too generic.
    if not steps:
        return FALLBACK_MESSAGE

    # Construct response; keep it within ~150-250 words.
    response_lines: List[str] = []
    response_lines.append("I understand you’re trying to resolve this issue.")
    response_lines.append("")
    response_lines.append("To resolve your issue, please try the following steps:")

    for idx, step in enumerate(steps, start=1):
        response_lines.append(f"{idx}. {_safe_truncate_words(step, 35)}")

    if next_steps:
        response_lines.append("")
        response_lines.append(_safe_truncate_words(next_steps[0], 40))

    response = "\n".join(response_lines).strip()

    # Final safety check: heuristic guard against being too generic.
    normalized = _normalize_text(response)
    if not any(
        kw in normalized
        for kw in (
            "go to",
            "click",
            "select",
            "check",
            "ensure",
            "follow",
            "enter",
            "submit",
            "reset",
            "forgot",
            "password",
        )
    ):
        return FALLBACK_MESSAGE

    # Word limit (soft).
    words = re.findall(r"\S+", response)
    if len(words) > 260:
        trimmed_lines: List[str] = []
        trimmed_lines.append("I understand you’re trying to resolve this issue.")
        trimmed_lines.append("")
        trimmed_lines.append("To resolve your issue, please try the following steps:")
        for idx, step in enumerate(steps[:3], start=1):
            trimmed_lines.append(f"{idx}. {_safe_truncate_words(step, 35)}")
        response = "\n".join(trimmed_lines).strip()

    return response
