from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

FALLBACK_MESSAGE = (
    "I'm unable to confidently answer this based on available information. "
    "Please contact support for further assistance."
)

CLOSING_LINE = "If the issue persists, please contact support."

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _tokenize_issue(issue: str) -> List[str]:
    """
    Extract stable keyword tokens from the issue to drive extractive selection.
    """
    normalized = _normalize_text(issue)
    tokens = re.findall(r"\b[a-z0-9_]{2,}\b", normalized)
    seen = set()
    out: List[str] = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


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
    "sign-in",
    "reset",
    "password",
    "forgot",
    "update",
    "configure",
    "install",
    "enable",
    "disable",
    "verify",
    "confirm",
    "change",
)


def _sentence_action_score(sentence: str) -> int:
    s = _normalize_text(sentence)
    score = 0
    for kw in _ACTION_KEYWORDS:
        if kw in s:
            score += 1
    return score


def _sentence_keyword_overlap(sentence: str, tokens: Sequence[str]) -> int:
    s = _normalize_text(sentence)
    if not tokens or not s:
        return 0
    overlap = 0
    for tok in tokens:
        if tok in s:
            overlap += 1
    return overlap


def _split_sentences(text: str) -> List[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT_RE.split(cleaned)
    return [p.strip() for p in parts if p and p.strip()]


def _clean_sentence(sentence: str) -> str:
    s = (sentence or "").strip()
    # Ensure proper punctuation (deterministic and safe).
    if s and s[-1] not in ".!?":
        s += "."
    return s


def _dedupe_sentences_by_norm(sentences: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in sentences:
        ns = _normalize_text(s)
        if not ns or ns in seen:
            continue
        seen.add(ns)
        out.append(s)
    return out


def _extract_candidate_sentences(issue: str, top_docs: List[Dict]) -> List[Tuple[float, str]]:
    """
    Deterministically returns sentence candidates as (sentence_score, sentence),
    prioritizing higher scoring docs and instructional/action sentences.
    """
    issue_tokens = _tokenize_issue(issue)
    candidates: List[Tuple[float, str]] = []

    for doc in top_docs:
        if not isinstance(doc, dict):
            continue

        doc_text = doc.get("text", "")
        if not isinstance(doc_text, str) or not doc_text.strip():
            continue

        try:
            doc_score = float(doc.get("score", 0.0))
        except Exception:
            doc_score = 0.0

        for sentence in _split_sentences(doc_text):
            s = sentence.strip()
            if not s:
                continue

            # Keep relevance filtering, but don't be overly strict:
            # accept sentence if it has action cues OR any issue keyword overlap.
            overlap = _sentence_keyword_overlap(s, issue_tokens)
            action = _sentence_action_score(s)
            if overlap <= 0 and action <= 0:
                continue

            # Prefer action + keyword overlap, and slightly weight doc relevance.
            sentence_score = (overlap * 2.0) + action + (doc_score * 0.15)
            candidates.append((sentence_score, _clean_sentence(s)))

    # Determinism: tie-break by sentence text
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates


def generate_response(issue: str, retrieved_docs: list) -> str:
    """
    Generate a grounded, deterministic structured response using ONLY retrieved_docs content.

    If candidate sentences exist:
      - select top 2–3 sentences based on score
      - DO NOT reject them unless completely irrelevant

    Fallback ONLY when:
      - no documents retrieved
      - no candidate sentences extracted
    """
    if not retrieved_docs or not isinstance(retrieved_docs, list):
        return FALLBACK_MESSAGE

    valid_docs: List[Dict] = []
    for d in retrieved_docs:
        if not isinstance(d, dict):
            continue
        text = d.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        valid_docs.append(d)

    if not valid_docs:
        return FALLBACK_MESSAGE

    issue_text = (issue or "").strip()
    # Do not over-filter based on issue length; only require that docs/candidates exist.

    def _doc_sort_key(d: Dict) -> Tuple[float, str]:
        try:
            score = float(d.get("score", 0.0))
        except Exception:
            score = 0.0
        text = d.get("text", "")
        text_str = text if isinstance(text, str) else ""
        return (score, text_str)

    valid_docs.sort(key=lambda d: (_doc_sort_key(d)[0], _doc_sort_key(d)[1]), reverse=True)
    top_docs = valid_docs[:3]

    candidates = _extract_candidate_sentences(issue=issue_text, top_docs=top_docs)
    if not candidates:
        return FALLBACK_MESSAGE

    top_sentences = [s for _, s in candidates[:3]]
    top_sentences = _dedupe_sentences_by_norm(top_sentences)
    if not top_sentences:
        return FALLBACK_MESSAGE

    # Use top 2–3 sentences; prefer 3 if available.
    use_n = 3 if len(top_sentences) >= 3 else len(top_sentences)
    selected = top_sentences[:use_n]

    lines: List[str] = []
    lines.append("To resolve your issue, please try the following steps:")
    lines.append("")

    for idx, s in enumerate(selected, start=1):
        lines.append(f"{idx}. {s}")

    lines.append("")
    lines.append(CLOSING_LINE)

    return "\n".join(lines).strip()
