from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure local `code/` package is importable when running this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from code.classifier import classify_ticket
from code.responder import FALLBACK_MESSAGE, generate_response


class SupportAgent:
    # Keyword-based high-risk detection for escalation.
    _FRAUD_SECURITY_IDENTITY_KEYWORDS: Tuple[str, ...] = (
        "fraud",
        "unauthorized",
        "stolen",
        "hacked",
        "scam",
        "identity theft",
        "identity stolen",
    )

    _UNAUTHORIZED_PERMISSION_KEYWORDS: Tuple[str, ...] = (
        "permission",
        "permissions",
        "access denied",
        "not authorized",
        "unauthorized access",
    )

    _PAYMENT_DISPUTE_KEYWORDS: Tuple[str, ...] = (
        "charged twice",
        "wrong charge",
        "refund not received",
    )

    # Requirement-aligned word count logic:
    # word_count = len(issue.strip().split())
    @staticmethod
    def _word_count_issue(issue: str) -> int:
        return len((issue or "").strip().split())

    def __init__(self):
        # Retrievers rely on optional heavy deps (e.g., faiss, sentence-transformers).
        self.retriever = None
        try:
            from code.retriever import Retriever

            self.retriever = Retriever()
        except Exception:
            self.retriever = None

    @staticmethod
    def _first_keyword_hit(text: str, keywords: Tuple[str, ...]) -> Optional[str]:
        normalized = (text or "").strip().lower()
        for kw in keywords:
            if kw in normalized:
                return kw
        return None

    @staticmethod
    def _top_score(docs: List[Dict[str, Any]]) -> Optional[float]:
        if not docs:
            return None
        best: Optional[float] = None
        for d in docs:
            if not isinstance(d, dict):
                continue
            score = d.get("score", 0.0)
            try:
                score_f = float(score)
            except Exception:
                continue
            if best is None or score_f > best:
                best = score_f
        return best

    def process_ticket(self, issue: str, subject: str, company: str) -> dict:
        # 1. Combine input:
        combined_text = ((issue or "") + " " + (subject or "")).strip()
        combined_norm = combined_text.lower()

        # Word count MUST use issue only.
        issue_word_count = self._word_count_issue(issue)
        issue_is_empty = (issue or "").strip() == ""

        # 2. Classification:
        request_type: str
        product_area: str
        try:
            request_type, product_area = classify_ticket(issue, subject)
            if not isinstance(request_type, str) or not isinstance(product_area, str):
                raise ValueError("Invalid classification output types")
        except Exception:
            request_type, product_area = "invalid", "general"

        # 3. Retrieval (used only for response generation; NOT for escalation):
        docs: List[Dict[str, Any]] = []
        if self.retriever is not None:
            docs = self.retriever.query(combined_text, top_k=5)

        # 4. Escalation decision rule:
        # status = "escalated" only when high_risk == True; else "replied".
        high_risk = False
        justification = ""

        fraud_hit = self._first_keyword_hit(combined_norm, self._FRAUD_SECURITY_IDENTITY_KEYWORDS)
        perm_hit = self._first_keyword_hit(combined_norm, self._UNAUTHORIZED_PERMISSION_KEYWORDS)
        payment_hit = self._first_keyword_hit(combined_norm, self._PAYMENT_DISPUTE_KEYWORDS)

        # Fraud/security/identity theft
        if fraud_hit is not None:
            high_risk = True
            justification = f"Detected high-risk keyword ({fraud_hit}) → escalated"

        # Unauthorized access / permissions
        elif perm_hit is not None:
            high_risk = True
            justification = f"Detected unauthorized/permission keyword ({perm_hit}) → escalated"

        # Payment disputes
        elif payment_hit is not None:
            high_risk = True
            justification = f"Detected payment dispute keyword ({payment_hit}) → escalated"

        # request_type == invalid AND word_count < 3
        elif request_type == "invalid" and issue_word_count < 3:
            high_risk = True
            if issue_is_empty:
                justification = "Input issue is empty → escalated"
            else:
                justification = f"Input too short (word_count={issue_word_count}) and classified as invalid → escalated"

        status = "escalated" if high_risk else "replied"

        if status == "escalated":
            return {
                "status": "escalated",
                "product_area": product_area,
                "response": "This issue requires further review by our support team.",
                "justification": justification or "High-risk conditions met → escalated",
                "request_type": request_type,
            }

        # 5. Response generation:
        response = generate_response(combined_text, docs)

        # 6. Final output (must NOT escalate just because fallback happened):
        fallback = response == FALLBACK_MESSAGE
        return {
            "status": "replied",
            "product_area": product_area,
            "response": response,
            "justification": (
                "Strong semantic match with support documentation → response generated"
                if not fallback
                else "No grounded answer extracted from retrieved documentation → using fallback message"
            ),
            "request_type": request_type,
        }


if __name__ == "__main__":
    agent = SupportAgent()

    test_ticket = {
        "issue": "I see an unauthorized transaction on my card",
        "subject": "",
        "company": "Visa",
    }

    result = agent.process_ticket(test_ticket["issue"], test_ticket["subject"], test_ticket["company"])
    print(result)
