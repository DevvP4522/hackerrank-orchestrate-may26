from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure local `code/` package is importable when running this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from code.classifier import classify_ticket
from code.escalation import should_escalate
from code.responder import generate_response


class SupportAgent:
    def __init__(self):
        # Retrievers rely on optional heavy deps (e.g., faiss, sentence-transformers).
        # If unavailable in the runtime, we must safely escalate (and never generate a reply).
        self.retriever = None
        try:
            from code.retriever import Retriever

            self.retriever = Retriever()
        except Exception:
            self.retriever = None

    def process_ticket(self, issue: str, subject: str, company: str) -> dict:
        # 1. Combine input:
        text = (issue or "") + " " + (subject or "")

        # 2. Classification:
        request_type: str
        product_area: str
        try:
            request_type, product_area = classify_ticket(issue, subject)
            if not isinstance(request_type, str) or not isinstance(product_area, str):
                raise ValueError("Invalid classification output types")
        except Exception:
            # Edge case: classification returns invalid → escalate
            request_type, product_area = "invalid", "general"

        # 3. Retrieval:
        docs: List[Dict[str, Any]] = []
        if self.retriever is not None:
            docs = self.retriever.query(text, top_k=5)

        # 4. Escalation Decision:
        # ALWAYS check escalation BEFORE generating response
        try:
            escalate_by_risk = should_escalate(text, docs)
        except Exception:
            escalate_by_risk = True

        issue_is_empty = (issue or "").strip() == ""
        classification_is_invalid = request_type == "invalid"
        docs_are_empty = not docs

        if issue_is_empty or classification_is_invalid or docs_are_empty or escalate_by_risk:
            if issue_is_empty:
                justification = "Empty issue provided → escalated for safety."
            elif classification_is_invalid:
                justification = "Classification returned invalid request type → escalated."
            elif docs_are_empty:
                justification = "No retrieval matches found → escalated."
            else:
                justification = "Detected fraud/high-risk signals or low retrieval confidence → escalated."

            return {
                "status": "escalated",
                "product_area": product_area,
                "response": "This issue requires further review by our support team.",
                "justification": justification,
                "request_type": request_type,
            }

        # 5. Response Generation:
        response = generate_response(text, docs)

        # 6. Final Output:
        return {
            "status": "replied",
            "product_area": product_area,
            "response": response,
            "justification": "Response generated using retrieved support documentation.",
            "request_type": request_type,
        }


if __name__ == "__main__":
    agent = SupportAgent()

    test_ticket = {
        "issue": "I see an unauthorized transaction on my card",
        "subject": "",
        "company": "Visa",
    }

    result = agent.process_ticket(
        test_ticket["issue"],
        test_ticket["subject"],
        test_ticket["company"],
    )

    print(result)
