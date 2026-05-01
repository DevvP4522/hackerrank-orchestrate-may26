from __future__ import annotations

import csv
import sys
from pathlib import Path

# Ensure local `code/` package is importable when running this file directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from code.agent import SupportAgent


def main() -> None:
    agent = SupportAgent()

    # TEST CASE inside main() (required)
    print(agent.process_ticket("I can't login to my account", "", "HackerRank"))

    project_root = Path(__file__).resolve().parent.parent
    input_csv = project_root / "support_tickets" / "support_tickets.csv"
    output_csv = project_root / "support_tickets" / "output.csv"

    fieldnames = [
        "status",
        "product_area",
        "response",
        "justification",
        "request_type",
    ]

    fallback_output = {
        "status": "escalated",
        "product_area": "general",
        "response": "This issue requires further review by our support team.",
        "justification": "Error during processing",
        "request_type": "invalid",
    }

    empty_issue_output = {
        "status": "escalated",
        "product_area": "general",
        "response": "This issue requires further review by our support team.",
        "justification": "Insufficient or unclear input (empty issue) → escalated",
        "request_type": "invalid",
    }

    with input_csv.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        print("CSV Columns:", reader.fieldnames)

        if reader.fieldnames is None:
            rows: list[dict[str, str]] = []
        else:
            rows = list(reader)

    with output_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            print(f"Processing ticket {idx}...")

            issue = (row.get("issue") or row.get("Issue") or "").strip()
            subject = (row.get("subject") or row.get("Subject") or "").strip()
            company = (row.get("company") or row.get("Company") or "").strip()

            # robust fallback (required)
            if not issue and subject:
                issue = subject

            # DEBUG logging (required)
            print(f"DEBUG → Issue: '{issue}' | Subject: '{subject}' | Company: '{company}'")

            try:
                # validation (required)
                if not issue and not subject:
                    output_row = {k: empty_issue_output.get(k, fallback_output[k]) for k in fieldnames}
                else:
                    result = agent.process_ticket(issue, subject, company)
                    output_row = {k: result.get(k, fallback_output[k]) for k in fieldnames}
            except Exception:
                output_row = fallback_output

            writer.writerow(output_row)

    print("Output saved to support_tickets/output.csv")


if __name__ == "__main__":
    main()
