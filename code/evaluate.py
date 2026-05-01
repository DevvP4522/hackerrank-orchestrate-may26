from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure local `hackerrank-orchestrate-may26/code` package is importable as `code`.
# Note: `code` clashes with Python stdlib `code`, so we must force local package resolution.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# If stdlib `code` was imported already, remove it so `code.agent` can resolve to our package.
if "code" in sys.modules:
    mod = sys.modules["code"]
    if not hasattr(mod, "__path__"):  # not a package
        del sys.modules["code"]

from code.agent import SupportAgent
from code.escalation import should_escalate


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _normalize_for_compare(value: Any) -> str:
    return _safe_str(value).strip().lower()


def _compute_accuracy(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return (correct / total) * 100.0


def _extract_expected_fields(row: Dict[str, str]) -> Tuple[str, str, str]:
    # Expected outputs in sample CSV:
    # - Status
    # - Product Area
    # - Request Type
    expected_status = row.get("Status", "")
    expected_product_area = row.get("Product Area", "")
    expected_request_type = row.get("Request Type", "")
    return expected_status, expected_product_area, expected_request_type


def _extract_inputs(row: Dict[str, str]) -> Tuple[str, str, str]:
    issue = row.get("Issue", "")
    subject = row.get("Subject", "")
    company = row.get("Company", "")
    return issue, subject, company


def _get_top_doc_score(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "None"
    first = docs[0]
    score = first.get("score", None)
    if score is None:
        return "None"
    try:
        return f"{float(score):.4f}"
    except Exception:
        return _safe_str(score)


def main() -> None:
    # Ensure local `code/` package is importable when running this file directly.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    sample_csv_path = project_root / "support_tickets" / "sample_support_tickets.csv"
    if not sample_csv_path.exists():
        raise FileNotFoundError(f"Missing sample CSV: {sample_csv_path}")

    agent = SupportAgent()

    total_cases = 0
    correct_status = 0
    correct_product_area = 0
    correct_request_type = 0

    wrong_cases: List[Dict[str, Any]] = []
    escalation_wrong_cases: List[int] = []
    product_area_mismatch_cases: List[int] = []
    request_type_mismatch_cases: List[int] = []

    compared_fields = ("status", "product_area", "request_type")

    with sample_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        case_num = 0
        for row in reader:
            case_num += 1
            total_cases += 1

            try:
                issue, subject, company = _extract_inputs(row)
                expected_status, expected_product_area, expected_request_type = _extract_expected_fields(row)

                result: Dict[str, Any] = agent.process_ticket(issue=issue, subject=subject, company=company)

                predicted_status = _safe_str(result.get("status", ""))
                predicted_product_area = _safe_str(result.get("product_area", ""))
                predicted_request_type = _safe_str(result.get("request_type", ""))

                exp_status_n = _normalize_for_compare(expected_status)
                exp_area_n = _normalize_for_compare(expected_product_area)
                exp_type_n = _normalize_for_compare(expected_request_type)

                pred_status_n = _normalize_for_compare(predicted_status)
                pred_area_n = _normalize_for_compare(predicted_product_area)
                pred_type_n = _normalize_for_compare(predicted_request_type)

                status_ok = pred_status_n == exp_status_n
                area_ok = pred_area_n == exp_area_n
                type_ok = pred_type_n == exp_type_n

                correct_status += 1 if status_ok else 0
                correct_product_area += 1 if area_ok else 0
                correct_request_type += 1 if type_ok else 0

                any_mismatch = not (status_ok and area_ok and type_ok)

                print("\n-----------------------------------")
                print(f"CASE {case_num}")
                print("-----------------------------------")

                print(f"Issue: {_safe_str(issue).strip()}")
                print(
                    f"Expected: status={_safe_str(expected_status)}, "
                    f"area={_safe_str(expected_product_area)}, "
                    f"type={_safe_str(expected_request_type)}"
                )
                print(
                    f"Predicted: status={predicted_status}, "
                    f"area={predicted_product_area}, "
                    f"type={predicted_request_type}"
                )

                if not any_mismatch:
                    print("✔ correct")
                else:
                    print("❌ MISMATCH")
                    diffs: List[str] = []
                    if not status_ok:
                        diffs.append(f"status (expected='{expected_status}' predicted='{predicted_status}')")
                    if not area_ok:
                        diffs.append(f"product_area (expected='{expected_product_area}' predicted='{predicted_product_area}')")
                    if not type_ok:
                        diffs.append(f"request_type (expected='{expected_request_type}' predicted='{predicted_request_type}')")
                    print("Differences: " + "; ".join(diffs))

                    # Track failure patterns
                    wrong_cases.append(
                        {
                            "case_num": case_num,
                            "issue": issue,
                            "expected": {
                                "status": expected_status,
                                "product_area": expected_product_area,
                                "request_type": expected_request_type,
                            },
                            "predicted": {
                                "status": predicted_status,
                                "product_area": predicted_product_area,
                                "request_type": predicted_request_type,
                            },
                            "status_ok": status_ok,
                            "area_ok": area_ok,
                            "type_ok": type_ok,
                        }
                    )

                    # Escalation wrong = mismatch in status where both sides correspond to replied/escalated.
                    if pred_status_n != exp_status_n:
                        if exp_status_n in {"replied", "escalated"} or pred_status_n in {"replied", "escalated"}:
                            escalation_wrong_cases.append(case_num)

                    if not area_ok:
                        product_area_mismatch_cases.append(case_num)
                    if not type_ok:
                        request_type_mismatch_cases.append(case_num)

                    # Optional debug insight:
                    # top retrieved doc score + whether escalation triggered
                    try:
                        text = (_safe_str(issue) + " " + _safe_str(subject)).strip()
                        docs: List[Dict[str, Any]] = []
                        if getattr(agent, "retriever", None) is not None:
                            docs = agent.retriever.query(text, top_k=5)

                        top_score = _get_top_doc_score(docs)
                        escalation_triggered = should_escalate(text, docs)
                        print(f"Debug: top retrieved doc score={top_score}")
                        print(f"Debug: escalation_triggered={escalation_triggered}")
                    except Exception as dbg_exc:
                        print(f"Debug: failed to compute retrieval/escalation insight: {dbg_exc}")

            except Exception as exc:
                print("\n-----------------------------------")
                print(f"CASE {case_num}")
                print("-----------------------------------")
                # Best-effort extraction for visibility; but must not crash.
                issue = ""
                subject = ""
                company = ""
                try:
                    issue, subject, company = _extract_inputs(row)
                except Exception:
                    pass

                print(f"Issue: {_safe_str(issue).strip()}")
                print("❌ MISMATCH")
                print(f"Processing failed: {exc}")

                wrong_cases.append({"case_num": case_num, "issue": issue, "error": str(exc)})

    print("\n-----------------------------------")
    print("FINAL SUMMARY")
    print("-----------------------------------")
    print(f"Total Cases: {total_cases}")
    print(f"Status Accuracy: {_compute_accuracy(correct_status, total_cases):.2f}%")
    print(f"Product Area Accuracy: {_compute_accuracy(correct_product_area, total_cases):.2f}%")
    print(f"Request Type Accuracy: {_compute_accuracy(correct_request_type, total_cases):.2f}%")

    print("\n-----------------------------------")
    print("COMMON FAILURE PATTERNS")
    print("-----------------------------------")

    def _print_case_list(title: str, case_list: List[int]) -> None:
        print(f"\n{title}")
        if not case_list:
            print("✔ none")
            return
        for n in case_list:
            print(f"- Case #{n}")

    _print_case_list("Cases where escalation was wrong:", escalation_wrong_cases)
    _print_case_list("Cases where product_area mismatch:", product_area_mismatch_cases)
    _print_case_list("Cases where request_type mismatch:", request_type_mismatch_cases)


if __name__ == "__main__":
    main()
