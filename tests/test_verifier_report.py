import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.crew import build_verification_report


def test_build_verification_report_structure():
    policy_text = """
    Policy Control 1: All documents must include a data retention schedule.
    Policy Control 2: Contact information must be present.
    """
    document_text = """
    This document contains contact information.
    """

    report = build_verification_report(policy_text=policy_text, document_text=document_text)

    assert report["policy_version"] == "v1"
    assert report["summary"]["counts"]["missing"] >= 1
    assert report["findings"]
    assert "revised_document" not in report
    first_finding = report["findings"][0]
    assert first_finding["requirement_id"]
    assert first_finding["verdict"] in {"satisfied", "missing", "conflicting", "ambiguous"}
    assert first_finding["severity"] in {"error", "warning"}
    assert first_finding["issue_type"] in {"conflicting", "missing", "ambiguous"}
    assert first_finding["policy_excerpt"].endswith("[Page 1, Policy]")
    assert first_finding["user_excerpt"].endswith("[Page 1, User Document]")
    assert "*" in first_finding["deviation_highlight"]
    assert first_finding["suggested edits"]
    assert isinstance(json.dumps(report), str)
