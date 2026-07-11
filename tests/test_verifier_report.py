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
    assert report["revised_document"]["change_log"]
    assert isinstance(json.dumps(report), str)
