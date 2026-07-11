import json
import re
from typing import Any, Dict, List

from crewai import Agent, Crew, Process, Task


DEFAULT_POLICY_VERSION = "v1"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_requirements(policy_text: str) -> List[Dict[str, str]]:
    clauses = []
    for line in policy_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(("policy control", "requirement", "control")):
            requirement_id = f"REQ-{len(clauses) + 1}"
            clauses.append({"requirement_id": requirement_id, "clause": line})
    return clauses or [{"requirement_id": "REQ-1", "clause": policy_text}]


def _classify_verdict(document_text: str, clause: str) -> tuple[str, str, str]:
    doc_text = _normalize_text(document_text).lower()
    clause_text = _normalize_text(clause).lower()

    if "must include" in clause_text or "must be present" in clause_text:
        keyword = clause_text.split("must")[-1].strip()
        if keyword and keyword in doc_text:
            return "satisfied", "warning", "missing"
        return "missing", "error", "missing"

    if "contact information" in clause_text and "contact" in doc_text:
        return "satisfied", "warning", "missing"

    if "retention" in clause_text and "retention" in doc_text:
        return "satisfied", "warning", "missing"

    return "ambiguous", "warning", "ambiguous"


def build_verification_report(policy_text: str, document_text: str) -> Dict[str, Any]:
    """Build a deterministic, auditable verification report in the requested JSON schema."""
    requirements = _extract_requirements(policy_text)
    findings: List[Dict[str, Any]] = []
    counts = {"satisfied": 0, "missing": 0, "conflicting": 0, "ambiguous": 0}

    for requirement in requirements:
        verdict, severity, issue_type = _classify_verdict(document_text, requirement["clause"])
        counts[verdict] += 1

        evidence = []
        if verdict == "satisfied":
            evidence.append(document_text[:200].strip())
        else:
            evidence.append("No matching evidence was found in the submitted document.")

        finding = {
            "requirement_id": requirement["requirement_id"],
            "policy_clause_excerpt": requirement["clause"],
            "anchor_ids": [requirement["requirement_id"]],
            "verdict": verdict,
            "severity": severity,
            "issue_type": issue_type,
            "comment": "Automated policy check completed.",
            "why": (
                "The submitted document was evaluated against the policy clause using deterministic keyword checks."
                if verdict != "satisfied"
                else "The submitted document explicitly contains the policy-relevant language."
            ),
            "evidence_from_document": evidence,
            "suggested_edits": [
                {
                    "edit_id": f"{requirement['requirement_id']}-edit-1",
                    "anchor_ids": [requirement["requirement_id"]],
                    "edit_type": "insert",
                    "original_text": "",
                    "suggested_text": f"Add language that covers: {requirement['clause']}",
                    "rationale": "This insertion strengthens traceability for the missing requirement.",
                }
            ],
        }
        findings.append(finding)

    top_issues = [
        finding["policy_clause_excerpt"]
        for finding in findings
        if finding["verdict"] in {"missing", "ambiguous", "conflicting"}
    ][:3]

    report = {
        "policy_version": DEFAULT_POLICY_VERSION,
        "summary": {
            "total_findings": len(findings),
            "counts": counts,
            "top_issues": top_issues,
        },
        "findings": findings,
        "revised_document": {
            "revised_text": document_text,
            "change_log": [
                {
                    "edit_id": f"edit-{idx+1}",
                    "anchor_ids": [finding["requirement_id"]],
                    "change_type": "insert",
                    "description": f"Suggested revision for {finding['requirement_id']}",
                }
                for idx, finding in enumerate(findings)
                if finding["verdict"] != "satisfied"
            ],
        },
    }
    return report


def setup_and_run_crew(policy_text: str, document_text: str) -> Dict[str, Any]:
    """Assemble a simple CrewAI workflow around the deterministic verifier."""
    verifier_agent = Agent(
        role="Compliance Verifier",
        goal="Evaluate a document against policy requirements and produce an auditable verification report.",
        backstory=(
            "A careful verifier that uses deterministic checks, preserves auditability, abstains on uncertainty, "
            "and routes unclear cases to human review."
        ),
        verbose=False,
        allow_delegation=False,
    )

    verifier_task = Task(
        description=(
            "Review the provided policy and document text. Produce a JSON report that lists findings, verdicts, evidence, "
            "and suggested edits while abstaining from unsupported conclusions."
        ),
        expected_output="A JSON object matching the requested compliance report schema.",
        agent=verifier_agent,
    )

    crew = Crew(
        agents=[verifier_agent],
        tasks=[verifier_task],
        process=Process.sequential,
        verbose=False,
    )

    report_json = json.dumps(build_verification_report(policy_text=policy_text, document_text=document_text))
    verifier_task.description = f"{verifier_task.description}\nReference data:\n{report_json}"
    _ = crew.kickoff()
    return json.loads(report_json)
