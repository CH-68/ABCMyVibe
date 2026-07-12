import csv
import json
import re
from typing import Any, Dict, List

import streamlit as st

from helper_functions import llm as llm_client
from crewai import Agent, Crew, Process, Task


DEFAULT_POLICY_VERSION = "v1"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _append_position(text: str, page: str, header: str) -> str:
    return f"{text.strip()} [Page {page}, {header}]"


def _highlight_deviation(policy_text: str, document_text: str, verdict: str) -> str:
    normalized_policy = _normalize_text(policy_text)
    _ = _normalize_text(document_text)

    if verdict == "satisfied":
        return "No deviation detected"

    if verdict == "missing":
        if "must include" in normalized_policy.lower() or "must be present" in normalized_policy.lower():
            return "*omission*"
        return "*missing*"

    if verdict == "conflicting":
        return "*contradiction*"

    return "*ambiguous*"


def _find_relevant_excerpt(full_text: str, query: str, max_length: int = 250) -> str:
    if not full_text:
        return "No excerpt available."

    query = query or ""
    normalized_full = full_text.lower()
    query_phrases = [query.strip()]
    if len(query) < 40:
        query_phrases = [piece.strip() for piece in re.split(r"[\n\r,.;:\-()]+", query) if piece.strip()]

    query_phrases = sorted(query_phrases, key=len, reverse=True)
    for phrase in query_phrases:
        if len(phrase) < 8:
            continue
        idx = normalized_full.find(phrase.lower())
        if idx != -1:
            start = max(idx - 80, 0)
            end = min(idx + len(phrase) + 80, len(full_text))
            excerpt = full_text[start:end].strip().replace("\n", " ")
            if start > 0:
                excerpt = "..." + excerpt
            if end < len(full_text):
                excerpt = excerpt + "..."
            return excerpt

    excerpt = full_text.strip().replace("\n", " ")
    if len(excerpt) > max_length:
        excerpt = excerpt[:max_length].rstrip() + "..."
    return excerpt or "No excerpt available."


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

        policy_excerpt = _append_position(requirement["clause"], "1", "Policy")
        user_excerpt = _append_position(document_text.strip() or "No user excerpt provided", "1", "User Document")
        deviation_highlight = _highlight_deviation(requirement["clause"], document_text, verdict)

        suggested_edits = f"Add language that covers: {requirement['clause']}"
        finding = {
            "requirement_id": requirement["requirement_id"],
            "verdict": verdict,
            "severity": severity,
            "issue_type": issue_type,
            "policy_excerpt": policy_excerpt,
            "user_excerpt": user_excerpt,
            "deviation_highlight": deviation_highlight,
            "suggested edits": suggested_edits,
            "sggested edits": suggested_edits,
        }
        findings.append(finding)

    report = {
        "policy_version": DEFAULT_POLICY_VERSION,
        "summary": {
            "total_findings": len(findings),
            "counts": counts,
        },
        "findings": findings,
    }
    return report


def _parse_llm_items(llm_text: str) -> List[str]:
    """Split the LLM narrative into discrete items (one per line/numbered entry)."""
    items: List[str] = []
    # split on numbered lines or bullets
    for line in llm_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # remove leading numbering like '1.' or '-'
        line = re.sub(r'^\d+\.|^-|^•', '', line).strip()
        if line:
            items.append(line)
    return items


def _verify_item(policy_text: str, document_text: str, item_text: str) -> Dict[str, str]:
    """Deterministic verifier for a single LLM-produced item.

    Returns a dict with status (pass/fail/insufficient), rationale, and suggested_edit.
    """
    normalized_policy = policy_text.lower()
    normalized_doc = document_text.lower()
    normalized_item = item_text.lower()

    # Try to find explicit citations like [page 2] or [section 2.1]
    citation_match = re.search(r"\[(page|section)\s*[^\]]+\]", item_text, re.IGNORECASE)
    citation = citation_match.group(0) if citation_match else ""

    # Heuristics: if item references policy language present in policy_text
    found_in_policy = any(phrase.strip() in normalized_policy for phrase in re.findall(r"\b[\w\- ]{6,}\b", normalized_item))
    found_in_doc = any(phrase.strip() in normalized_doc for phrase in re.findall(r"\b[\w\- ]{6,}\b", normalized_item))

    # More robust checks: look for exact quoted snippets
    quoted = re.findall(r'"([^"]{6,})"', item_text)
    if quoted:
        found_in_policy = any(q.lower() in normalized_policy for q in quoted)
        found_in_doc = any(q.lower() in normalized_doc for q in quoted)

    result = {
        "status": "insufficient evidence",
        "rationale": "Unable to deterministically verify the claim from provided texts.",
        "suggested_edit": "Provide explicit policy clause or user text to support this claim.",
        "citation": citation,
    }

    if found_in_policy and found_in_doc:
        result["status"] = "pass"
        result["rationale"] = "Claim matches language present in both Policy and User Document."
        result["suggested_edit"] = "None required."
    elif found_in_policy and not found_in_doc:
        result["status"] = "fail"
        result["rationale"] = "Policy requires or references this clause but the User Document does not contain matching language."
        result["suggested_edit"] = f"Add language aligning to policy clause: {item_text}"
    elif not found_in_policy and found_in_doc:
        result["status"] = "insufficient evidence"
        result["rationale"] = "User Document contains related language but no matching policy clause was found; unclear if it's required."
        result["suggested_edit"] = "Clarify whether this is intentionally different from policy or provide policy citation."

    return result



def build_llm_messages(policy_text: str, document_text: str) -> List[Dict[str, str]]:
    fallback = "I couldn't find that information in the uploaded documents."

    system_prompt = (
        'You are a senior compliance analyst.\n\n'
        'Your task is to compare the User Document against the Policy Document, using the Policy Document as the master standard.\n\n'
        'Instructions:\n\n'
        '1. Use the Policy Document as the authoritative baseline.\n\n'
        '2. Compare the User Document against it and identify any gaps, mismatches, policy violations, or missing requirements.\n\n'
        '3. Structure your answer as: finding, evidence, confidence level.\n\n'
        '4. Include short citations like [Page 2] or [Section 2.2.3] when available in the provided context.\n\n'
        '5. Use only the provided context. Do not use outside facts or assumptions.\n\n'
        '6. If the context is insufficient, say: "{fallback}"\n\n'
        'Now answer the user by comparing the User Document to the Policy Document using only the context provided in the user message.'
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps({"policy_context": policy_text, "user_context": document_text})},
    ]


def generate_llm_narrative(policy_text: str, document_text: str) -> str:
    # build structured chat messages for the LLM helper
    messages = build_llm_messages(policy_text, document_text)

    try:
        llm_response = llm_client.get_completion(messages=messages, model="gpt-4o-mini", temperature=0)
    except Exception as e:
        st.warning(f"LLM call failed: {e}")
        llm_response = "I couldn't find that information in the uploaded documents."

    return llm_response


def setup_and_run_crew(policy_text: str, document_text: str, llm_text: str | None = None) -> Dict[str, Any]:
    """New workflow: ask LLM for a concise narrative, then verify each item deterministically."""
    if llm_text is None:
        llm_text = generate_llm_narrative(policy_text, document_text)

    # use the same structured messages as the LLM call
    messages = build_llm_messages(policy_text, document_text)
    items = _parse_llm_items(llm_text)

    verifier_results = []
    counts = {"pass": 0, "fail": 0, "insufficient evidence": 0}
    for idx, item in enumerate(items, start=1):
        vr = _verify_item(policy_text, document_text, item)
        counts[vr["status"]] += 1
        verifier_results.append({
            "id": f"LLM-{idx}",
            "llm_item": item,
            "verifier": vr,
            "document_excerpt": _find_relevant_excerpt(document_text, item),
            "policy_excerpt": _find_relevant_excerpt(policy_text, item),
        })

    # Build a simple revised document suggestion by aggregating suggested edits for fails
    revised_suggestions = [v["verifier"]["suggested_edit"] for v in verifier_results if v["verifier"]["status"] == "fail"]
    revised_text = "\n".join(revised_suggestions)

    # Construct CSV content lines
    csv_rows = []
    for v in verifier_results:
        csv_rows.append({
            "id": v["id"],
            "llm_item": v["llm_item"],
            "status": v["verifier"]["status"],
            "rationale": v["verifier"]["rationale"],
            "suggested_edit": v["verifier"]["suggested_edit"],
            "citation": v["verifier"].get("citation", ""),
        })

    report = {
        "policy_version": DEFAULT_POLICY_VERSION,
        "llm_narrative": llm_text,
        "llm_system_prompt": messages[0]["content"],
        "summary": {
            "total_findings": len(verifier_results),
            "counts": counts,
        },
        "findings": verifier_results,
        "revised_document": {
            "revised_text": revised_text,
            "change_log": csv_rows,
        },
        "csv_rows": csv_rows,
    }

    return report
