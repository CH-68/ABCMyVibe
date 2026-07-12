import io
import streamlit as st
from pypdf import PdfReader
from helper_functions.utility import check_password
from src.crew import setup_and_run_crew
import pandas as pd
import base64

st.set_page_config(page_title="Compliance Verifier", page_icon="🛡️")
st.title("Agentic Compliance Verifier")
st.caption(
    "CrewAI-based verification for repeatable, auditable policy checks to aid human-in-the-loop."
)

if "report" not in st.session_state:
    st.session_state.report = None

# Do not continue if check_password is not True.
if not check_password():
        st.stop()


def extract_pdf_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages).strip()


with st.sidebar:
    st.header("Policy and target documents")
    policy_file = st.file_uploader("Upload policy PDF", type=["pdf"], key="policy_pdf")
    target_file = st.file_uploader("Upload target document PDF", type=["pdf"], key="target_pdf")

    st.markdown("---")
    if st.button("Run verification", disabled=not (policy_file and target_file)):
        with st.spinner("Running verification..."):
            policy_text = extract_pdf_text(policy_file)
            target_text = extract_pdf_text(target_file)
            if not policy_text or not target_text:
                st.error("Unable to extract readable text from one or both PDFs.")
            else:
                st.session_state.report = setup_and_run_crew(policy_text=policy_text, document_text=target_text)
                st.success("Verification report generated.")


if st.session_state.report:
    report = st.session_state.report
    summary = report.get("summary", {})
    counts = summary.get("counts", {})
    llm_narrative = report.get("llm_narrative", "")

    # Dashboard counts and color indicators
    st.subheader("Agent Verifier Dashboard")
    pass_count = counts.get("pass", 0)
    fail_count = counts.get("fail", 0)
    insuff_count = counts.get("insufficient evidence", 0)
    col_pass, col_fail, col_insuff = st.columns(3)
    col_pass.metric("Pass", pass_count)
    col_fail.metric("Fail", fail_count)
    col_insuff.metric("Insufficient", insuff_count)

    def _status_color(status: str) -> str:
        if status == "pass":
            return "#d4edda"
        if status == "fail":
            return "#f8d7da"
        return "#fff3cd"

    st.subheader("Verification summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Policy version", report.get("policy_version", "n/a"))
    col2.metric("Total findings", summary.get("total_findings", 0))
    col3.metric("Satisfied", counts.get("satisfied", 0))
    col4.metric("Missing", counts.get("missing", 0))

    st.caption("Top issues")
    st.write(", ".join(summary.get("top_issues", [])) or "No major issues identified.")

    st.subheader("Findings table")
    findings = report.get("findings", [])
    if findings:
        # show LLM narrative
        st.subheader("LLM Narrative")
        st.text_area("LLM output", llm_narrative, height=200)

        st.subheader("Verifier Results")
        for f in findings:
            fid = f.get("id")
            item = f.get("llm_item")
            verifier = f.get("verifier", {})
            status = verifier.get("status", "insufficient evidence")
            color = _status_color(status)
            st.markdown(f"<div style='background:{color};padding:8px;border-radius:4px'>\n**{fid}** — {status.upper()}  \n\n**LLM:** {item}  \n\n**Rationale:** {verifier.get('rationale')}  \n\n**Suggested edit:** {verifier.get('suggested_edit')}  \n\n**Citation:** {verifier.get('citation','')}\n</div>", unsafe_allow_html=True)

        # CSV download
        csv_rows = report.get("csv_rows", [])
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f"data:file/csv;base64,{b64}"
            st.markdown(f"[Download findings CSV]({href})")
    else:
        st.info("No findings were generated.")

    st.subheader("Revised document")
    revised = report.get("revised_document", {})
    st.text_area("Revised text preview", revised.get("revised_text", ""), height=220)
    st.dataframe(revised.get("change_log", []))

    with st.expander("Raw JSON report"):
        st.json(report)
else:
    st.info("Upload a policy PDF and a target document PDF, then run the verification workflow.")
