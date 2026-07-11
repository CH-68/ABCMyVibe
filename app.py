import io
import streamlit as st
from pypdf import PdfReader
from helper_functions.utility import check_password
from src.crew import setup_and_run_crew

st.set_page_config(page_title="Compliance Verifier", page_icon="🛡️")
st.title("Agentic Compliance Verifier")
st.caption(
    "CrewAI-based verification for repeatable, auditable policy checks for human-in-the-loop."
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
        st.dataframe(findings)
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
