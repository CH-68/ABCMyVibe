import os
from dotenv import load_dotenv
import tempfile
import traceback
from typing import Any, Sequence

import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import hashlib

# Ensure you import your refactored CrewAI semantic orchestrator function
from src.crew import ComplianceCrew

# ----------------------------
# LLM setup
# ----------------------------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# SET TRACING ENVIRONMENT VARIABLES
os.environ["CREWAI_TRACING"] = "true"

def normalize_uploaded_files(uploaded_files: Any) -> list[Any]:
    """Accept either a single uploaded file or a list of uploaded files."""
    if uploaded_files is None:
        return []
    if isinstance(uploaded_files, (list, tuple)):
        return list(uploaded_files)
    return [uploaded_files]


# ----------------------------
# RAG: Load + Split PDF
# ----------------------------
def load_and_split(uploaded_files: Sequence[Any]) -> list[Document]:
    """Load one or more uploaded PDFs, split them into chunks, and return LangChain Document chunks."""
    files = normalize_uploaded_files(uploaded_files)
    if not files:
        raise ValueError("No files were uploaded.")

    all_chunks: list[Document] = []

    for uploaded_file in files:
        data = uploaded_file.read()
        if not data:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            temp_file_path = tmp.name

        try:
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()

            if not pages:
                continue

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""],
            )
            chunks = text_splitter.split_documents(pages)

            for chunk in chunks:
                metadata = dict(getattr(chunk, "metadata", {}) or {})
                metadata["source"] = getattr(uploaded_file, "name", "uploaded.pdf")
                chunk.metadata = metadata

            all_chunks.extend(chunks)
        except Exception as e:
            st.error(f"Error splitting document {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    if not all_chunks:
        raise ValueError("No readable text could be extracted from the uploaded PDFs.")

    return all_chunks


# ----------------------------
# RAG: Build Vector Store
# ----------------------------
def file_fingerprint(uploaded_files: Sequence[Any]) -> str:
    """Stable-ish id for the uploaded content."""
    files = normalize_uploaded_files(uploaded_files)
    entries = []

    for uploaded_file in files:
        name = getattr(uploaded_file, "name", None) or "uploaded.pdf"
        size = getattr(uploaded_file, "size", None)
        mtime = getattr(uploaded_file, "last_modified", None)
        entries.append(f"{name}|{size}|{mtime}")

    raw = "|".join(entries).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@st.cache_resource(show_spinner=False)
def build_vectorstore(chunks: list[Document], collection_name: str) -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=collection_name,
    )


# ----------------------------
# RAG: System Prompt (Comparison-focused)
# ----------------------------
def build_grounded_rag_system_prompt(policy_context: str, user_context: str) -> str:
    fallback = "I couldn't find that information in the uploaded documents."

    return f"""You are a senior compliance analyst.

Your task is to compare the User Document against the Policy Document, using the Policy Document as the master standard.

Instructions:
1. Use the Policy Document as the authoritative baseline.
2. Compare the User Document against it and identify any gaps, mismatches, policy violations, or missing requirements.
3. Structure your answer as: finding, evidence, confidence level.
4. Include short citations like [Page 2] or [Section 2.2.3] when available in the provided context.
5. Use only the provided context. Do not use outside facts or assumptions.
6. If the context is insufficient, say: "{fallback}"

Policy Document Context:
-----------------
{policy_context}
-----------------

User Document Context:
-----------------
{user_context}
-----------------

Now answer the user's question by comparing the User Document to the Policy Document using only the context above."""


# ----------------------------
# RAG: Retrieve Context
# ----------------------------
def retrieve_context(vectorstore: Chroma, query: str, k: int, score_threshold: float):
    if query is None:
        return None
    query = str(query).strip()
    if not query:
        return None

    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )
        results = retriever.invoke(query)
        if not results:
            st.warning("No relevant context found matching the query and current retrieval settings.")
            return None
        return results
    except Exception as e:
        st.error(f"Error during document retrieval (Check parameters/API): {e}")
        traceback.print_exc()
        return None


# def safe_doc_to_context(doc: Document) -> str:
#     meta = getattr(doc, "metadata", None)
#     if not isinstance(meta, dict):
#         meta = {}

#     source = meta.get("source") or "Unknown Source"
#     try:
#         source = str(source).strip()
#     except Exception:
#         source = "Unknown Source"

#     content = getattr(doc, "page_content", "") or ""
#     try:
#         content = str(content).strip()
#     except Exception:
#         content = ""

#     return f"[Chunk | {source}] {content}".strip()


# def format_docs(label: str, docs: list[Document]) -> str:
#     if not docs:
#         return f"{label}:\nNo relevant excerpts found."

#     chunks = []
#     for doc in docs:
#         context = safe_doc_to_context(doc)
#         if context:
#             chunks.append(context)

#     if not chunks:
#         return f"{label}:\nNo relevant excerpts found."

#     return f"{label}:\n" + "\n\n---\n\n".join(chunks)


# def build_comparison_context(policy_docs: list[Document], user_docs: list[Document], max_chars: int = 12000) -> str:
#     policy_context = format_docs("Policy Document", policy_docs)
#     user_context = format_docs("User Document", user_docs)

#     combined = (
#         "Policy Document Context:\n"
#         f"{policy_context}\n\n"
#         "User Document Context:\n"
#         f"{user_context}"
#     )

#     return combined[:max_chars]


# ----------------------------
# Streamlit UI: Initialise
# ----------------------------
st.set_page_config(page_title="AI Chatbot (RAG Grounded)", page_icon="📚🤖")
st.title("Compliance Checker")

if "messages" not in st.session_state:
    st.session_state["messages"] = []
st.session_state.setdefault("k_value", 9)
st.session_state.setdefault("score_threshold", 0.05)


# ----------------------------
# Streamlit UI: Sidebar & Document Persistence
# ----------------------------
with st.sidebar:
    st.header("📂 Knowledge Bases")

    st.markdown("**Policy Document**")
    policy_files = st.file_uploader(
        "Upload Reference PDF",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.markdown("---")

    st.subheader("**User Document**")
    user_files = st.file_uploader(
        "Upload PDF to verify",
        type=["pdf"],
        accept_multiple_files=True,
    )

    # --- Policy Document Processing ---
    if policy_files:
        try:
            policy_hash = file_fingerprint(policy_files)

            if st.session_state.get("policy_fingerprint") != policy_hash:
                with st.spinner("Processing Policy Documents..."):
                    chunks = load_and_split(policy_files)
                    coll_name = f"chroma_policy_{policy_hash}"

                    st.session_state["policy_vectorstore"] = build_vectorstore(chunks, coll_name)
                    st.session_state["policy_chunks"] = chunks
                    st.session_state["policy_text"] = "\n\n".join([getattr(c, "page_content", "") for c in chunks])
                    st.session_state["policy_fingerprint"] = policy_hash
        except Exception as e:
            st.error(f"Failed to load Policy PDFs: {e}")

    # --- User Document Processing ---
    if user_files:
        try:
            user_hash = file_fingerprint(user_files)

            if st.session_state.get("user_fingerprint") != user_hash:
                with st.spinner("Processing User Documents..."):
                    chunks = load_and_split(user_files)
                    coll_name = f"chroma_user_{user_hash}"

                    st.session_state["user_vectorstore"] = build_vectorstore(chunks, coll_name)
                    st.session_state["user_chunks"] = chunks
                    st.session_state["user_text"] = "\n\n".join([getattr(c, "page_content", "") for c in chunks])
                    st.session_state["user_fingerprint"] = user_hash
        except Exception as e:
            st.error(f"Failed to load User PDFs: {e}")

    # --- Run Verification ---
    st.markdown("---")
    st.subheader("Verification")
    if st.button("Run Verification"):
        policy_text = st.session_state.get("policy_text", "")
        user_text = st.session_state.get("user_text", "")
        if not policy_text or not user_text:
            st.error("Please upload both Policy PDFs and User PDFs to run verification.")
        else:
            with st.spinner("Running semantic agentic verification..."):
                try:
                    # 1. Initialize your new CrewAI class
                    compliance_crew_instance = ComplianceCrew()
                    
                    # 2. Get the crew object 
                    crew_obj = compliance_crew_instance.crew()
                    
                    # 3. Pass your session_state variables as inputs to the crew
                    # Note: Match these keys with the placeholders in your config/tasks.yaml
                    inputs = {
                        "policy_text": policy_text,
                        "document_text": user_text
                    }
                    
                    # 4. Kick off the crew and grab the structured Pydantic report
                    result = crew_obj.kickoff(inputs=inputs)
                    
                    # 5. Save the structured data right into your session state
                    st.session_state["verification_report"] = result.pydantic.model_dump()
                    
                    st.success("Verification completed")
                except Exception as e:
                    st.error(f"Verification failed: {e}")


# ----------------------------
# Streamlit UI: Render Chat
# ----------------------------
for entry in st.session_state["messages"]:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])


# ----------------------------
# Streamlit UI: Verification Results
# ----------------------------
verification_report = st.session_state.get("verification_report")
if verification_report: 
    st.markdown("---")
    st.subheader("Verification Findings")
    summary = verification_report.get("summary", {}).get("counts", {})
    st.markdown(
        f"""**Total Serious Issues:** {summary.get('failed', 0)}
        **Verification cleared:** {summary.get('cleared', 0)}"""
    )

    # Filter findings where alignment isn't completely satisfied
    findings = [
        finding
        for finding in verification_report.get("findings", [])
        if finding.get("verification") in ["failed", "partially_cleared", "contradiction", "omission", "additional"]
    ]
    
    if findings:
        df = pd.DataFrame(findings)
        
        # Select and map column keys matching your new ComplianceReportSchema Pydantic attributes
        df = df[
            [
                "requirement_id",
                "verification",
                "issue_type",
                "policy_citation",
                "user_citation",
                "review_highlight",
                "suggested_edits",
            ]
        ]

        df_columns = {
            "requirement_id": "Requirement",
            "verification": "Verification",
            "issue_type": "Issue Type",
            "policy_citation": "Policy Citation",
            "user_citation": "User Citation",
            "review_highlight": "Deviation Analysis",
            "suggested_edits": "Suggested Edits",
        }
        df = df.rename(columns=df_columns)

        st.dataframe(df, use_container_width=True)

        st.markdown(
            "<style>"
            "div[data-testid='stDataFrame'] td {white-space: pre-wrap; word-break: break-word;}"
            "</style>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No edits are currently required. The document is policy compliant.")

# ----------------------------
# Streamlit UI: Accept user input
# ----------------------------
user_input = st.chat_input("Ask more specific questions on the uploaded documents")

if user_input:
    user_input = user_input.strip()

    policy_vectorstore = st.session_state.get("policy_vectorstore")
    user_vectorstore = st.session_state.get("user_vectorstore")

    if not policy_vectorstore or not user_vectorstore:
        st.warning("Please upload both Policy PDFs and User PDFs to compare them.")
        st.stop()

    with st.spinner("Searching both knowledge bases and generating comparison..."):
        policy_docs = retrieve_context(
            policy_vectorstore,
            user_input,
            k=st.session_state["k_value"],
            score_threshold=st.session_state["score_threshold"],
        ) or []

        user_docs = retrieve_context(
            user_vectorstore,
            user_input,
            k=st.session_state["k_value"],
            score_threshold=st.session_state["score_threshold"],
        ) or []

        context_str = build_comparison_context(policy_docs, user_docs)
        
        policy_split = context_str.split("User Document Context:", 1)[0].replace("Policy Document Context:\n", "").strip()
        user_split = context_str.split("User Document Context:", 1)[1].strip() if "User Document Context:" in context_str else ""
        
        contextual_system_message = build_grounded_rag_system_prompt(
            policy_context=policy_split,
            user_context=user_split,
        )

        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": contextual_system_message},
                    {"role": "user", "content": user_input},
                ],
                temperature=0.1,
            )
            assistant_reply = response.choices[0].message.content

        with st.chat_message("assistant"):
            st.write(assistant_reply)

        st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})