import os
import traceback
import tempfile
import hashlib

import streamlit as st
from openai import OpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ----------------------------
# LLM setup
# ----------------------------
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ----------------------------
# RAG: Load + Split PDF
# ----------------------------
def load_and_split(uploaded_file) -> list[Document]:
    """Load a PDF from an uploaded file object, split into chunks, return LangChain Document chunks."""
    data = uploaded_file.read()
    if not data:
        raise ValueError("Uploaded file is empty or unreadable.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        temp_file_path = tmp.name

    try:
        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
        )
        return text_splitter.split_documents(pages)
    finally:
        os.remove(temp_file_path)


# ----------------------------
# RAG: Build Vector Store
# ----------------------------
def file_fingerprint(uploaded_file) -> str:
    """
    Stable-ish id for the uploaded content.
    Note: this does NOT hash file bytes; it uses filename/size/mtime-like fields if available.
    """
    name = uploaded_file.name or "uploaded.pdf"
    size = getattr(uploaded_file, "size", None)
    mtime = getattr(uploaded_file, "last_modified", None)
    raw = f"{name}|{size}|{mtime}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


@st.cache_resource(show_spinner=False)
def build_vectorstore(chunks: list[Document], collection_name: str) -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=collection_name,
    )


# ----------------------------
# RAG: System Prompt (Ground-only)
# ----------------------------
def build_grounded_rag_system_prompt(context: str) -> str:
    fallback = "I couldn't find that information in the uploaded document."

    # With "pseudo citations", we ask for bracket citations but do not guarantee they map to
    # actual page/section IDs unless your metadata includes that. (This matches your request.)
    return f"""You are a highly accurate and comprehensive document QA engine.

Your goal is to help the user synthesize, summarize, and extract insights from the provided Context.

Rules:
1. Grounded Synthesis: You may integrate insights across different parts of the Context.
2. No Outside Facts: Use ONLY the provided Context. Do not bring in external information, assumptions, or real-world data not mentioned in the text.
3. Reasoning vs. Guessing: Use only reasoning that is explicitly supported by the Context. Do not speculate beyond what is written.
4. Fallback: If the Context completely lacks information to address the query or form a summary, state clearly: "{fallback}"
5. True citations: Include a short bracket citation like [Page 2] and [Section 2.2.3] (or any bracketed label or header mentioned in the context) in each response.
6. When asked to summarize, list insights with citations, based on the topic asked

Context:
-----------------
{context}
-----------------

Now answer the user's question using ONLY the Context provided above."""


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


def safe_doc_to_context(doc: Document) -> str:
    meta = getattr(doc, "metadata", None)
    if not isinstance(meta, dict):
        meta = {}

    source = meta.get("source") or "Unknown Source"
    try:
        source = str(source).strip()
    except Exception:
        source = "Unknown Source"

    content = getattr(doc, "page_content", "") or ""
    try:
        content = str(content).strip()
    except Exception:
        content = ""

    # Pseudo-citation-friendly prefix
    return f"[Chunk | {source}] {content}".strip()


# ----------------------------
# Streamlit UI: Initialise
# ----------------------------
st.set_page_config(page_title="AI Chatbot (RAG Grounded)", page_icon="📚🤖")
st.title("Document Q&A Engine")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ----------------------------
# Streamlit UI: Sidebar & Document Persistence
# ----------------------------
with st.sidebar:
    st.subheader("📄 Document")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)

    if uploaded_file:
        current_fingerprint = file_fingerprint(uploaded_file)

        if st.session_state.get("current_file_hash") != current_fingerprint:
            with st.spinner("Processing new document..."):
                try:
                    chunks = load_and_split(uploaded_file)
                    coll_name = f"chroma_docs_{current_fingerprint}"

                    st.session_state["vectorstore"] = build_vectorstore(chunks, coll_name)
                    st.session_state["current_file_hash"] = current_fingerprint
                    st.session_state["uploaded_file_name"] = uploaded_file.name

                    # Clear past chat history since a brand new document was uploaded
                    st.session_state["messages"] = []
                except Exception as e:
                    st.error(f"Failed to load and process PDF: {e}")
                    traceback.print_exc()
                    st.session_state.pop("vectorstore", None)
                    st.session_state.pop("current_file_hash", None)

        st.success(f"Active Document: {st.session_state.get('uploaded_file_name')}")
    else:
        st.session_state.pop("vectorstore", None)
        st.session_state.pop("current_file_hash", None)
        st.session_state.pop("uploaded_file_name", None)
        if "vectorstore" not in st.session_state:
            st.info("Upload a PDF to enable document Q&A.")

    st.subheader("🔎 Retrieval")
    st.session_state["k_value"] = st.slider("Top K", min_value=1, max_value=10, value=9, step=1)
    st.session_state["score_threshold"] = st.slider("Score Threshold", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

    st.subheader("🗑️ Conversation")
    if st.button("Clear Conversation"):
        st.session_state["messages"] = []
        st.rerun()

# ----------------------------
# Streamlit UI: Render Chat
# ----------------------------
for entry in st.session_state["messages"]:
    with st.chat_message(entry["role"]):
        st.write(entry["content"])

# ----------------------------
# Streamlit UI: Accept user input
# ----------------------------
user_input = st.chat_input("Ask a question about the uploaded document...")

# Only run inference if the user actually typed something
if user_input:
    user_input = user_input.strip()

    if "vectorstore" not in st.session_state or st.session_state["vectorstore"] is None:
        st.warning("Please upload a PDF document first to enable Q&A functionality.")
        st.stop()

    # Render user message (and persist it)
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.spinner("Searching knowledge base and generating response..."):
        relevant_docs = retrieve_context(
            st.session_state["vectorstore"],
            user_input,
            k=st.session_state["k_value"],
            score_threshold=st.session_state["score_threshold"],
        )

        # Build context string (cap size)
        max_chars = 12000
        if relevant_docs:
            context_chunks = [safe_doc_to_context(doc) for doc in relevant_docs if safe_doc_to_context(doc)]
            context_str = "\n\n---\n\n".join(context_chunks)
            context_str = context_str[:max_chars]
        else:
            context_str = ""

        contextual_system_message = build_grounded_rag_system_prompt(context_str)

        # Call the LLM
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

        # Render + persist assistant message
        with st.chat_message("assistant"):
            st.write(assistant_reply)
        st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
