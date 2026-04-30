# app.py

import streamlit as st

from src.pdf_loader import extract_pdf
from src.chunker import build_chunks
from src.hybrid_indexing import HybridIndex
from src.router import route_query
from src.rag_chain import answer_with_langchain


st.set_page_config(
    page_title="RAG4LAW",
    page_icon="⚖️",
    layout="wide"
)


def load_css(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        css = f.read()

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css("styles.css")


# -----------------------------
# Session state
# -----------------------------

if "case_index" not in st.session_state:
    st.session_state.case_index = None

if "legal_index" not in st.session_state:
    st.session_state.legal_index = None

if "case_chunks" not in st.session_state:
    st.session_state.case_chunks = []

if "legal_chunks" not in st.session_state:
    st.session_state.legal_chunks = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_route" not in st.session_state:
    st.session_state.last_route = None

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# -----------------------------
# Header
# -----------------------------

st.markdown("""
<div class="main-card title-card">
    <h1 class="rag-title">RAG4LAW</h1>
    <p class="small-muted">
        Citation-grounded legal case preparation assistant for uploaded case documents and legal corpus.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.title("Case Workspace")

    st.subheader("1. Upload Court / Case Documents")

    case_files = st.file_uploader(
        "Upload transcripts, affidavits, petitions, evidence PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key="case_files"
    )

    if st.button("Process Case Documents", use_container_width=True):
        if not case_files:
            st.warning("Upload at least one case document.")
        else:
            with st.spinner("Extracting, chunking, and indexing case documents..."):
                all_pages = []

                for file in case_files:
                    pages = extract_pdf(file)
                    all_pages.extend(pages)

                chunks = build_chunks(all_pages, source_type="case_docs")

                index = HybridIndex()
                index.build(chunks)

                st.session_state.case_chunks = chunks
                st.session_state.case_index = index

                st.session_state.chat_history = []
                st.session_state.last_sources = []
                st.session_state.last_route = None

            st.success(
                f"Processed {len(case_files)} case file(s), {len(chunks)} chunks."
            )

    st.divider()

    st.subheader("2. Upload Legal Corpus")

    legal_files = st.file_uploader(
        "Upload laws, bare acts, judgments, precedents",
        type=["pdf"],
        accept_multiple_files=True,
        key="legal_files"
    )

    if st.button("Process Legal Corpus", use_container_width=True):
        if not legal_files:
            st.warning("Upload at least one legal corpus document.")
        else:
            with st.spinner("Extracting, chunking, and indexing legal corpus..."):
                all_pages = []

                for file in legal_files:
                    pages = extract_pdf(file)
                    all_pages.extend(pages)

                chunks = build_chunks(all_pages, source_type="legal_db")

                index = HybridIndex()
                index.build(chunks)

                st.session_state.legal_chunks = chunks
                st.session_state.legal_index = index

                st.session_state.chat_history = []
                st.session_state.last_sources = []
                st.session_state.last_route = None

            st.success(
                f"Processed {len(legal_files)} legal file(s), {len(chunks)} chunks."
            )

    st.divider()

    st.subheader("Status")

    if st.session_state.case_index is None:
        st.warning("No case documents indexed yet.")
    else:
        st.success("Case index ready.")
        st.caption(f"Case chunks: {len(st.session_state.case_chunks)}")

    if st.session_state.legal_index is None:
        st.warning("No legal corpus indexed yet.")
    else:
        st.success("Legal corpus index ready.")
        st.caption(f"Legal chunks: {len(st.session_state.legal_chunks)}")

    if st.session_state.last_route:
        st.caption(f"Last route: {st.session_state.last_route}")

    st.divider()

    if st.button("Reset Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_sources = []
        st.session_state.last_route = None
        st.success("Chat memory reset.")


# -----------------------------
# Main layout
# -----------------------------

left, right = st.columns([2, 1])

with left:
    st.markdown("### Case Chat")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

with right:
    st.markdown("### Retrieval Panel")

    if st.session_state.last_route:
        st.markdown(
            f'<span class="route-pill">Route: {st.session_state.last_route}</span>',
            unsafe_allow_html=True
        )
    else:
        st.caption("Ask a question to see route and retrieved sources.")

    st.write("")

    if st.session_state.last_sources:
        with st.expander("Retrieved Sources", expanded=True):
            for i, c in enumerate(st.session_state.last_sources, start=1):
                doc_name = c.get("doc_name", "unknown")
                page = c.get("page", "N/A")
                source_type = c.get("source_type", "unknown")
                doc_type = c.get("doc_type", "unknown")
                chunk_type = c.get("chunk_type", "unknown")
                text = c.get("text", "")

                st.markdown(f"**Source {i}**")
                st.caption(
                    f"{doc_name} — Page {page}  \n"
                    f"Source: {source_type}  \n"
                    f"Type: {doc_type} | Chunk: {chunk_type}"
                )
                st.write(text[:900])
                st.divider()
    else:
        st.info("No sources retrieved yet.")


# -----------------------------
# Query handling
# -----------------------------

query = st.chat_input(
    "Ask about facts, evidence, arguments, contradictions, laws, precedents, or timeline..."
)

if query:
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })

    route_info = route_query(query)
    route = route_info["route"]
    st.session_state.last_route = route

    retrieved_chunks = []

    if route == "case_docs":
        if st.session_state.case_index is None:
            st.warning("Process case documents first.")
            st.stop()

        retrieved_chunks = st.session_state.case_index.search(query, top_k=8)

    elif route == "legal_db":
        if st.session_state.legal_index is None:
            st.warning("Process legal corpus first.")
            st.stop()

        retrieved_chunks = st.session_state.legal_index.search(query, top_k=8)

    elif route == "both":
        if st.session_state.case_index is None:
            st.warning("Process case documents first.")
            st.stop()

        case_chunks = st.session_state.case_index.search(query, top_k=5)

        legal_chunks = []
        if st.session_state.legal_index is not None:
            legal_chunks = st.session_state.legal_index.search(query, top_k=5)

        retrieved_chunks = case_chunks + legal_chunks

    else:
        retrieved_chunks = []

    st.session_state.last_sources = retrieved_chunks

    with st.spinner("Analyzing retrieved sources..."):
        answer = answer_with_langchain(
            question=query,
            chunks=retrieved_chunks,
            route=route,
            chat_history=st.session_state.chat_history
        )

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()