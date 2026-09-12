import os
import streamlit as st
from pypdf import PdfReader

from app.graph.workflow import CRAGPipeline
from app.retrieval.chunking import MetadataChunker

st.set_page_config(
    page_title="CRAG AI Assistant",
    page_icon="🔍",
    layout="wide",
)

# Custom styling for badges and sources
st.markdown(
    """
<style>
    .badge-local {
        background-color: #065f46;
        color: #6ee7b7;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-web {
        background-color: #7c2d12;
        color: #fdba74;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Initialize CRAG Pipeline in Session State (persists across queries)
if "pipeline" not in st.session_state:
  chunker = MetadataChunker(chunk_size=128, chunk_overlap=20)
  initial_docs = [
      (
          "doc_1",
          "PostgreSQL is an open-source object-relational database system"
          " emphasizing extensibility and SQL compliance.",
      ),
      (
          "doc_2",
          "FastAPI is a modern web framework for Python with automatic"
          " interactive OpenAPI documentation.",
      ),
      (
          "doc_3",
          "Error ERR_CONN_REFUSED_502 indicates a bad gateway or unreachable"
          " upstream microservice behind a reverse proxy.",
      ),
      (
          "doc_4",
          "Qdrant is an open-source vector similarity search engine and vector"
          " database written in Rust.",
      ),
  ]
  seed_chunks = []
  for doc_id, text in initial_docs:
    seed_chunks.extend(chunker.split_text(text=text, doc_id=doc_id))

  st.session_state.chunker = chunker
  st.session_state.pipeline = CRAGPipeline(chunks=seed_chunks)
  st.session_state.indexed_files = ["doc_1", "doc_2", "doc_3", "doc_4"]

if "messages" not in st.session_state:
  st.session_state.messages = [{
      "role": "assistant",
      "content": (
          "Hello! I am your Corrective RAG assistant. Ask me questions about"
          " your documents, or upload a PDF from the sidebar to expand my"
          " knowledge."
      ),
      "badge": None,
      "sources": [],
  }]

# Sidebar: Document Ingestion
with st.sidebar:
  st.title("📂 Document Ingestion")
  st.caption("Add knowledge into the hybrid retrieval store")

  uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
  if st.button("Index PDF", use_container_width=True) and uploaded_file:
    with st.spinner("Extracting text and chunking..."):
      reader = PdfReader(uploaded_file)
      pdf_text = ""
      for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
          pdf_text += extracted + "\n\n"

      if pdf_text.strip():
        chunks = st.session_state.chunker.split_text(
            pdf_text, doc_id=uploaded_file.name
        )
        st.session_state.pipeline.add_documents(chunks)
        st.session_state.indexed_files.append(uploaded_file.name)
        st.success(
            f"Indexed {uploaded_file.name} ({len(chunks)} chunks created)!"
        )
      else:
        st.error("Could not extract readable text from this PDF.")

  st.divider()

  with st.expander("Or Paste Raw Text"):
    raw_doc_id = st.text_input("Document ID", placeholder="e.g., office_rules")
    raw_text = st.text_area("Text Content", placeholder="Enter policy text...")
    if st.button("Upload Text", use_container_width=True):
      if raw_doc_id and raw_text:
        chunks = st.session_state.chunker.split_text(
            raw_text, doc_id=raw_doc_id
        )
        st.session_state.pipeline.add_documents(chunks)
        st.session_state.indexed_files.append(raw_doc_id)
        st.success(f"Indexed {raw_doc_id} ({len(chunks)} chunks)!")
      else:
        st.warning("Please provide both an ID and text.")

  st.divider()
  st.subheader("📚 Indexed Documents")
  for doc in set(st.session_state.indexed_files):
    st.write(f"• {doc}")

# Main Chat View
st.title("🔍 Corrective RAG (CRAG) Assistant")
st.caption(
    "Hybrid Retrieval (BM25 + Qdrant) • Reflective Evaluation • Autonomous Web"
    " Fallback"
)

# Display Chat History
for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    if msg.get("badge") == "local":
      st.markdown(
          '<span class="badge-local">Local Document Verified</span>',
          unsafe_allow_html=True,
      )
    elif msg.get("badge") == "web":
      st.markdown(
          '<span class="badge-web">Web Search Fallback Triggered</span>',
          unsafe_allow_html=True,
      )

    st.markdown(msg["content"])

    if msg.get("sources"):
      with st.expander("View Cited Sources"):
        for src in msg["sources"]:
          doc_id = src.get("doc_id", "")
          title = src.get("metadata", {}).get("title", doc_id)
          if doc_id.startswith("http"):
            st.markdown(f"- [{title}]({doc_id})")
          else:
            st.markdown(f"- *Local Doc:* {doc_id}")

# Chat Input & Processing
if user_prompt := st.chat_input("Ask a question..."):
  # Display user message
  st.session_state.messages.append({"role": "user", "content": user_prompt})
  with st.chat_message("user"):
    st.markdown(user_prompt)

  # Run pipeline and display response
  with st.chat_message("assistant"):
    with st.spinner("Retrieving, grading context, and synthesizing answer..."):
      result = st.session_state.pipeline.run(user_prompt)
      answer = result.get("generation", "No response generated.")
      fallback = result.get("web_search_fallback", False)
      sources = result.get("sources", [])
      badge = "web" if fallback else "local"

      if badge == "local":
        st.markdown(
            '<span class="badge-local">Local Document Verified</span>',
            unsafe_allow_html=True,
        )
      else:
        st.markdown(
            '<span class="badge-web">Web Search Fallback Triggered</span>',
            unsafe_allow_html=True,
        )

      st.markdown(answer)

      if sources:
        with st.expander("View Cited Sources"):
          for src in sources:
            doc_id = src.get("doc_id", "")
            title = src.get("metadata", {}).get("title", doc_id)
            if doc_id.startswith("http"):
              st.markdown(f"- [{title}]({doc_id})")
            else:
              st.markdown(f"- *Local Doc:* {doc_id}")

  st.session_state.messages.append({
      "role": "assistant",
      "content": answer,
      "badge": badge,
      "sources": sources,
  })