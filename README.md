# Corrective RAG (CRAG) Production System

A high-reliability, self-correcting Retrieval-Augmented Generation (CRAG) system designed to eliminate LLM hallucinations. Built with *FastAPI, **LangGraph, hybrid retrieval (Qdrant* dense vectors + *BM25* sparse keyword matching), cross-encoder reranking, and autonomous *Tavily Web Search* fallback.

---

## 🏗️ Architecture & Flow

Unlike standard RAG pipelines that blindly forward retrieved text to an LLM, this engine implements an active verification loop:
---

## ✨ Key Features

- *Dual-Tier Hybrid Retrieval:* Combines semantic meaning (dense embeddings via Qdrant) with exact keyword matching (BM25) to accurately surface technical terms, codes, and acronyms.
- *Reflective Document Grading:* Inspects retrieved chunks before generation. If internal knowledge is insufficient, it avoids hallucination by triggering live fallback.
- *Autonomous Web Fallback:* Rewrites the query and queries the live web via Tavily Search API when local context fails evaluation.
- *Strict Citation Attribution:* Enforces bracketed source references ([doc_id] or verified URLs) on all generated claims.
- *In-Browser Client & PDF Extraction:* Built-in web UI capable of reading and chunking client-side PDFs on the fly using pdf.js.

---

## 📂 Project Structure

```text
crag-production/
├── app/
│   ├── api/
│   │   ├── routes.py          # REST endpoints for ingestion and query
│   │   └── schemas.py         # Pydantic input/output contracts
│   ├── graph/
│   │   ├── edges.py           # Conditional routing logic
│   │   ├── nodes.py           # Search, grading, web fallback, and generator nodes
│   │   ├── state.py           # Graph execution state schema
│   │   └── workflow.py        # LangGraph pipeline compiler
│   ├── retrieval/
│   │   ├── chunking.py        # Sliding-window metadata chunker
│   │   ├── reranker.py        # Cross-encoder scoring
│   │   ├── sparse_search.py   # BM25 lexical retriever
│   │   └── vector_store.py    # Qdrant client connection
│   └── main.py                # FastAPI lifecycle & embedded web UI
├── evaluation/
│   └── benchmark_ragas.py     # Faithfulness and answer relevance evaluation
├── .env.example               # Environment variable template
├── .gitignore                 # Files excluded from version control
└── requirements.txt           # Python project dependencies
