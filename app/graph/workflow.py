from typing import Any, Dict, List
from langgraph.graph import END, StateGraph

from app.graph.edges import decide_to_generate
from app.graph.nodes import (
    generate_node,
    grade_documents_node,
    transform_query_node,
    web_search_node,
)
from app.graph.state import GraphState
from app.retrieval.chunking import DocumentChunk
from app.retrieval.reranker import CrossEncoderReranker, reciprocal_rank_fusion
from app.retrieval.sparse_search import BM25SearchEngine
from app.retrieval.vector_store import QdrantVectorStore


class CRAGPipeline:
    def __init__(self, chunks: List[DocumentChunk]):
        # Initialize Two-Tier Retrieval Engine
        self.bm25 = BM25SearchEngine()
        self.bm25.index_chunks(chunks)

        self.vector_store = QdrantVectorStore()
        self.vector_store.index_chunks(chunks)

        self.reranker = CrossEncoderReranker()

        # Build Graph
        self.app = self._build_graph()

    def _retrieve_node(self, state: GraphState) -> Dict[str, Any]:
        """
        Two-tier hybrid retrieval: Dense + Sparse combined via RRF, then pruned by cross-encoder.
        """
        query = state["question"]

        # Tier 1: Candidate retrieval
        bm25_candidates = self.bm25.search(query, top_k=10)
        vector_candidates = self.vector_store.search(query, top_k=10)
        fused = reciprocal_rank_fusion([bm25_candidates, vector_candidates], k=60)
        candidate_chunks = [chunk for chunk, _ in fused]

        # Tier 2: Precision reranking (Top 3)
        reranked = self.reranker.rerank(query, candidate_chunks, top_n=3)
        top_chunks = [chunk for chunk, _ in reranked]

        return {"documents": top_chunks, "web_search": False, "retry_count": 0}

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        # Register Nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", grade_documents_node)
        workflow.add_node("transform_query", transform_query_node)
        workflow.add_node("web_search", web_search_node)
        workflow.add_node("generate", generate_node)

        # Wire Graph Flow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # Conditional Branching Edge
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )

        workflow.add_edge("transform_query", "web_search")
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    def run(self, question: str) -> Dict[str, Any]:
        initial_state: GraphState = {
            "question": question,
            "documents": [],
            "generation": "",
            "web_search": False,
            "retry_count": 0,
        }
        return self.app.invoke(initial_state)