from collections import defaultdict
from typing import Dict, List, Tuple
from sentence_transformers import CrossEncoder
from app.retrieval.chunking import DocumentChunk


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[DocumentChunk, float]]],
    k: int = 60,
) -> List[Tuple[DocumentChunk, float]]:
    """
    Combines ranks from diverse search algorithms (Dense + BM25) into a unified list.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    chunk_map: Dict[str, DocumentChunk] = {}

    for ranked_list in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked_list, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf_scores[chunk.chunk_id] += 1.0 / (k + rank)

    sorted_chunks = sorted(
        [(chunk_map[cid], score) for cid, score in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    return sorted_chunks


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Fast, low-latency cross-encoder for local CPU/GPU execution
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[DocumentChunk],
        top_n: int = 3,
    ) -> List[Tuple[DocumentChunk, float]]:
        if not candidates:
            return []

        pairs = [[query, chunk.content] for chunk in candidates]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(candidates, [float(s) for s in scores]),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_n]