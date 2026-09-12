import re
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from app.retrieval.chunking import DocumentChunk


class BM25SearchEngine:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.index: BM25Okapi | None = None

    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and split on non-alphanumeric characters for clean lexical matching
        return re.findall(r"\w+", text.lower())

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(c.content) for c in chunks]
        self.index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 25) -> List[Tuple[DocumentChunk, float]]:
        if not self.index or not self.chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.index.get_scores(tokenized_query)

        # Pair each chunk with its BM25 score
        scored_chunks = list(zip(self.chunks, scores))
        # Sort descending by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return scored_chunks[:top_k]