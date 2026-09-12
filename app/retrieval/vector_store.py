from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from app.retrieval.chunking import DocumentChunk


class QdrantVectorStore:
    def __init__(
        self,
        collection_name: str = "crag_collection",
        model_name: str = "BAAI/bge-small-en-v1.5",
    ):
        self.collection_name = collection_name
        self.encoder = SentenceTransformer(model_name)
        self.vector_size = self.encoder.get_sentence_embedding_dimension()

        # In-memory client for instant local execution
        self.client = QdrantClient(":memory:")
        self._init_collection()

    def _init_collection(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return

        texts = [c.content for c in chunks]
        embeddings = self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        points = [
            PointStruct(
                id=idx,
                vector=embeddings[idx].tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "token_count": chunk.token_count,
                    "metadata": chunk.metadata,
                },
            )
            for idx, chunk in enumerate(chunks)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 25) -> List[Tuple[DocumentChunk, float]]:
        query_vector = self.encoder.encode(query, convert_to_numpy=True).tolist()

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        ).points

        results: List[Tuple[DocumentChunk, float]] = []
        for point in search_results:
            payload = point.payload
            chunk = DocumentChunk(
                chunk_id=payload["chunk_id"],
                doc_id=payload["doc_id"],
                chunk_index=payload["chunk_index"],
                content=payload["content"],
                token_count=payload["token_count"],
                metadata=payload["metadata"],
            )
            results.append((chunk, float(point.score)))

        return results