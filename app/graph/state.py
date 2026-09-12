from typing import List, TypedDict
from app.retrieval.chunking import DocumentChunk


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: bool
    documents: List[DocumentChunk]
    retry_count: int