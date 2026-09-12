from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., example="What is the error code ERR_CONN_REFUSED_502?")


class SourceDocument(BaseModel):
    doc_id: str
    content: str
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    question: str
    generation: str
    web_search_fallback: bool
    sources: List[SourceDocument]


class IngestDocumentRequest(BaseModel):
    doc_id: str = Field(..., example="doc_custom_01")
    text: str = Field(..., example="Your internal company documentation or knowledge base text.")
    metadata: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    status: str
    doc_id: str
    chunks_created: int