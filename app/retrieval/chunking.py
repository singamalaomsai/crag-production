import hashlib
from typing import Any, Dict, List
import tiktoken
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    content: str
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetadataChunker:
    def __init__(
        self,
        model_name: str = "cl100k_base",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")

    def _generate_chunk_id(self, doc_id: str, index: int, content: str) -> str:
        payload = f"{doc_id}:{index}:{content[:64]}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def split_text(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] | None = None,
    ) -> List[DocumentChunk]:
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        chunks: List[DocumentChunk] = []

        if total_tokens == 0:
            return chunks

        base_meta = metadata or {}
        step = self.chunk_size - self.chunk_overlap
        chunk_index = 0

        for start_idx in range(0, total_tokens, step):
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_content = self.tokenizer.decode(chunk_tokens).strip()

            if not chunk_content:
                continue

            chunk_id = self._generate_chunk_id(doc_id, chunk_index, chunk_content)
            
            chunk_meta = {
                **base_meta,
                "doc_id": doc_id,
                "token_start": start_idx,
                "token_end": end_idx,
                "is_first": (start_idx == 0),
                "is_last": (end_idx >= total_tokens),
            }

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    token_count=len(chunk_tokens),
                    metadata=chunk_meta,
                )
            )
            chunk_index += 1

            if end_idx >= total_tokens:
                break

        return chunks