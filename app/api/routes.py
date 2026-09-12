from fastapi import APIRouter, HTTPException, Request
from app.api.schemas import IngestDocumentRequest, IngestResponse, QueryRequest, QueryResponse, SourceDocument

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "CRAG Engine"}


@router.post("/query", response_model=QueryResponse)
async def query_crag(request: Request, body: QueryRequest):
    pipeline = getattr(request.app.state, "crag_pipeline", None)
    if not pipeline:
        raise HTTPException(status_code=503, detail="CRAG Pipeline is not initialized yet.")

    try:
        result = pipeline.run(body.question)
        sources = [
            SourceDocument(
                doc_id=d.doc_id,
                content=d.content,
                metadata=d.metadata or {},
            )
            for d in result.get("documents", [])
        ]

        return QueryResponse(
            question=body.question,
            generation=result.get("generation", ""),
            web_search_fallback=result.get("web_search", False),
            sources=sources,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/ingest", response_model=IngestResponse)
async def ingest_document(request: Request, body: IngestDocumentRequest):
    chunker = getattr(request.app.state, "chunker", None)
    pipeline = getattr(request.app.state, "crag_pipeline", None)

    if not chunker or not pipeline:
        raise HTTPException(status_code=503, detail="Services not ready.")

    new_chunks = chunker.split_text(
        text=body.text,
        doc_id=body.doc_id,
        metadata=body.metadata or {},
    )

    if new_chunks:
        pipeline.bm25.index_chunks(pipeline.bm25.chunks + new_chunks)
        pipeline.vector_store.index_chunks(new_chunks)

    return IngestResponse(
        status="success",
        doc_id=body.doc_id,
        chunks_created=len(new_chunks),
    )