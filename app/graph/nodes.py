import os
from typing import Any, Dict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tavily import TavilyClient

from app.graph.state import GraphState
from app.retrieval.chunking import DocumentChunk

load_dotenv()


# Pydantic schema for strict binary grading
class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def get_evaluator_llm():
    # Fast, free open-source model running on Groq LPU inference
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)


def grade_documents_node(state: GraphState) -> Dict[str, Any]:
    """
    Evaluates whether retrieved documents are relevant to the user's question.
    If irrelevant chunks are found, triggers web search fallback.
    """
    question = state["question"]
    documents = state["documents"]

    llm = get_evaluator_llm()
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system_prompt = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant ('yes').\n"
        "If it is irrelevant or cannot help answer the question, grade it as not relevant ('no')."
    )

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
        ]
    )

    evaluator = grade_prompt | structured_llm_grader

    filtered_docs = []
    web_search = False

    for doc in documents:
        score = evaluator.invoke({"question": question, "document": doc.content})
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(doc)
        else:
            web_search = True

    if len(filtered_docs) == 0:
        web_search = True

    return {"documents": filtered_docs, "web_search": web_search}


def transform_query_node(state: GraphState) -> Dict[str, Any]:
    """
    Rewrites the user query to optimize it for external web search engines.
    """
    question = state["question"]
    llm = get_evaluator_llm()

    system_prompt = (
        "You are an expert query rewriter. Transform the following user question into an optimized, "
        "keyword-focused web search query that will return the most accurate results."
    )

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Initial question:\n\n{question}\n\nProvide ONLY the improved search query string:"),
        ]
    )

    query_rewriter = re_write_prompt | llm
    better_query = query_rewriter.invoke({"question": question}).content.strip()

    return {"question": better_query}


def web_search_node(state: GraphState) -> Dict[str, Any]:
    """
    Executes a web search via Tavily API and converts results to DocumentChunk format.
    """
    question = state["question"]
    documents = state.get("documents", [])

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=tavily_api_key)

    search_response = client.search(query=question, max_results=3)

    for idx, r in enumerate(search_response.get("results", [])):
        web_chunk = DocumentChunk(
            chunk_id=f"web_{idx}",
            doc_id=r.get("url", "web_source"),
            chunk_index=idx,
            content=r.get("content", ""),
            token_count=len(r.get("content", "").split()),
            metadata={"title": r.get("title", "Web Search Result"), "source": r.get("url", "")},
        )
        documents.append(web_chunk)

    return {"documents": documents}


def generate_node(state: GraphState) -> Dict[str, Any]:
    """
    Synthesizes the final answer using verified context chunks.
    """
    question = state["question"]
    documents = state["documents"]

    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1)

    context = "\n\n---\n\n".join([f"Source ({d.doc_id}):\n{d.content}" for d in documents])

    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
        "to answer the question. If you do not know the answer based on the context, state that you do not know. "
        "Provide factual, concise answers with source citations where appropriate."
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ]
    )

    rag_chain = qa_prompt | llm
    generation = rag_chain.invoke({"context": context, "question": question}).content

    return {"generation": generation}