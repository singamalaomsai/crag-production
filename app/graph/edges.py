from typing import Literal
from app.graph.state import GraphState


def decide_to_generate(state: GraphState) -> Literal["transform_query", "generate"]:
    """
    Evaluates whether the document context is sufficient or requires web fallback.
    """
    web_search = state.get("web_search", False)
    if web_search:
        return "transform_query"
    return "generate"