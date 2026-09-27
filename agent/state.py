from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    # User input
    input_query: str = ""
    query_type: Optional[Literal["topic", "paper"]] = None

    # arXiv retrieval and selection
    candidate_papers: List[Dict[str, Any]] = Field(default_factory=list)
    selected_paper: Optional[Dict[str, Any]] = None

    # Paper data
    paper_metadata: Dict[str, Any] = Field(default_factory=dict)
    paper_text: str = ""
    sections: Dict[str, str] = Field(default_factory=dict)

    # RAG data
    # RAG data
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    vector_store: Any = None

    # Generated briefing and QA history
    briefing: Optional[Dict[str, Any]] = None
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)

    # Execution status and errors
    status: str = "initialized"
    errors: List[str] = Field(default_factory=list)