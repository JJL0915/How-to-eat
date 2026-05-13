"""
RAG pipeline data models.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class RAGPipelineResult:
    """一次问答流水线的执行结果，供 API 与评测复用。"""

    question: str
    answer: str = ""
    answer_stream: Optional[Iterable[str]] = None
    query_plan: Dict[str, Any] = field(default_factory=dict)
    processing_question: str = ""
    route_type: str = ""
    answer_style: str = ""
    target_dishes: List[str] = field(default_factory=list)
    focus_dish: str = ""
    conversation_context: str = ""
    conversation_state: Dict[str, Any] = field(default_factory=dict)
    generation_context: str = ""
    filters: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    relevant_chunks: List[Any] = field(default_factory=list)
    relevant_docs: List[Any] = field(default_factory=list)
    refusal_reason: str = ""
