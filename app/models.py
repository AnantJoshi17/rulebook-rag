"""Pydantic schemas for the /ask contract."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ResponseType(str, Enum):
    """The three outcomes the system must choose correctly between."""

    ANSWERED = "answered"
    NOT_COVERED = "not_covered"
    CONFLICT = "conflict"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(5, ge=1, le=15, description="How many passages to retrieve.")


class Citation(BaseModel):
    """A retrieved passage, always returned alongside the answer."""

    section_id: str = Field(..., description="e.g. 'AR-3.2'")
    heading: str
    document: str = Field(..., description="Human-readable source document title.")
    source_file: str
    source_format: str = Field(..., description="markdown | pdf | table")
    similarity: float = Field(..., description="Cosine similarity to the question, 0-1.")
    text: str


class ConflictClaim(BaseModel):
    """One side of a detected disagreement."""

    section_id: str
    document: str
    stated_value: str
    kind: str = Field(..., description="value | override")
    sentence: str


class ConflictDetail(BaseModel):
    topic: str = Field(..., description="Machine key for the contested rule.")
    topic_label: str = Field(..., description="Human-readable description of the contested rule.")
    claims: List[ConflictClaim]


class AskResponse(BaseModel):
    question: str
    response_type: ResponseType
    answer: str
    citations: List[Citation]
    conflict: Optional[ConflictDetail] = None
    reasoning: str = Field(
        ..., description="Why this response type was chosen, in one line. Shown in the UI."
    )
    retrieval_backend: str
    answer_backend: str
    latency_ms: float
