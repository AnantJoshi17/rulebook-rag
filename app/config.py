"""
Every threshold in one place.

These numbers decide which of the three response types you get, so they are not
buried in the code. `evaluate.py` reports accuracy per response type, which is
how they were tuned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # ---- retrieval ----
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    use_dense_embeddings: bool = _env_bool("USE_DENSE_EMBEDDINGS", True)
    top_k: int = 5
    # How much of the blended score comes from dense embeddings vs TF-IDF.
    dense_weight: float = _env_float("DENSE_WEIGHT", 0.65)
    # Conflict detection looks deeper than the citation list.
    conflict_pool_size: int = 12

    # ---- not_covered gate ----
    # Below this blended similarity, nothing in the corpus is plausibly on-topic.
    weak_match_threshold: float = _env_float("WEAK_MATCH_THRESHOLD", 0.10)
    # A question can score well and still be unanswerable: it may name a specific
    # circumstance ("wedding", "strike") that appears nowhere in the rulebook.
    # This is the main defence against confidently answering the wrong question.
    unsupported_term_gate: bool = _env_bool("UNSUPPORTED_TERM_GATE", True)

    # ---- answer synthesis ----
    # Optional. Without a key the answerer is extractive and fully offline.
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    llm_model: str = os.environ.get("LLM_MODEL", "claude-sonnet-4-6")
    use_llm: bool = _env_bool("USE_LLM", True)

    @property
    def llm_available(self) -> bool:
        return bool(self.anthropic_api_key) and self.use_llm


settings = Settings()
