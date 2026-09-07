"""
Retrieval over the section chunks.

Two scorers, blended:

  * **Dense** — sentence-transformer embeddings (`all-MiniLM-L6-v2`). Catches
    paraphrase: "how many classes must I attend" -> AR-3.2, which shares almost
    no vocabulary with the question.
  * **Lexical** — TF-IDF over word 1-2 grams. Catches the things embeddings are
    bad at in a legal corpus: exact section IDs, numbers, and rare tokens like
    "caution deposit" or "bonafide".

Blending them is not decoration. Dense alone misses literal lookups; lexical
alone fails every paraphrase. The blend weight is in `config.py`.

If `sentence-transformers` is unavailable, the retriever degrades to lexical-only
rather than crashing. That is a real quality drop and the API reports which
backend served the request, so a demo never silently misleads you.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import settings
from .corpus_loader import Chunk

log = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    dense_score: float
    lexical_score: float


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class Retriever:
    """Builds both indexes at startup and serves blended top-k queries."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks: List[Chunk] = list(chunks)
        self._texts = [c.embedding_text for c in self.chunks]

        # --- lexical index (always available) ---
        self._tfidf = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w[\w\.\-%]*\b",  # keep "8.00", "75%", "AR-3.2" whole
        )
        self._lexical_matrix = self._tfidf.fit_transform(self._texts)

        # --- dense index (optional) ---
        self._model = None
        self._dense_matrix: np.ndarray | None = None
        if settings.use_dense_embeddings:
            self._try_load_dense()

        self.backend = "hybrid(minilm+tfidf)" if self._model else "tfidf-only"
        log.info("Retriever ready: %s over %d sections", self.backend, len(self.chunks))

    def _try_load_dense(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            log.warning(
                "sentence-transformers not installed — falling back to lexical retrieval. "
                "Install it for materially better paraphrase handling."
            )
            return
        try:
            self._model = SentenceTransformer(settings.embedding_model)
            vectors = self._model.encode(
                self._texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
            )
            self._dense_matrix = _l2_normalise(np.asarray(vectors, dtype=np.float32))
        except Exception as exc:  # model download failure, offline machine, etc.
            log.warning("Dense model unavailable (%s) — using lexical retrieval only.", exc)
            self._model = None
            self._dense_matrix = None

    # ------------------------------------------------------------------ #

    def _dense_scores(self, question: str) -> np.ndarray:
        if self._model is None or self._dense_matrix is None:
            return np.zeros(len(self.chunks), dtype=np.float32)
        vector = self._model.encode([question], convert_to_numpy=True)
        vector = _l2_normalise(np.asarray(vector, dtype=np.float32))[0]
        # Both sides are unit vectors, so the dot product *is* cosine similarity.
        return self._dense_matrix @ vector

    def _lexical_scores(self, question: str) -> np.ndarray:
        query_vector = self._tfidf.transform([question])
        # TfidfVectorizer L2-normalises rows, so this dot product is also cosine.
        return np.asarray((self._lexical_matrix @ query_vector.T).todense()).ravel()

    def search(self, question: str, top_k: int | None = None) -> List[ScoredChunk]:
        top_k = top_k or settings.top_k
        dense = self._dense_scores(question)
        lexical = self._lexical_scores(question)

        if self._model is not None:
            weight = settings.dense_weight
            blended = weight * dense + (1.0 - weight) * lexical
        else:
            blended = lexical

        order = np.argsort(-blended)[:top_k]
        return [
            ScoredChunk(
                chunk=self.chunks[i],
                score=float(blended[i]),
                dense_score=float(dense[i]),
                lexical_score=float(lexical[i]),
            )
            for i in order
        ]

    def vocabulary(self) -> set[str]:
        """Every token the corpus actually contains — used by the coverage gate."""
        return set(self._tfidf.vocabulary_.keys())

    def expanded_search(self, question: str, top_k: int) -> Tuple[List[ScoredChunk], List[ScoredChunk]]:
        """
        Returns (display_hits, conflict_pool).

        Conflict detection needs a wider net than the citation list: two clauses
        can disagree while one of them sits at rank 9. The user still only sees
        the top-k.
        """
        pool = self.search(question, top_k=max(top_k, settings.conflict_pool_size))
        return pool[:top_k], pool
