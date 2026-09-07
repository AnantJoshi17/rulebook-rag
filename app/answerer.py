"""
Turning retrieved passages into one of the three permitted responses.

Answers are **extractive by default**: the text you read is assembled from
sentences that exist in the corpus, so the answer cannot drift from its
citations. The whole point of this system is that it does not invent, and
generative phrasing is the usual way that guarantee gets lost.

An optional LLM pass (set `ANTHROPIC_API_KEY`) rewrites the same retrieved
sentences into smoother prose. It is given only the retrieved passages and is
instructed to refuse if they do not contain the answer. If the call fails for
any reason the extractive answer is returned instead, so the demo never breaks
because a network call did.
"""

from __future__ import annotations

import logging
import re
from typing import List, Sequence

from .claims import Claim, ClaimIndex, QUANTITIES_BY_KEY
from .config import settings
from .coverage import CoverageVerdict
from .models import Citation, ConflictClaim, ConflictDetail, ResponseType
from .retriever import ScoredChunk

log = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(])")
_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "from", "with", "and",
    "or", "is", "are", "be", "shall", "may", "must", "that", "this", "which",
    "what", "how", "do", "does", "i", "my", "me", "can", "if", "it",
}


def _clean(text: str) -> str:
    text = text.replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(_clean(text)) if len(s.strip()) > 20]


def _keywords(question: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9%\.]+", question.lower()) if w not in _STOP and len(w) > 2}


def _rank_sentences(question: str, hits: Sequence[ScoredChunk], limit: int = 3) -> List[tuple[str, str]]:
    """Pick the sentences that actually answer the question. Returns (sentence, section_id)."""
    keys = _keywords(question)
    scored: List[tuple[float, str, str]] = []
    for rank, hit in enumerate(hits):
        for sentence in _sentences(hit.chunk.text):
            words = _keywords(sentence)
            if not words:
                continue
            overlap = len(keys & words) / max(len(keys), 1)
            # Retrieval rank is a prior; sentence overlap refines within a section.
            score = overlap + hit.score * 0.5 - rank * 0.04
            scored.append((score, sentence, hit.chunk.section_id))
    scored.sort(key=lambda t: -t[0])

    picked: List[tuple[str, str]] = []
    seen: set[str] = set()
    for _, sentence, section_id in scored:
        if sentence in seen:
            continue
        seen.add(sentence)
        picked.append((sentence, section_id))
        if len(picked) == limit:
            break
    return picked


def to_citations(hits: Sequence[ScoredChunk]) -> List[Citation]:
    return [
        Citation(
            section_id=h.chunk.section_id,
            heading=h.chunk.heading,
            document=h.chunk.document,
            source_file=h.chunk.source_file,
            source_format=h.chunk.source_format,
            similarity=round(h.score, 4),
            text=_clean(h.chunk.text),
        )
        for h in hits
    ]


# --------------------------------------------------------------------------- #
# The three response builders
# --------------------------------------------------------------------------- #


def build_answered(question: str, hits: Sequence[ScoredChunk]) -> str:
    picked = _rank_sentences(question, hits)
    if not picked:
        return _clean(hits[0].chunk.text)[:600]

    parts = [f"{sentence} [{section_id}]" for sentence, section_id in picked]
    body = " ".join(parts)

    if settings.llm_available:
        polished = _llm_polish(question, hits, body)
        if polished:
            return polished
    return body


def build_not_covered(verdict: CoverageVerdict, hits: Sequence[ScoredChunk]) -> str:
    lines = ["The rulebook does not answer this. Nothing in the corpus covers it."]
    if verdict.unsupported_terms:
        listed = ", ".join(verdict.unsupported_terms)
        lines.append(
            f"No section mentions {listed}. The closest sections deal with related "
            "circumstances, but they do not extend to this one."
        )
    if hits:
        nearest = ", ".join(f"{h.chunk.section_id} ({h.chunk.heading})" for h in hits[:3])
        lines.append(f"Nearest sections, for context only: {nearest}.")
    lines.append(
        "Treating any of those as an answer would be inventing a rule. If you need a "
        "decision on this, it has to come from the authority named in AR-1.2."
    )
    return " ".join(lines)


def build_conflict(
    question: str, quantity_key: str, claims: Sequence[Claim], hits: Sequence[ScoredChunk]
) -> tuple[str, ConflictDetail]:
    quantity = QUANTITIES_BY_KEY[quantity_key]
    ordered = sorted(claims, key=lambda c: (c.kind == "override", c.section_id))

    lines = [
        f"The rulebook gives more than one answer to this. {quantity.label} is stated "
        f"{len(ordered)} different ways:"
    ]
    for claim in ordered:
        if claim.kind == "override":
            lines.append(
                f"— {claim.section_id} ({claim.document}) claims the power to set the "
                f"requirement aside altogether."
            )
        else:
            lines.append(f"— {claim.section_id} ({claim.document}) fixes it at {claim.stated_value}.")
    lines.append(
        "These cannot all be applied at once, and none is written as an exception to "
        "the others. This needs a ruling from the Academic Council under AR-1.2 rather "
        "than an answer from me."
    )

    detail = ConflictDetail(
        topic=quantity_key,
        topic_label=quantity.label,
        claims=[
            ConflictClaim(
                section_id=c.section_id,
                document=c.document,
                stated_value=c.stated_value,
                kind=c.kind,
                sentence=_clean(c.sentence),
            )
            for c in ordered
        ],
    )
    return "\n".join(lines), detail


# --------------------------------------------------------------------------- #
# Optional LLM polish
# --------------------------------------------------------------------------- #

_POLISH_SYSTEM = (
    "You rewrite answers drawn from a university rulebook. You are given the retrieved "
    "passages and a draft answer built only from them. Rewrite the draft into two or "
    "three clear sentences. Keep every bracketed section reference exactly as it "
    "appears. Add no fact that is not in the passages. If the passages do not answer "
    "the question, reply with exactly: INSUFFICIENT."
)


def _llm_polish(question: str, hits: Sequence[ScoredChunk], draft: str) -> str | None:
    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        passages = "\n\n".join(
            f"[{h.chunk.section_id}] {h.chunk.heading}\n{_clean(h.chunk.text)}" for h in hits
        )
        message = client.messages.create(
            model=settings.llm_model,
            max_tokens=400,
            system=_POLISH_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\nRetrieved passages:\n{passages}\n\n"
                        f"Draft answer:\n{draft}"
                    ),
                }
            ],
        )
        text = "".join(block.text for block in message.content if block.type == "text").strip()
        if not text or text.strip().upper().startswith("INSUFFICIENT"):
            return None
        return text
    except Exception as exc:
        log.warning("LLM polish failed (%s) — using extractive answer.", exc)
        return None


def answer_backend_name() -> str:
    return f"llm-polish({settings.llm_model})" if settings.llm_available else "extractive"
