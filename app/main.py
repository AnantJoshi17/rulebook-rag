"""
FastAPI service.

    POST /ask     the one endpoint that matters
    GET  /        the UI
    GET  /health  index stats, useful in the demo video
    GET  /corpus  what got indexed, by document and format

Decision order in `/ask` is deliberate:

    1. retrieve
    2. is this a contested quantity?   -> conflict
    3. is the corpus silent?           -> not_covered
    4. otherwise                       -> answered

Conflict is checked **before** coverage because a contested question is not an
unanswered one — the corpus answers it too many times, which is a different
failure and deserves a different response.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .answerer import (
    answer_backend_name,
    build_answered,
    build_conflict,
    build_not_covered,
    to_citations,
)
from .claims import ClaimIndex
from .config import settings
from .corpus_loader import load_corpus
from .coverage import build_gate
from .models import AskRequest, AskResponse, ResponseType
from .retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rulebook")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    started = time.perf_counter()
    chunks = load_corpus()
    retriever = Retriever(chunks)
    state["chunks"] = chunks
    state["retriever"] = retriever
    state["claims"] = ClaimIndex(chunks)
    state["gate"] = build_gate(
        (f"{c.section_id} {c.heading} {c.text}" for c in chunks), settings.weak_match_threshold
    )
    contested = state["claims"].contested_quantities()
    log.info(
        "Indexed %d sections in %.1fs — %d contested quantities found: %s",
        len(chunks),
        time.perf_counter() - started,
        len(contested),
        ", ".join(contested) or "none",
    )
    yield
    state.clear()


app = FastAPI(
    title="Rulebook QA",
    description=(
        "Question answering over a self-contradicting university rulebook. "
        "Every answer carries its citations; the service distinguishes answered, "
        "not_covered and conflict."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    started = time.perf_counter()
    retriever: Retriever = state["retriever"]
    claims: ClaimIndex = state["claims"]

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    display_hits, pool = retriever.expanded_search(question, top_k=request.top_k)
    if not display_hits:
        raise HTTPException(status_code=500, detail="Retrieval returned nothing; index may be empty.")

    top_score = display_hits[0].score

    # --- 2. conflict ---
    conflict = claims.conflict_for([h.chunk.section_id for h in pool], question)
    if conflict is not None:
        quantity_key, conflicting = conflict
        cited_ids = {c.section_id for c in conflicting}
        # Make sure every clause in the conflict is visible in the citation list,
        # not just the ones that happened to rank in the top-k.
        promoted = [h for h in pool if h.chunk.section_id in cited_ids]
        others = [h for h in display_hits if h.chunk.section_id not in cited_ids]
        shown = promoted + others[: max(0, request.top_k - len(promoted))]
        text, detail = build_conflict(question, quantity_key, conflicting, shown)
        return AskResponse(
            question=question,
            response_type=ResponseType.CONFLICT,
            answer=text,
            citations=to_citations(shown),
            conflict=detail,
            reasoning=(
                f"{len(conflicting)} sections fix different values for the same rule "
                f"({quantity_key}); the corpus disagrees with itself."
            ),
            retrieval_backend=retriever.backend,
            answer_backend=answer_backend_name(),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    # --- 3. not covered ---
    verdict = state["gate"].check(question, top_score)
    if not verdict.covered:
        return AskResponse(
            question=question,
            response_type=ResponseType.NOT_COVERED,
            answer=build_not_covered(verdict, display_hits),
            citations=to_citations(display_hits),
            reasoning=verdict.reason,
            retrieval_backend=retriever.backend,
            answer_backend=answer_backend_name(),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    # --- 4. answered ---
    return AskResponse(
        question=question,
        response_type=ResponseType.ANSWERED,
        answer=build_answered(question, display_hits),
        citations=to_citations(display_hits),
        reasoning=(
            f"Top section {display_hits[0].chunk.section_id} matched at "
            f"{top_score:.2f}; every salient term in the question is present in the corpus."
        ),
        retrieval_backend=retriever.backend,
        answer_backend=answer_backend_name(),
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@app.get("/health")
def health() -> dict:
    retriever: Retriever = state["retriever"]
    claims: ClaimIndex = state["claims"]
    return {
        "status": "ok",
        "sections_indexed": len(state["chunks"]),
        "indexed_words": sum(len(c.text.split()) for c in state["chunks"]),
        "retrieval_backend": retriever.backend,
        "answer_backend": answer_backend_name(),
        "contested_quantities": list(claims.contested_quantities().keys()),
    }


@app.get("/corpus")
def corpus_summary() -> dict:
    documents: dict = {}
    for chunk in state["chunks"]:
        entry = documents.setdefault(
            chunk.document,
            {"source_file": chunk.source_file, "formats": set(), "sections": 0, "words": 0},
        )
        entry["formats"].add(chunk.source_format)
        entry["sections"] += 1
        entry["words"] += len(chunk.text.split())
    for entry in documents.values():
        entry["formats"] = sorted(entry["formats"])
    return {"documents": documents, "total_words": sum(d["words"] for d in documents.values())}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
