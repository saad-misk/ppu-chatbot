"""
NLP Server
Runs on port 8001. Called only by the gateway, never directly by clients.

Endpoints:
  POST /process   — full pipeline (preprocess → intent → NER → RAG → generate)
  POST /ingest    — chunk + embed + index a PDF into ChromaDB
  POST /classify  — intent + NER only, no generation (for unit testing)
  POST /evaluate  — run eval suite, return metrics
  GET  /health    — liveness check

Fix applied (silent crash on Windows after "BM25 index built"):
  Root cause: torch uses libiomp5md.dll (OpenMP) on Windows. When torch's
  encode() was called from inside a ThreadPoolExecutor worker thread
  (via run_in_executor) while torch was already initialised in the main
  thread, Windows raised OMP Error #15 (double DLL init) which calls
  abort() — no Python traceback, no uvicorn error, just silence.

  Fix: convert CPU-heavy endpoints (process, classify) from `async def`
  to plain `def`. FastAPI automatically runs sync endpoints in anyio's
  thread pool — one dedicated thread per request, isolated from the
  asyncio event loop. Torch initialises once per thread and stays there,
  eliminating the double-init entirely.

  ingest() stays `async def` because it must `await file.read()`, but the
  CPU work (ingest_pdf) is offloaded with run_in_executor so the event
  loop is not blocked during PDF parsing and embedding.

  generator.py was also converted from async (AsyncInferenceClient) to
  sync (InferenceClient) to match the sync endpoint pattern.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, HTTPException

from shared.schemas.message import ProcessRequest, ProcessResponse, Source
from shared.config.settings import settings

# NLP modules
from nlp_engine.knowledge_base.chroma_store import get_store
from nlp_engine.preprocessing.normalizer import normalize_for_classification
from nlp_engine.preprocessing.tokenizer import light_arabic_stem
from nlp_engine.intent.classifier import get_classifier
from nlp_engine.ner.extractor import extract_entities, entities_to_dict
from nlp_engine.rag.hybrid_retriever import hybrid_retrieve
from nlp_engine.rag.reranker import rerank
from nlp_engine.rag.generator import generate          # now sync
from nlp_engine.knowledge_base.ingest import ingest_pdf
from nlp_engine.dialogue.state_machine import StateMachine
from nlp_engine.dialogue.context_manager import ContextManager
from nlp_engine.evaluation.eval_runner import run_evaluation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan — warm up heavy models on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up NLP models…")
    total_start = time.perf_counter()

    # ── Windows / OMP fix ───────────────────────────────────────────────────
    # KMP_DUPLICATE_LIB_OK suppresses the OMP Error #15 abort() that occurs
    # when libiomp5md.dll is loaded more than once in the same process.
    # set_num_threads(1) prevents torch from spawning additional OMP threads.
    # Both must be set BEFORE any torch/sentence-transformers import runs.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        logger.info("Torch threading configured (single-threaded, OMP guard set).")
    except Exception as e:
        logger.warning("Could not configure torch threading: %s", e)
    # ────────────────────────────────────────────────────────────────────────

    # Warm up all singletons in the main thread so they are fully initialised
    # before any request thread touches them.
    loop = asyncio.get_event_loop()

    step_start = time.perf_counter()
    await loop.run_in_executor(None, get_classifier)
    logger.info("Warmup intent model: %.2fs", time.perf_counter() - step_start)

    step_start = time.perf_counter()
    from nlp_engine.knowledge_base.embed import get_embedder
    await loop.run_in_executor(None, get_embedder)
    logger.info("Warmup embedder: %.2fs", time.perf_counter() - step_start)

    step_start = time.perf_counter()
    await loop.run_in_executor(None, get_store)
    logger.info("Warmup ChromaDB: %.2fs", time.perf_counter() - step_start)

    logger.info("NLP Engine ready. Total warmup: %.2fs", time.perf_counter() - total_start)
    yield
    logger.info("NLP Engine shutting down.")


app = FastAPI(title="PPU NLP Engine", version="1.0.0", lifespan=lifespan)

_state_machine = StateMachine(confidence_threshold=settings.CONFIDENCE_THRESHOLD)


# ---------------------------------------------------------------------------
# Timing middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    elapsed  = time.perf_counter() - start
    logger.info(
        "[%s] %s %s — %.3fs",
        response.status_code, request.method, request.url.path, elapsed,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "service": "nlp_engine"}


# ---------------------------------------------------------------------------
# /process  — sync def so FastAPI runs it in anyio thread pool
# ---------------------------------------------------------------------------

@app.post("/process", response_model=ProcessResponse)
def process(req: ProcessRequest):
    """
    Full NLP pipeline:
      1. Normalize
      2. Intent classification (BERT / torch)
      3. NER
      4. State machine routing
      5. RAG: hybrid retrieve -> rerank -> generate
      6. Return structured response

    Declared as `def` (not `async def`) so FastAPI runs this in its thread
    pool executor — torch, SentenceTransformer, and ChromaDB calls are all
    safe from a plain thread context with no asyncio involvement.
    """
    # ---- 1. Preprocessing ----
    clean_text = normalize_for_classification(req.message)
    if not clean_text:
        raise HTTPException(status_code=400, detail="Empty message after normalization.")

    # ---- 2. Intent Classification ----
    classifier = get_classifier()
    clf_result = classifier.predict(clean_text)
    intent     = clf_result["intent"]
    confidence = clf_result["confidence"]
    logger.info(
        "Session %s | intent=%s (%.2f) | msg='%s'",
        req.session_id, intent, confidence, req.message[:60],
    )

    # ---- 3. NER ----
    entities_list = extract_entities(req.message)
    entities_dict = entities_to_dict(entities_list)

    # ---- 4. State machine ----
    decision = _state_machine.route(intent, confidence, entities_dict, query=req.message)
    if decision.handled:
        return ProcessResponse(
            reply=decision.reply,
            intent=intent,
            confidence=confidence,
            sources=[],
        )

    # ---- 5. RAG Pipeline ----
    ctx = ContextManager(max_turns=6)
    ctx.load_history([t.model_dump() for t in req.history])

    enriched_query  = _enrich_query(clean_text, entities_dict)
    retrieval_query = _expand_retrieval_query(enriched_query)

    raw_chunks = hybrid_retrieve(retrieval_query, n_results=10)
    top_chunks = rerank(raw_chunks, top_k=3)

    # generate() is now sync — called directly, no await
    reply = generate(
        query=enriched_query,
        context_chunks=top_chunks,
        history=ctx.get_history(),
    )

    sources = [
        Source(
            text=chunk["document"][:300],
            doc_name=chunk["metadata"].get("doc_name", "unknown"),
            page=chunk["metadata"].get("page"),
        )
        for chunk in top_chunks
    ]

    return ProcessResponse(
        reply=reply,
        intent=intent,
        confidence=confidence,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# /ingest  — must stay async (needs await file.read())
# ---------------------------------------------------------------------------

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Chunk a PDF, embed, and index in ChromaDB."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ingest_pdf is CPU-heavy (PDF parse + batch embed) — offload from event loop
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        functools.partial(ingest_pdf, file_bytes=file_bytes, doc_name=file.filename),
    )
    return result


@app.get("/documents")
def list_documents():
    """List indexed documents from ChromaDB."""
    store = get_store()
    return {"documents": store.list_documents()}


# ---------------------------------------------------------------------------
# /classify  — sync def
# ---------------------------------------------------------------------------

@app.post("/classify")
def classify(req: ProcessRequest):
    """Intent classification + NER only — no RAG, no generation."""
    clean_text = normalize_for_classification(req.message)
    classifier = get_classifier()
    clf_result = classifier.predict(clean_text)
    entities   = extract_entities(req.message)
    return {
        "intent":     clf_result["intent"],
        "confidence": clf_result["confidence"],
        "entities":   entities_to_dict(entities),
        "all_scores": clf_result.get("all_scores", {}),
    }


# ---------------------------------------------------------------------------
# /evaluate  — sync def
# ---------------------------------------------------------------------------

@app.post("/evaluate")
def evaluate():
    """Run the evaluation suite and return metrics."""
    try:
        return run_evaluation()
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enrich_query(text: str, entities: dict) -> str:
    extras = []
    seen   = set()
    for values in entities.values():
        for value in values:
            norm = normalize_for_classification(value)
            if norm and norm not in seen:
                seen.add(norm)
                extras.append(norm)
    if extras:
        return text + " " + " ".join(extras)
    return text


def _expand_retrieval_query(text: str) -> str:
    if not text:
        return text
    stemmed = light_arabic_stem(text)
    if stemmed and stemmed != text:
        return f"{text} {stemmed}"
    return text