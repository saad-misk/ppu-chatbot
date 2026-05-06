"""
NLP Server
Runs on port 8001. Called only by the gateway, never directly by clients.

Endpoints:
  POST /process   — full pipeline (preprocess → intent → NER → RAG → generate)
  POST /ingest    — chunk + embed + index a PDF into ChromaDB
  POST /classify  — intent + NER only, no generation (for unit testing)
  POST /evaluate  — run eval suite, return metrics
  GET  /health    — liveness check
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, UploadFile, File, HTTPException

from shared.schemas.message import ProcessRequest, ProcessResponse, Source
from shared.config.settings import settings

# NLP modules
from nlp_engine.preprocessing.normalizer import normalize_for_classification, normalize_for_display
from nlp_engine.preprocessing.tokenizer import light_arabic_stem
from nlp_engine.intent.classifier import get_classifier
from nlp_engine.ner.extractor import extract_entities, entities_to_dict
from nlp_engine.rag.hybrid_retriever import hybrid_retrieve
from nlp_engine.rag.reranker import rerank
from nlp_engine.rag.generator import generate
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

    step_start = time.perf_counter()
    get_classifier()          # loads BERT / zero-shot pipeline
    logger.info("Warmup intent model: %.2fs", time.perf_counter() - step_start)

    step_start = time.perf_counter()
    from nlp_engine.knowledge_base.embed import get_embedder
    get_embedder()            # loads sentence-transformers
    logger.info("Warmup embedder: %.2fs", time.perf_counter() - step_start)

    step_start = time.perf_counter()
    from nlp_engine.knowledge_base.chroma_store import get_store
    get_store()               # opens ChromaDB connection
    logger.info("Warmup ChromaDB: %.2fs", time.perf_counter() - step_start)

    logger.info("NLP Engine ready. Total warmup: %.2fs", time.perf_counter() - total_start)
    yield
    logger.info("NLP Engine shutting down.")


app = FastAPI(title="PPU NLP Engine", version="1.0.0", lifespan=lifespan)

# Module-level helpers (stateless per request)
_state_machine = StateMachine(confidence_threshold=settings.CONFIDENCE_THRESHOLD)


# ---------------------------------------------------------------------------
# Timing middleware — logs per-request latency for every endpoint
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start   = time.perf_counter()
    response = await call_next(request)
    elapsed  = time.perf_counter() - start
    logger.info(
        "[%s] %s %s — %.3fs",
        response.status_code,
        request.method,
        request.url.path,
        elapsed,
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok", "service": "nlp_engine"}


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    """
    Full NLP pipeline:
      1. Normalize & tokenize
      2. Intent classification (BERT)
      3. NER / entity extraction
      4. State machine routing (canned replies / low-confidence fallback)
      5. RAG: retrieve → rerank → generate
      6. Return structured response
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
    logger.info("Session %s | intent=%s (%.2f) | msg='%s'",
                req.session_id, intent, confidence, req.message[:60])

    # ---- 3. NER ----
    entities_list = extract_entities(req.message)   # run on original (not lowercased)
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
    # Load conversation history into context manager
    ctx = ContextManager(max_turns=6)
    ctx.load_history([t.model_dump() for t in req.history])

    # Enrich query with extracted entities for better retrieval
    enriched_query = _enrich_query(clean_text, entities_dict)
    retrieval_query = _expand_retrieval_query(enriched_query)

    # Retrieve top-10, rerank to top-3
    raw_chunks  = hybrid_retrieve(retrieval_query, n_results=10)
    top_chunks  = rerank(enriched_query, raw_chunks, top_k=3)

    # Generate reply
    reply = await generate(
        query=enriched_query,
        context_chunks=top_chunks,
        history=ctx.get_history(),
    )

    # Build Source objects for the response
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


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Chunk a PDF, embed, and index in ChromaDB."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = ingest_pdf(file_bytes=file_bytes, doc_name=file.filename)
    return result


@app.post("/classify")
async def classify(req: ProcessRequest):
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


@app.post("/evaluate")
async def evaluate():
    """Run the evaluation suite and return intent accuracy, NER F1, Precision@k."""
    try:
        metrics = run_evaluation()
        return metrics
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enrich_query(text: str, entities: dict) -> str:
    """
    Optionally append extracted entity values to the query for better retrieval.
    e.g. if COURSE_CODE=CS401, append it to ensure keyword overlap with docs.
    """
    extras = []
    seen = set()
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
    """
    Expand query for retrieval with a light Arabic stemmed variant.
    Keeps the generation query intact while improving recall.
    """
    if not text:
        return text

    stemmed = light_arabic_stem(text)
    if stemmed and stemmed != text:
        return f"{text} {stemmed}"
    return text