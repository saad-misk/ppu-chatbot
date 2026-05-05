"""
NLP Server
Runs on port 8001. Called only by the gateway , never directly by clients.

Endpoints:
  POST /process   — full pipeline (intent + NER + RAG + generation)
  POST /ingest    — chunk + embed + index a PDF
  POST /classify  — intent + NER only, no generation (for testing)
  POST /evaluate  — run eval suite, return metrics
  GET  /health    — liveness check
"""
from fastapi import FastAPI, UploadFile, File
from shared.schemas.message import ProcessRequest, ProcessResponse

app = FastAPI(title="PPU NLP Engine", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "nlp_engine"}


@app.post("/process", response_model=ProcessResponse)
async def process(req: ProcessRequest):
    # TODO wire up the full pipeline
    # 1. preprocess (tokenizer + normalizer)
    # 2. intent classification (BERT)
    # 3. NER entity extraction
    # 4. RAG: retrieve → rerank → context inject → generate
    # 5. confidence check → fallback if below threshold
    return ProcessResponse(
        reply="[NLP engine not yet implemented]",
        intent="unknown",
        confidence=0.0,
        sources=[],
    )


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # TODO chunk PDF → embed → store in ChromaDB
    return {"status": "queued", "filename": file.filename}


@app.post("/classify")
async def classify(req: ProcessRequest):
    # TODO return intent + entities, no generation
    return {"intent": "unknown", "confidence": 0.0, "entities": {}}


@app.post("/evaluate")
async def evaluate():
    # TODO  run eval_runner.py, return metrics dict
    return {
        "intent_accuracy": 0.0,
        "ner_f1": 0.0,
        "retrieval_precision_at_3": 0.0,
    }