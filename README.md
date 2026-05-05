# PPU University Chatbot

An NLP-powered university assistant for Palestine Polytechnic University (PPU), built as part of an NLP course project. The system answers questions about the university's fees, schedules, registration, staff, and more. using a full Retrieval-Augmented Generation (RAG) pipeline backed by fine-tuned BERT for intent classification and ChromaDB for semantic search.

---

## System Architecture

The project is split into two independently deployable services that communicate over a shared internal API.

```
[Web UI]  ──POST /chat/message  ->  [Gateway :8000] POST /process ->  [NLP Engine :8001]
                                          │                                      │
                                     PostgreSQL                             ChromaDB + HF models
```

### Gateway (port 8000)

Public-facing FastAPI service. Handles session management, input validation, JWT auth, rate limiting, chat history persistence (PostgreSQL), and the admin panel. Forwards every user message to the NLP engine and returns the structured reply to the frontend.

### NLP Engine (port 8001)

Internal FastAPI service, never called directly by clients. Runs the full NLP pipeline:

1. **Preprocessing** — tokenization, normalization, language detection (spaCy + NLTK)
2. **Intent classification** — fine-tuned `bert-base-uncased` via HuggingFace Transformers
3. **NER / entity extraction** — slot filling (course name, student ID, dates)
4. **RAG pipeline** — retrieve top-K chunks from ChromaDB → re-rank by cosine similarity → inject as context → generate grounded reply (HuggingFace Inference API)
5. **Dialogue management** — state machine for well-defined flows + LLM fallback router when confidence < threshold
6. **Evaluation** — intent accuracy, NER F1, retrieval Precision@k

---

## Repository Structure

```
ppu-chatbot/
│
├── shared/                        # ← Both students read this, neither changes it unilaterally
│   ├── schemas/
│   │   └── message.py             # ProcessRequest / ProcessResponse / FeedbackRequest (the contract)
│   └── config/
│       └── settings.py            # Central config loaded from .env
│
├── nlp_engine/                    # ← Student A owns this
│   ├── preprocessing/
│   │   ├── tokenizer.py
│   │   └── normalizer.py
│   ├── intent/
│   │   ├── classifier.py          # BERT intent classifier
│   │   ├── fine_tune.py           # Fine-tuning script
│   │   └── labels.json            # Intent label list
│   ├── ner/
│   │   ├── extractor.py           # Named entity extraction
│   │   └── entities.py            # Entity type definitions
│   ├── rag/
│   │   ├── retriever.py           # ChromaDB semantic search
│   │   ├── reranker.py            # Cosine similarity re-ranking
│   │   └── generator.py          # Context injection + generation
│   ├── knowledge_base/
│   │   ├── ingest.py              # PDF → chunks → embeddings → ChromaDB
│   │   ├── embed.py               # Sentence embedding logic
│   │   └── chroma_store.py        # ChromaDB client wrapper
│   ├── dialogue/
│   │   ├── state_machine.py       # Deterministic dialogue flows
│   │   └── context_manager.py     # Multi-turn conversation history
│   ├── evaluation/
│   │   ├── metrics.py             # Accuracy, F1, Precision@k
│   │   └── eval_runner.py         # Runs the eval suite
│   └── nlp_server.py              # FastAPI app, port 8001
│
├── gateway/                       # ← Student B owns this
│   ├── api/
│   │   ├── main.py                # FastAPI app, port 8000
│   │   ├── routes/
│   │   │   ├── chat.py            # /chat/* and /sessions/* endpoints
│   │   │   ├── admin.py           # /admin/* endpoints
│   │   │   └── health.py          # /health
│   │   ├── middleware/
│   │   │   ├── rate_limit.py
│   │   │   ├── session.py
│   │   │   └── validator.py
│   │   └── auth/
│   │       └── jwt_handler.py
│   ├── storage/
│   │   ├── db.py                  # SQLAlchemy engine setup
│   │   ├── models.py              # ORM models (Session, Turn, Feedback)
│   │   └── chat_repo.py           # DB read/write helpers
│   ├── frontend/
│   │   ├── index.html             # Web chat UI
│   │   ├── chat.js
│   │   ├── style.css
│   │   ├── admin.html             # Admin panel (upload PDFs, view stats)
│   │   └── admin.js
│   └── channels/
│       ├── telegram_adapter.py    # Future: Telegram bot adapter
│       └── gmail_adapter.py       # Future: Gmail integration adapter
│
├── tests/
│   ├── test_nlp.py
│   ├── test_api.py
│   ├── test_rag.py
│   └── fixtures/
│       └── sample_queries.json    # Ground-truth intent labels for evaluation
│
├── data/
│   ├── raw/                       # Source PDFs (not tracked by git)
│   ├── processed/                 # Chunked text (not tracked by git)
│   └── models/                    # Fine-tuned weights (not tracked by git)
│
├── .env.example                   # Copy to .env and fill in values
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── README.md
```

---

## API Contract

This is the agreed JSON schema between the two services. **Neither student changes field names without notifying the other.**

### Gateway → NLP Engine  `POST /process`

Request:

```json
{
  "session_id": "uuid-string",
  "message":    "What are the CS tuition fees?",
  "history":    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "channel":    "web"
}
```

Response:

```json
{
  "reply":      "The annual tuition fee for Computer Science is...",
  "intent":     "faq_fees",
  "confidence": 0.94,
  "sources":    [{"text": "...", "doc_name": "student_handbook.pdf", "page": 12}]
}
```

If `confidence` is below the threshold in settings (default `0.55`), the NLP engine returns a graceful fallback reply and the gateway displays an uncertainty warning in the UI.

---

## Gateway Endpoints — Port 8000 (public)

| Method     | Path                           | Description                                      |
| ---------- | ------------------------------ | ------------------------------------------------ |
| `POST`   | `/chat/message`              | Send a message; returns reply + intent + sources |
| `GET`    | `/chat/history/{session_id}` | Retrieve full turn history for a session         |
| `POST`   | `/chat/feedback`             | Submit thumbs up / down rating for a reply       |
| `POST`   | `/sessions/new`              | Create a new session; returns `session_id`     |
| `DELETE` | `/sessions/{session_id}`     | Clear session context                            |
| `POST`   | `/admin/upload`              | Upload a PDF → triggers RAG ingestion           |
| `GET`    | `/admin/documents`           | List indexed documents in the knowledge base     |
| `GET`    | `/admin/stats`               | Query count, intent breakdown, feedback ratio    |
| `GET`    | `/health`                    | Liveness check                                   |

## NLP Engine Endpoints — Port 8001 (internal only)

| Method   | Path          | Description                                          |
| -------- | ------------- | ---------------------------------------------------- |
| `POST` | `/process`  | Full pipeline: message → reply + intent + sources   |
| `POST` | `/ingest`   | PDF bytes → chunk → embed → index in ChromaDB     |
| `POST` | `/classify` | Intent + NER only, no generation (for unit testing)  |
| `POST` | `/evaluate` | Run eval suite; returns intent accuracy, NER F1, P@k |
| `GET`  | `/health`   | Liveness check                                       |

---

## Git Workflow

```
main              ← always deployable; merge only at sync points
student-a         ← NLP engine work (Student A's branch)
student-b         ← Gateway / frontend work (Student B's branch)
```

**Sync points** (both students merge their branch to `main`):

- Hour 3 — after initial stubs are in place and the /process contract is verified end-to-end
- Hour 7 — after RAG retrieval is wired and frontend is displaying real replies
- End of day 1 — stable demo-ready version on `main`

**Day 1 priority:** Student B should build a mock stub for `/process` that returns fake data immediately, so the full frontend → gateway → NLP round-trip can be tested before Student A's pipeline is ready.

---

## Tech Stack

| Component           | Technology                                       |
| ------------------- | ------------------------------------------------ |
| API framework       | FastAPI + Uvicorn                                |
| NLP preprocessing   | spaCy + NLTK                                     |
| Intent classifier   | `bert-base-uncased` fine-tuned via HuggingFace |
| Embedding model     | `sentence-transformers/all-MiniLM-L6-v2`       |
| Generation          | HuggingFace Inference API (Mistral-7B)           |
| Vector database     | ChromaDB                                         |
| Chat history DB     | SQLite (dev) → PostgreSQL (prod)                |
| Session cache       | In-memory (dev) → Redis (future)                |
| Web frontend        | Vanilla JS + WebSocket                           |
| PDF ingestion       | pypdf                                            |
| Deployment          | Docker Compose → Railway / Render               |
| Monitoring (future) | Prometheus + Grafana                             |

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ppu-chatbot.git
cd ppu-chatbot

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Configure environment
cp .env.example .env
# Edit .env and fill in HF_INFERENCE_API_KEY and JWT_SECRET

# 5. Run the NLP engine (Student A's service)
uvicorn nlp_engine.nlp_server:app --port 8001 --reload

# 6. Run the gateway (Student B's service) — in a separate terminal
uvicorn gateway.api.main:app --port 8000 --reload

# 7. Open the chat UI
# Visit http://localhost:8000
```

---

## Knowledge Base

The chatbot uses a **dual knowledge base**:

- **Semantic KB** (ChromaDB) — unstructured documents (student handbook PDF, department pages, FAQs). Queried via embedding similarity. Used for open-ended questions.
- **Structured KB** (future) — deterministic facts (exact fees, exam dates). Queried via intent-matched lookup. Returns exact answers with no hallucination risk.

To add documents to the semantic KB, use the admin panel at `/admin` or call `POST /admin/upload` with a PDF file.

---

## Course Concepts Demonstrated

| Module                       | Where used in this project                             |
| ---------------------------- | ------------------------------------------------------ |
| Tokenization & normalization | `nlp_engine/preprocessing/`                          |
| Intent classification        | Fine-tuned BERT in `intent/classifier.py`            |
| Named entity recognition     | `ner/extractor.py`                                   |
| Semantic similarity          | `rag/retriever.py` + `rag/reranker.py`             |
| Language generation          | `rag/generator.py` (RAG pattern)                     |
| Multi-turn dialogue          | `dialogue/context_manager.py`                        |
| Evaluation metrics           | `evaluation/metrics.py` — accuracy, F1, Precision@k |
| Deployment                   | Docker Compose, Railway                                |
