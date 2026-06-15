import logging
import sys
import uuid
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

from ingestion.ingest import ingest_source
from retrieval.retriever import query, stream_query, get_session_sources
from clients.embeddings import embedding_client
from clients.llm import llm_client
from models.query_request import QueryRequest
from models.query_response import QueryResponse
from models.quiz_request import QuizRequest
from models.url_request import URLRequest

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

log = logging.getLogger(__name__)

app_state = {}

def ensure_qdrant_indexes():
    """
    WHY HERE NOT MANUAL:
    Index must exist before any filtered query runs.
    Running at startup = auto-created on every fresh Azure deployment.
    Safe to run multiple times — Qdrant ignores if already exists.
    """
    try:
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )


        client.create_payload_index(
            collection_name=os.getenv("QDRANT_COLLECTION"),
            field_name="metadata.session_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        log.info("Qdrant index ensured: metadata.session_id")
    except Exception as e:
        # WHY NOT RAISE:
        # If index already exists Qdrant returns error
        # We don't want startup to fail for this
        log.warning(f"Index creation skipped (may already exist): {e}")

# ══════════════════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — initializing clients...")
    try:
        validate_env()
        app_state["embeddings"] = embedding_client()
        app_state["llm"] = llm_client()
        ensure_qdrant_indexes()
        log.info("Clients ready.")
    except Exception as e:
        log.critical(f"Startup failed: {e}")
        raise
    yield
    app_state.clear()
    log.info("Shutdown complete.")


app = FastAPI(
    title="Multi-Source RAG API",
    description="Samasocial AI Learning Assistant",
    version="2.0.0",
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════════════
# CORS
# ══════════════════════════════════════════════════════════════════════
def get_allowed_origins() -> list[str]:
    configured = os.getenv("FRONTEND_ORIGINS", "")
    origins = [o.strip() for o in configured.split(",") if o.strip()]
    return origins or [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_origin_regex=os.getenv(
        "FRONTEND_ORIGIN_REGEX", r"https://.*\.onrender\.com"
    ),
    allow_credentials=False,
    allow_methods=["*"],      # ← CHANGE TO * — allows all methods including OPTIONS
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════
@app.get("/health")
async def health():
    issues = []
    if not app_state.get("embeddings"):
        issues.append("embeddings client not ready")
    if not app_state.get("llm"):
        issues.append("llm client not ready")
    if issues:
        raise HTTPException(
            status_code=503,
            detail={"status": "unhealthy", "issues": issues}
        )
    return {"status": "healthy", "clients_ready": True}


# ══════════════════════════════════════════════════════════════════════
# INGEST — FILE (PDF + PPTX)
# ══════════════════════════════════════════════════════════════════════
@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...),session_id: str = ""):
    """
    Upload PDF or PPTX.
    Returns summary immediately after processing.
    ASSIGNMENT: Source badge with summary shown after upload.
    """
    session_id = session_id or str(uuid.uuid4())
    filename = file.filename

    if filename.endswith(".pdf"):
        source_type = "pdf"
    elif filename.endswith(".pptx"):
        source_type = "pptx"
    else:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and PPTX supported."
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file.")

        result = ingest_source(
            source_type=source_type,
            embeddings=app_state["embeddings"],
            file_bytes=file_bytes,
            filename=filename,
            session_id=session_id,
        )
        result["session_id"] = session_id
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error(f"File ingest error: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed.")


# ══════════════════════════════════════════════════════════════════════
# INGEST — URL (YouTube + Webpage)
# ══════════════════════════════════════════════════════════════════════
@app.post("/ingest/url")
async def ingest_url(request: URLRequest, session_id: str = ""):
    """
    Ingest YouTube URL or any webpage URL.
    ASSIGNMENT: Multi-source — mix PDF + YouTube + URL in one session.
    """
    session_id = session_id or request.session_id or str(uuid.uuid4())
    if request.source_type not in ["youtube", "url"]:
        raise HTTPException(
            status_code=400,
            detail="source_type must be 'youtube' or 'url'"
        )

    try:
        result = ingest_source(
            source_type=request.source_type,
            embeddings=app_state["embeddings"],
            url=request.url,
            session_id=session_id, 
        )
        result["session_id"] = session_id
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error(f"URL ingest error: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed.")


# ══════════════════════════════════════════════════════════════════════
# QUERY — with streaming support
# ══════════════════════════════════════════════════════════════════════
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Query with session memory + source citations.
    Supports both streaming and non-streaming.
    ASSIGNMENT: Follow-up questions, citations, streaming responses.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())

    # ── Streaming response ────────────────────────────────────────────
    # WHY SSE NOT WEBSOCKET:
    # SSE = one-way server→client stream — perfect for chat responses
    # WebSocket = bidirectional — overkill for text streaming
    # SSE works with fetch() in browser, no extra library needed
    if request.stream:
        async def event_generator():
            try:
                async for chunk in stream_query(
                    question=request.question,
                    embeddings=app_state["embeddings"],
                    llm=app_state["llm"],
                    session_id=session_id,
                ):
                    yield {"data": chunk}
            except Exception as e:
                log.error(f"Stream error: {e}")
                yield {"data": "[ERROR]"}

        return EventSourceResponse(event_generator())

    # ── Non-streaming response ────────────────────────────────────────
    try:
        result = query(
            question=request.question,
            embeddings=app_state["embeddings"],
            llm=app_state["llm"],
            session_id=session_id,
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            citations=result["citations"],
            session_id=session_id,
        )
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail="Query failed.")


# ══════════════════════════════════════════════════════════════════════
# SESSION — clear history
# ══════════════════════════════════════════════════════════════════════
@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear session = wipe entire collection + all memory.
    Single user app — clear means complete fresh start.
    """
    from retrieval.retriever import clear_session_data
    try:
        result = clear_session_data(session_id)
        return {
            "status": "cleared",
            "points_deleted": result["points_deleted"],
            "session_id": session_id,
        }
    except Exception as e:
        log.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear session.")


@app.get("/session/{session_id}/sources")
async def get_sources(session_id: str):
    """
    Get all sources ingested in this session.
    ASSIGNMENT: Source badges showing what has been loaded.
    """
    sources = get_session_sources(session_id)
    return {"session_id": session_id, "sources": sources}


# ══════════════════════════════════════════════════════════════════════
# QUIZ MODE — ASSIGNMENT BONUS
# ══════════════════════════════════════════════════════════════════════
@app.post("/quiz")
async def generate_quiz(request: QuizRequest):
    """
    Auto-generate quiz questions from loaded content.
    ASSIGNMENT BONUS: Quiz me mode.
    """
    from retrieval.retriever import generate_quiz_questions
    try:
        questions = await generate_quiz_questions(
            session_id=request.session_id,
            embeddings=app_state["embeddings"],
            llm=app_state["llm"],
            num_questions=request.num_questions,
        )
        return {"session_id": request.session_id, "questions": questions}
    except Exception as e:
        log.error(f"Quiz generation error: {e}")
        raise HTTPException(status_code=500, detail="Quiz generation failed.")


# ══════════════════════════════════════════════════════════════════════
# ENV VALIDATION
# ══════════════════════════════════════════════════════════════════════
def validate_env():
    required = [
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION", "AZURE_EMBEDDING_DEPLOYMENT",
        "AZURE_CHAT_DEPLOYMENT", "QDRANT_URL",
        "QDRANT_API_KEY", "QDRANT_COLLECTION",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Missing env vars: {missing}")
