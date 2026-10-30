import logging
import os
from time import perf_counter
from urllib.request import urlopen
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Any

from src.ingestion.txt_reader import read_txt
from src.ingestion.pdf_reader import read_pdf
from src.ingestion.docx_reader import read_docx
from src.ingestion.upload_validator import (
    validate_file_extension,
    validate_file_size,
)
from src.embeddings.embedding_model import EmbeddingModel
from src.storage.qdrant_store import (
    create_collection_if_not_exists,
    get_qdrant_client,
)
from src.rag.pipeline import answer_question
from pydantic import BaseModel, Field, field_validator
from src.chunking.text_chunker import chunk_text
from src.storage.qdrant_store import (
    store_chunk,
    delete_document_chunks,
)
from qdrant_client import QdrantClient

from src.api import auth

UPLOAD_DIR = Path("data/uploads")

app = FastAPI(title="Enterprise GenAI Knowledge Assistant")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

embedding_model: EmbeddingModel | None = None
qdrant_client: QdrantClient | None = None


def get_embedding_model() -> EmbeddingModel:
    """Create and cache the embedding model on first use."""
    global embedding_model
    if embedding_model is None:
        embedding_model = EmbeddingModel()
    return embedding_model


def get_qdrant_client_instance() -> QdrantClient:
    """Create and cache the Qdrant client on first use."""
    global qdrant_client
    if qdrant_client is None:
        qdrant_client = get_qdrant_client()
        create_collection_if_not_exists(qdrant_client)
    return qdrant_client


@app.middleware("http")
async def log_request_diagnostics(request: Request, call_next: Any) -> Response:
    """Log request metadata without logging query strings or user content."""
    request_id = uuid4().hex
    request.state.request_id = request_id
    started_at = perf_counter()
    route = request.scope.get("route")
    route_template = getattr(route, "path", "<unmatched>")
    try:
        response = await call_next(request)
    except Exception as exc:
        elapsed_ms = (perf_counter() - started_at) * 1000
        route = request.scope.get("route")
        route_template = getattr(route, "path", "<unmatched>")
        logger.error(
            "event=unhandled_exception request_id=%s method=%s route=%s "
            "error_type=%s duration_ms=%.2f",
            request_id,
            request.method,
            route_template,
            type(exc).__name__,
            elapsed_ms,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error.", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )
    elapsed_ms = (perf_counter() - started_at) * 1000
    route = request.scope.get("route")
    route_template = getattr(route, "path", "<unmatched>")
    status_code = response.status_code
    if status_code >= 500:
        log_method = logger.error
    elif status_code >= 400:
        log_method = logger.warning
    else:
        log_method = logger.info
    log_method(
        "event=request_complete request_id=%s method=%s route=%s "
        "status_code=%d duration_ms=%.2f",
        request_id,
        request.method,
        route_template,
        status_code,
        elapsed_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
class QuestionRequest(BaseModel):
    """Validate a question submitted to the RAG API."""
    document_id: str | None = Field(default=None, min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class AuthRequest(BaseModel):
    """Validate a username and password submitted to the auth API."""

    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=12, max_length=128)

    @field_validator("username")
    @classmethod
    def trim_username(cls, value: str) -> str:
        normalized = value.strip()
        if len(normalized) < 3:
            raise ValueError("Username must contain at least 3 characters.")
        return normalized


class AuthResponse(BaseModel):
    """Return a short-lived access token."""

    access_token: str
    token_type: str = "bearer"
    username: str


@app.post("/auth/register", response_model=AuthResponse, status_code=201)
def register(request: AuthRequest) -> AuthResponse:
    """Create a user account and issue its access token."""
    try:
        auth.validate_signing_key()
        user = auth.register_user(request.username, request.password)
        return AuthResponse(
            access_token=auth.create_access_token(user),
            username=user.username,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/auth/login", response_model=AuthResponse)
def login(request: AuthRequest) -> AuthResponse:
    """Authenticate a user and issue its access token."""
    user = auth.authenticate_user(request.username, request.password)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return AuthResponse(
        access_token=auth.create_access_token(user),
        username=user.username,
    )


@app.post("/questions/ask")
def ask_question(
    request: QuestionRequest,
    current_user: auth.UserIdentity = Depends(auth.get_current_user),
) -> dict[str, Any]:
    """Answer a question using uploaded documents and return sources."""
    try:
        return answer_question(
            question=request.question,
            embedding_model=get_embedding_model(),
            qdrant_client=get_qdrant_client_instance(),
            top_k=request.top_k,
            document_id=request.document_id,
            owner_id=current_user.user_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""
    return {"status": "ok"}


@app.get("/health/ready")
def readiness_check() -> JSONResponse:
    """Check that the API can reach its configured Qdrant service."""
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
    try:
        with urlopen(f"{qdrant_url}/healthz", timeout=2):
            return JSONResponse(
                status_code=200,
                content={"status": "ok", "components": {"qdrant": "ok"}},
            )
    except Exception as exc:
        logger.error(
            "event=readiness_failed component=qdrant error_type=%s",
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "components": {"qdrant": "unavailable"},
            },
        )


@app.get("/documents")
def list_documents(
    current_user: auth.UserIdentity = Depends(auth.get_current_user),
) -> list[dict[str, str | int]]:
    """Return only the current user's uploaded documents."""
    return [
        document
        for document in auth.list_user_documents(current_user.user_id)
        if (UPLOAD_DIR / str(document["document_id"])).is_file()
    ]


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: auth.UserIdentity = Depends(auth.get_current_user),
) -> dict[str, str | int]:
    """Validate and save an uploaded document."""
    destination: Path | None = None
    stored_filename: str | None = None
    storage_client: QdrantClient | None = None

    try:
        validate_file_extension(file.filename or "")

        content = bytearray()

        while True:
            remaining = MAX_FILE_SIZE_BYTES + 1 - len(content)
            chunk = await file.read(min(64 * 1024, remaining))

            if not chunk:
                break

            content.extend(chunk)

            if len(content) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Uploaded file exceeds the 10 MB size limit.",
                )

        validate_file_size(len(content), MAX_FILE_SIZE_BYTES)

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        extension = Path(file.filename or "").suffix.lower()
        stored_filename = f"{uuid4().hex}{extension}"
        destination = UPLOAD_DIR / stored_filename
        destination.write_bytes(bytes(content))

        if extension == ".txt":
            extracted_text = read_txt(str(destination))
        elif extension == ".pdf":
            extracted_text = read_pdf(destination)
        else:
            extracted_text = read_docx(destination)

        if not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from the uploaded document.",
            )
        chunks = chunk_text(
            extracted_text,
            document_id=stored_filename,
            filename=file.filename or "",
        )
        for chunk in chunks:
            chunk["owner_id"] = current_user.user_id

        chunk_texts: list[str] = [str(chunk["text"]) for chunk in chunks]
        embeddings = get_embedding_model().embed_texts(chunk_texts)
        storage_client = get_qdrant_client_instance()

        for chunk, embedding in zip(chunks, embeddings):
            store_chunk(
                client=storage_client,
                chunk=chunk,
                embedding=embedding,
            )
        auth.add_document_owner(
            document_id=stored_filename,
            owner_id=current_user.user_id,
            size_bytes=len(content),
        )
        return {
            "filename": file.filename or "",
            "document_id": stored_filename,
            "message": "File uploaded successfully.",
            "extracted_characters": len(extracted_text),
        }

    except ValueError as exc:
        if storage_client is not None and stored_filename is not None:
            try:
                delete_document_chunks(storage_client, stored_filename)
            except Exception as cleanup_exc:
                logger.error(
                    "event=cleanup_failed component=qdrant error_type=%s",
                    type(cleanup_exc).__name__,
                )

        if destination is not None:
            destination.unlink(missing_ok=True)
        if stored_filename is not None and storage_client is not None:
            try:
                auth.remove_document_owner(stored_filename, current_user.user_id)
            except Exception as cleanup_exc:
                logger.error(
                    "event=cleanup_failed component=ownership error_type=%s",
                    type(cleanup_exc).__name__,
                )

        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception:
        if storage_client is not None and stored_filename is not None:
            try:
                delete_document_chunks(storage_client, stored_filename)
            except Exception as cleanup_exc:
                logger.error(
                    "event=cleanup_failed component=qdrant error_type=%s",
                    type(cleanup_exc).__name__,
                )

        if destination is not None:
            destination.unlink(missing_ok=True)
        if stored_filename is not None and storage_client is not None:
            try:
                auth.remove_document_owner(stored_filename, current_user.user_id)
            except Exception as cleanup_exc:
                logger.error(
                    "event=cleanup_failed component=ownership error_type=%s",
                    type(cleanup_exc).__name__,
                )

        raise

    finally:
        await file.close()
