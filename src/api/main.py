from fastapi import FastAPI, File, HTTPException, UploadFile
from src.ingestion.txt_reader import read_txt
from src.ingestion.pdf_reader import read_pdf
from src.ingestion.docx_reader import read_docx
from src.ingestion.upload_validator import (
    validate_file_extension,
    validate_file_size,
)
from pathlib import Path
from uuid import uuid4
from typing import Any
from src.embeddings.embedding_model import EmbeddingModel
from src.storage.qdrant_store import (
    create_collection_if_not_exists,
    get_qdrant_client,
)
from src.rag.pipeline import answer_question
from pydantic import BaseModel, Field
UPLOAD_DIR = Path("data/uploads")

app = FastAPI(title="Enterprise GenAI Knowledge Assistant")

embedding_model = EmbeddingModel()
qdrant_client = get_qdrant_client()
create_collection_if_not_exists(qdrant_client)

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
class QuestionRequest(BaseModel):
    """Validate a question submitted to the RAG API."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)
@app.post("/questions/ask")
def ask_question(request: QuestionRequest) -> dict[str, Any]:
    """Answer a question using uploaded documents and return sources."""
    try:
        return answer_question(
            question=request.question,
            embedding_model=embedding_model,
            qdrant_client=qdrant_client,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""
    return {"status": "ok"}


@app.get("/documents")
def list_documents() -> list[dict[str, str | int]]:
    """Return the uploaded documents currently stored on disk."""
    if not UPLOAD_DIR.exists():
        return []

    documents: list[dict[str, str | int]] = []

    for path in sorted(UPLOAD_DIR.iterdir(), key=lambda path: path.name.lower()):
        if path.is_file() and path.suffix.lower() in {".txt", ".pdf", ".docx"}:
            documents.append(
                {
                    "document_id": path.name,
                    "size_bytes": path.stat().st_size,
                }
            )

    return documents


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, str | int]:
    """Validate and save an uploaded document."""
    destination: Path | None = None

    try:
        validate_file_extension(file.filename or "")

        content = await file.read()
        validate_file_size(len(content), MAX_FILE_SIZE_BYTES)

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        extension = Path(file.filename or "").suffix.lower()
        stored_filename = f"{uuid4().hex}{extension}"
        destination = UPLOAD_DIR / stored_filename
        destination.write_bytes(content)

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

        return {
            "filename": file.filename or "",
            "document_id": stored_filename,
            "message": "File uploaded successfully.",
            "extracted_characters": len(extracted_text),
        }

    except ValueError as exc:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except Exception:
        if destination is not None:
            destination.unlink(missing_ok=True)
        raise

    finally:
        await file.close()
