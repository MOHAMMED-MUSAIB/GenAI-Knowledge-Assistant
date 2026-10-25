# GenAI-Knowledge-Assistant

A local-first Retrieval-Augmented Generation (RAG) application for ingesting documents, generating embeddings, and retrieving relevant content using semantic search.

## Technology Stack

- **Language:** Python 3.12
- **Vector database:** Qdrant
- **Embedding model:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector dimensions:** 384
- **Similarity metric:** Cosine similarity
- **Testing:** pytest
- **Containerization:** Docker

## Current Development Status

### Phase 1 — Document Ingestion

- TXT, PDF, and DOCX text extraction
- File extension and size validation
- Error handling for invalid and unsupported files

### Phase 2 — Text Chunking

- Text splitting with configurable chunk size and overlap
- Sequential chunk indexes and word counts
- Document metadata preservation

### Phase 3 — Embeddings

- Text embedding generation
- 384-dimensional embedding vectors
- Input and vector validation

### Phase 4 — Vector Storage and Semantic Retrieval

- Qdrant client connection
- Idempotent collection creation
- Vector storage and upsert support
- Deterministic point IDs using UUID5
- Chunk payload retrieval by point ID
- Semantic search with ranked results
- Optional document-level filtering
- Persistent storage using the `qdrant-storage` Docker volume

## Verification Results

- **Automated regression tests:** 88 passed
- **Real Qdrant semantic retrieval:** Passed
- **Real Qdrant document-level filtering:** Passed
- **Persistence across container restart:** Passed
- **Temporary persistence test point:** Removed after verification

The automated test suite and the live-Qdrant integration checks were run separately.

## Run Automated Tests

From the project root, with the virtual environment activated:

```powershell
python -m pytest -v --basetemp=.pytest-tmp
```

## Local Qdrant

The development container is named `qdrant-local` and exposes:

- HTTP API: `http://localhost:6333`
- Dashboard: `http://localhost:6333/dashboard`

Start the existing container:

```powershell
docker start qdrant-local
```

Check its status:

```powershell
docker ps --filter "name=qdrant-local"
```

Qdrant data is stored in the Docker named volume `qdrant-storage`, mounted at `/qdrant/storage`. Avoid deleting this volume unless intentionally resetting the database.

## Next Development Steps

- Review and document the completed Phase 4 implementation
- Run integration tests as part of a repeatable verification workflow
- Continue building the RAG application backend
- Add document upload, ingestion orchestration, and question-answering workflows
- Evaluate retrieval quality, citation accuracy, latency, and cost as the application grows
