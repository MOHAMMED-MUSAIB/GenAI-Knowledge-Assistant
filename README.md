# GenAI Knowledge Assistant

A document question-answering application that extracts text from uploaded
files, retrieves relevant passages using vector search, and can generate
cited answers with an OpenRouter-hosted language model.

## Features

- Upload TXT, PDF, and DOCX documents (maximum 10 MiB per file).
- Extract text, split it into overlapping chunks, and create local CPU embeddings.
- Store chunks and normalized vectors in Qdrant; search with cosine similarity.
- Register and sign in with bearer-token authentication.
- Restrict document listings and retrieval to the authenticated owner.
- Run the web app, API, and Qdrant together with Docker Compose.

## Requirements

For the recommended Docker setup:

- Docker Desktop with Docker Compose v2.
- A `.env` file containing a JWT signing secret of at least 32 bytes.
- An OpenRouter API key only if you want generated answers.

For running the source code outside containers:

- Python 3.12.
- Node.js 22 and npm.
- Docker, to run Qdrant.

## Configuration and secrets

The Compose file reads its settings from `.env`. Create the file from the
template and replace the JWT placeholder with a random secret:

```powershell
Copy-Item .env.example .env
$secret = & .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
(Get-Content .env) -replace '^JWT_SECRET_KEY=.*$', "JWT_SECRET_KEY=$secret" |
    Set-Content .env
```

If the virtual environment has not been created yet, generate the secret with
an installed Python 3.12 interpreter instead:

```powershell
$secret = py -3.12 -c "import secrets; print(secrets.token_urlsafe(32))"
(Get-Content .env) -replace '^JWT_SECRET_KEY=.*$', "JWT_SECRET_KEY=$secret" |
    Set-Content .env
```

Available variables:

| Variable | Required | Purpose |
|---|---|---|
| `JWT_SECRET_KEY` | Yes for Compose; required by the API | HS256 token signing key; use at least 32 bytes and keep it stable between restarts. |
| `OPENROUTER_API_KEY` | No | Enables hosted answer generation. Without it, registration, upload, and document listing work, but `/questions/ask` cannot return an answer. |
| `OPENROUTER_MODEL` | No | OpenRouter model/router name. Defaults to `openrouter/free`. |
| `QDRANT_URL` | No for Compose | Qdrant URL. Compose sets `http://qdrant:6333`; source-run defaults to `http://localhost:6333`. |

Keep `.env` out of source control and do not include secrets in logs or
support tickets. Regenerating `JWT_SECRET_KEY` invalidates existing tokens.

## Start with Docker Compose

From the repository root:

```powershell
docker compose up --build -d
docker compose ps
```

The browser app is available at <http://localhost:8080>. It is bound to
loopback; backend and Qdrant ports are not published to the host. The Compose
stack stores SQLite accounts/document ownership, uploads, Qdrant vectors, and
the downloaded embedding model in separate named volumes.

Check service health:

```powershell
Invoke-RestMethod http://localhost:8080/health
Invoke-RestMethod http://localhost:8080/api/health
Invoke-RestMethod http://localhost:8080/api/health/ready
docker compose ps
```

`/health` is the frontend liveness endpoint. `/api/health` checks backend
liveness. `/api/health/ready` also checks Qdrant and returns HTTP 503 when it
cannot be reached. The interactive API reference is available at
<http://localhost:8080/api/docs>.

Follow logs or stop the services:

```powershell
docker compose logs --tail 100 backend
docker compose logs --tail 100 frontend
docker compose logs --tail 100 qdrant
docker compose down
```

`docker compose down` preserves the named volumes. Do not use
`docker compose down --volumes` unless you intend to delete accounts, uploaded
documents, vectors, and the cached model.

## Run from source (development)

Create and activate a Python environment, then install the backend requirements:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install the frontend dependencies:

```powershell
Set-Location frontend
npm ci
Set-Location ..
```

Start Qdrant using the Compose service:

```powershell
docker compose up -d qdrant
```

In the PowerShell terminal that will run the backend, load `.env` into that
process environment. The API does not automatically load `.env` for its JWT
configuration:

```powershell
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') {
        Set-Item "Env:$($matches[1].Trim())" $matches[2]
    }
}
$env:QDRANT_URL = "http://localhost:6333"
```

Run the API and frontend development server in separate PowerShell terminals
from the repository root. Start the frontend terminal in the `frontend`
directory:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

```powershell
Set-Location frontend
npm run dev
```

Vite proxies `/api` requests to the local backend. Open the URL printed by
Vite. For local API access and its interactive reference, use
<http://127.0.0.1:8000/docs>.

## Tests and checks

Run the complete backend suite, including live-Qdrant retrieval checks, with
coverage:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --cov=src --cov-report=term-missing
```

The live-Qdrant retrieval checks require a running Qdrant instance containing
the expected evaluation document. The document workflow integration test uses
an in-memory Qdrant instance and deterministic embeddings; it exercises
authenticated upload, storage, retrieval, prompt grounding, and citation
mapping without calling an external language-model provider.

Frontend checks (from `frontend/`):

```powershell
npm test -- --run
npm run lint
npm run build
```

The Docker frontend build runs the production build. Container health and logs
can be checked with the commands in [Start with Docker Compose](#start-with-docker-compose).

## Architecture

```text
Browser (React)
    |
    | /api/*; nginx proxy
    v
FastAPI backend
    |-- JWT auth + SQLite account/document ownership
    |-- TXT/PDF/DOCX extraction -> paragraph-aware chunks
    |-- Sentence Transformers all-MiniLM-L6-v2 (CPU, 384 dimensions)
    |-- owner/document-filtered cosine search
    |       |
    |       v
    |     Qdrant (persistent vectors and chunk payloads)
    |
    `-- optional OpenRouter request for cited answer generation
```

The backend stores account hashes and document ownership in
`data/app.sqlite3`; uploaded files are stored under `data/uploads/`. Compose
mounts the application data directory, Qdrant data, and Hugging Face model
cache as separate volumes. Embeddings are generated locally. If answer
generation is enabled, the question and retrieved document context are sent
to the configured OpenRouter model.

## API

In the Docker setup, prefix backend paths with `/api` when calling through
nginx. For example, `/auth/login` becomes `/api/auth/login`. For local backend
development, call the paths directly. Protected routes require
`Authorization: Bearer <access_token>`.

| Method and path | Auth | Description |
|---|---:|---|
| `GET /health` | No | Backend liveness: `{"status":"ok"}`. |
| `GET /health/ready` | No | Readiness including Qdrant; HTTP 503 if Qdrant is unavailable. |
| `POST /auth/register` | No | Create an account and return an access token. |
| `POST /auth/login` | No | Authenticate and return an access token. |
| `POST /documents/upload` | Yes | Upload and index a TXT, PDF, or DOCX file. Send multipart form field `file`. |
| `GET /documents` | Yes | List the signed-in user's documents. |
| `POST /questions/ask` | Yes | Retrieve relevant chunks and generate an answer with source metadata. |

Registration and login accept JSON:

```json
{
  "username": "alice",
  "password": "a-long-password-at-least-12-characters"
}
```

On success, both return an access token:

```json
{
  "access_token": "<bearer-token>",
  "token_type": "bearer",
  "username": "alice"
}
```

Usernames are trimmed and case-insensitive, 3–64 characters; passwords are
12–128 characters. Access tokens are bearer JWTs that expire after one hour.
The frontend stores the token in `sessionStorage`.

Upload with multipart form field `file`; successful responses include the
original filename, generated document ID, extracted character count, and
status message. `GET /documents` returns document IDs and file sizes for the
authenticated user only.

Question requests accept JSON:

```json
{
  "question": "Which embedding model does this project use?",
  "top_k": 3,
  "document_id": "optional-document-id.txt"
}
```

`question` is required (1–2000 characters); `top_k` is 1–10 and defaults to 3.
`document_id` is optional. The result contains `answer` and `sources`; each
source identifies the filename, document ID, and chunk index. Retrieval is
performed as part of this endpoint; without an OpenRouter key, this endpoint
cannot return its answer/source response.

Typical errors include HTTP 401 for missing/invalid authentication or invalid
credentials, 409 for a duplicate username, 400 for an unsupported upload type,
413 for an oversized upload, 422 for invalid request data or unextractable
content, and 503 from readiness when Qdrant is unavailable.

Unexpected backend failures return a generic HTTP 500 response with a request
ID. The same ID is returned in `X-Request-ID` and can be used to find the
sanitized backend log entry. Logs include route templates, status, duration,
and exception type, but exclude request bodies, headers, and query strings.
Nginx access logs also omit query strings and client IP addresses.

## Limitations and security notes

- Only TXT, PDF, and DOCX are accepted; the maximum upload is 10 MiB. Scanned
  PDFs without extractable text may not produce usable content.
- There is no document deletion, password reset, account administration, or
  account-recovery endpoint yet.
- SQLite is suitable for local/single-instance use; multi-replica deployments
  need shared database/storage design and migration/backup procedures.
- The browser app is intended for local use and the Compose port is bound to
  loopback. This setup does not configure TLS or a production ingress.
- Generated answers depend on OpenRouter availability, model behavior, and
  provider policy. Prompts include retrieved document text, so enable hosted
  generation only when sharing that content with the provider is acceptable.
- The current Qdrant Python client (1.19.x) emits a compatibility warning with
  the Compose Qdrant server (1.15.4). Tested basic health, collection, retrieval,
  and persistence operations work, but align client/server versions before
  production use.
- Legacy Qdrant points created before ownership metadata was added are not
  automatically assigned to a user. Re-upload those documents after signing in.
