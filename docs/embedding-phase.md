
# Phase 3: Embedding Generation and Semantic Search

## 1. Purpose

Convert document chunks into numerical vectors and retrieve
relevant chunks for a user's query using semantic similarity.

## 2. Embedding Model

- Model: sentence-transformers/all-MiniLM-L6-v2
- Library: Sentence Transformers
- Execution: CPU
- Embedding dimension: 384
- Output data type: float32
- Normalization: Enabled

The model maps text into fixed-length vectors that represent
semantic features learned during training.

## 3. Implementation

### Embedding module

File: `src/embeddings/embedding_model.py`

Methods:

- `embed_text(text)`: Embeds one text string.
- `embed_texts(texts)`: Embeds multiple strings in a batch.

Both methods reject empty text inputs.

### Semantic search module

File: `src/retrieval/semantic_search.py`

The `semantic_search()` function:

1. Validates the query, chunks, and embeddings.
2. Converts the query into an embedding.
3. Calculates cosine similarity between the query and each chunk.
4. Ranks chunks from highest to lowest similarity.
5. Returns up to `top_k` results with similarity scores.

The current implementation searches in-memory vectors.
Qdrant integration is a future step.

## 4. Similarity

Cosine similarity compares the direction of two vectors.

For normalized vectors, the dot product equals cosine similarity.

A higher score indicates greater vector similarity, but it
does not guarantee that a passage is correct or answers a question.

Initial experiment:

- Related sentences: 0.8635
- Unrelated sentences: 0.0366

These values come from a small experiment and are not universal
thresholds.

## 5. Setup

Activate the project's virtual environment, then install
dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 6. Run Tests

Run the complete project test suite:

```powershell
python -m pytest tests -q --basetemp=.pytest-tmp
```

Latest recorded result:

- 66 tests passed
- Duration: 49.81 seconds

## 7. Limitations and Next Steps

- Semantic search currently uses in-memory NumPy vectors.
- Retrieval quality has only been checked on a small dataset.
- Large document collections should use a vector database.
- Next integration step: store chunk embeddings and metadata in Qdrant.
