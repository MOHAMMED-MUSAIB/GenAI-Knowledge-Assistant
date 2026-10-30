
from typing import Any


def recall_at_k(
    retrieved_chunks: list[dict[str, Any]],
    relevant_document_id: str,
    k: int = 3,
) -> float:
    """Return 1.0 if a relevant document appears in the top k results, else 0.0."""
    if k < 1:
        raise ValueError("k must be a positive integer.")

    top_chunks = retrieved_chunks[:k]

    return float(
        any(
            chunk.get("document_id") == relevant_document_id
            for chunk in top_chunks
        )
    )


def test_recall_at_3_when_relevant_document_is_retrieved() -> None:
    retrieved_chunks = [
        {"document_id": "other-document.txt"},
        {"document_id": "b864659d7c124d6bb4ff06245779c653.txt"},
        {"document_id": "another-document.txt"},
    ]

    score = recall_at_k(
        retrieved_chunks,
        "b864659d7c124d6bb4ff06245779c653.txt",
        k=3,
    )

    assert score == 1.0


def test_recall_at_3_when_relevant_document_is_missing() -> None:
    retrieved_chunks = [
        {"document_id": "other-document.txt"},
        {"document_id": "another-document.txt"},
        {"document_id": "third-document.txt"},
    ]

    score = recall_at_k(
        retrieved_chunks,
        "b864659d7c124d6bb4ff06245779c653.txt",
        k=3,
    )

    assert score == 0.0

from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.semantic_search import semantic_search
from src.storage.qdrant_store import COLLECTION_NAME, get_qdrant_client


def test_live_retrieval_finds_embedding_model_document() -> None:
    relevant_document_id = "b864659d7c124d6bb4ff06245779c653.txt"

    embedding_model = EmbeddingModel()
    qdrant_client = get_qdrant_client()

    results = semantic_search(
        query="Which embedding model does this project use?",
        embedding_model=embedding_model,
        qdrant_client=qdrant_client,
        collection_name=COLLECTION_NAME,
        top_k=3,
    )

    score = recall_at_k(
        results,
        relevant_document_id=relevant_document_id,
        k=3,
    )

    print(f"Recall@3: {score}")
    print(
        "Retrieved documents:",
        [chunk.get("filename") for chunk in results],
    )

    assert score == 1.0, (
        "The relevant document was not retrieved in the top three results."
    )

def test_live_retrieval_respects_document_filter() -> None:
    relevant_document_id = "b864659d7c124d6bb4ff06245779c653.txt"

    embedding_model = EmbeddingModel()
    qdrant_client = get_qdrant_client()

    results = semantic_search(
        query="Which embedding model does this project use?",
        embedding_model=embedding_model,
        qdrant_client=qdrant_client,
        collection_name=COLLECTION_NAME,
        top_k=3,
        document_id=relevant_document_id,
    )

    assert results, "No chunks were retrieved for the selected document."

    assert all(
        chunk.get("document_id") == relevant_document_id
        for chunk in results
    ), "Retrieval returned chunks from another document."
