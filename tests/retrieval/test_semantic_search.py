
from typing import Any

import numpy as np
import pytest

from src.retrieval.semantic_search import semantic_search


class FakeEmbeddingModel:
    """Return a predictable vector for unit tests."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector
        self.received_text: str | None = None

    def embed_text(self, text: str) -> np.ndarray:
        self.received_text = text
        return np.asarray(self.vector, dtype=np.float32)


class FakePoint:
    """Represent one Qdrant search result."""

    def __init__(
        self,
        point_id: int,
        payload: dict[str, Any],
        score: float,
    ) -> None:
        self.id = point_id
        self.payload = payload
        self.score = score


class FakeResponse:
    """Represent a Qdrant query response."""

    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points


class FakeQdrantClient:
    """Capture Qdrant query arguments and return fake points."""

    def __init__(self, points: list[FakePoint]) -> None:
        self.points = points
        self.call_arguments: dict[str, Any] | None = None

   
    def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
        with_vectors: bool,
        query_filter: Any = None,
    ) -> FakeResponse:
        self.call_arguments = {
            "collection_name": collection_name,
            "query": query,
            "limit": limit,
            "with_payload": with_payload,
            "with_vectors": with_vectors,
            "query_filter": query_filter,
        }

        return FakeResponse(self.points[:limit])

@pytest.fixture
def search_dependencies() -> tuple[
    FakeEmbeddingModel,
    FakeQdrantClient,
]:
    embedding_model = FakeEmbeddingModel([1.0, 0.0, 0.0])

    qdrant_client = FakeQdrantClient(
        [
            FakePoint(
                point_id=101,
                payload={
                    "text": "Annual leave policy allows paid leave.",
                    "filename": "leave_policy.pdf",
                    "chunk_index": 0,
                },
                score=0.95,
            ),
            FakePoint(
                point_id=102,
                payload={
                    "text": "Sick leave requires notifying the manager.",
                    "filename": "sick_leave.pdf",
                    "chunk_index": 1,
                },
                score=0.82,
            ),
            FakePoint(
                point_id=103,
                payload={
                    "text": "The solar system contains eight planets.",
                    "filename": "science.txt",
                    "chunk_index": 0,
                },
                score=0.20,
            ),
        ]
    )

    return embedding_model, qdrant_client


def test_returns_ranked_chunks(
    search_dependencies: tuple[
        FakeEmbeddingModel,
        FakeQdrantClient,
    ],
) -> None:
    embedding_model, qdrant_client = search_dependencies

    results = semantic_search(
        query="What is the annual leave policy?",
        embedding_model=embedding_model,  # type: ignore[arg-type]
        qdrant_client=qdrant_client,  # type: ignore[arg-type]
        collection_name="document_chunks",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["text"] == (
        "Annual leave policy allows paid leave."
    )
    assert results[0]["similarity"] == pytest.approx(0.95)
    assert results[0]["id"] == "101"
    assert results[0]["filename"] == "leave_policy.pdf"
    assert results[1]["text"] == (
        "Sick leave requires notifying the manager."
    )


def test_sends_query_embedding_and_options_to_qdrant(
    search_dependencies: tuple[
        FakeEmbeddingModel,
        FakeQdrantClient,
    ],
) -> None:
    embedding_model, qdrant_client = search_dependencies

    semantic_search(
        query="Explain annual leave.",
        embedding_model=embedding_model,  # type: ignore[arg-type]
        qdrant_client=qdrant_client,  # type: ignore[arg-type]
        collection_name="document_chunks",
        top_k=3,
    )

    assert embedding_model.received_text == "Explain annual leave."

    assert qdrant_client.call_arguments == {
        "collection_name": "document_chunks",
        "query": [1.0, 0.0, 0.0],
        "limit": 3,
        "with_payload": True,
        "with_vectors": False,
        "query_filter": None,
    }

def test_filters_search_by_document_id(
    search_dependencies: tuple[
        FakeEmbeddingModel,
        FakeQdrantClient,
    ],
) -> None:
    embedding_model, qdrant_client = search_dependencies

    semantic_search(
        query="What is the leave policy?",
        embedding_model=embedding_model,  # type: ignore[arg-type]
        qdrant_client=qdrant_client,  # type: ignore[arg-type]
        collection_name="document_chunks",
        document_id="doc-123",
    )

    assert qdrant_client.call_arguments is not None

    query_filter = qdrant_client.call_arguments["query_filter"]

    assert query_filter is not None
    assert len(query_filter.must) == 1
    assert query_filter.must[0].key == "document_id"
    assert query_filter.must[0].match.value == "doc-123"


def test_returns_empty_list_when_no_points_match() -> None:
    embedding_model = FakeEmbeddingModel([1.0, 0.0])
    qdrant_client = FakeQdrantClient([])

    results = semantic_search(
        query="Find a document.",
        embedding_model=embedding_model,  # type: ignore[arg-type]
        qdrant_client=qdrant_client,  # type: ignore[arg-type]
        collection_name="document_chunks",
    )

    assert results == []


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
    ],
)
def test_rejects_empty_query(query: str) -> None:
    with pytest.raises(ValueError, match="Query must not be empty"):
        semantic_search(
            query=query,
            embedding_model=FakeEmbeddingModel([1.0, 0.0]),  # type: ignore[arg-type]
            qdrant_client=FakeQdrantClient([]),  # type: ignore[arg-type]
            collection_name="document_chunks",
        )


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_rejects_invalid_top_k(top_k: Any) -> None:
    with pytest.raises(ValueError, match="top_k"):
        semantic_search(
            query="Find a document.",
            embedding_model=FakeEmbeddingModel([1.0, 0.0]),  # type: ignore[arg-type]
            qdrant_client=FakeQdrantClient([]),  # type: ignore[arg-type]
            collection_name="document_chunks",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "vector",
    [
        [],
        [[1.0, 0.0]],
        [float("nan"), 0.0],
        [float("inf"), 0.0],
        [0.0, 0.0],
    ],
)
def test_rejects_invalid_query_embedding(
    vector: list[Any],
) -> None:
    with pytest.raises(ValueError, match="embedding"):
        semantic_search(
            query="Find a document.",
            embedding_model=FakeEmbeddingModel(vector),  # type: ignore[arg-type]
            qdrant_client=FakeQdrantClient([]),  # type: ignore[arg-type]
            collection_name="document_chunks",
        )


def test_rejects_empty_collection_name() -> None:
    with pytest.raises(ValueError, match="Collection name"):
        semantic_search(
            query="Find a document.",
            embedding_model=FakeEmbeddingModel([1.0, 0.0]),  # type: ignore[arg-type]
            qdrant_client=FakeQdrantClient([]),  # type: ignore[arg-type]
            collection_name=" ",
        )

def test_preserves_similarity_scores_for_all_retrieved_chunks() -> None:
    embedding_model = FakeEmbeddingModel([1.0, 0.0])
    qdrant_client = FakeQdrantClient(
        [
            FakePoint(
                point_id=201,
                payload={"text": "Qdrant stores document chunks as vectors."},
                score=0.3096,
            ),
            FakePoint(
                point_id=202,
                payload={"text": "Unrelated sample text."},
                score=0.1480,
            ),
        ]
    )

    results = semantic_search(
        query="What does Qdrant store?",
        embedding_model=embedding_model,  # type: ignore[arg-type]
        qdrant_client=qdrant_client,  # type: ignore[arg-type]
        collection_name="document_chunks",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["similarity"] == pytest.approx(0.3096)
    assert results[1]["similarity"] == pytest.approx(0.1480)
def test_returns_results_in_descending_similarity_order() -> None:
    embedding_model = FakeEmbeddingModel([1.0, 0.0])
    qdrant_client = FakeQdrantClient(
        [
            FakePoint(
                point_id=301,
                payload={"text": "Lower score"},
                score=0.30,
            ),
            FakePoint(
                point_id=302,
                payload={"text": "Higher score"},
                score=0.80,
            ),
        ]
    )

    results = semantic_search(
        query="Find relevant information.",
        embedding_model=embedding_model,  # type: ignore[arg-type]
        qdrant_client=qdrant_client,  # type: ignore[arg-type]
        collection_name="document_chunks",
        top_k=2,
    )

    assert results[0]["similarity"] >= results[1]["similarity"]