
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.semantic_search import semantic_search
from src.storage.qdrant_store import (
    COLLECTION_NAME,
    create_collection_if_not_exists,
    get_qdrant_client,
    store_chunk,
)
from uuid import uuid5, NAMESPACE_URL
from qdrant_client.http import models

def main() -> None:
    client = get_qdrant_client()
    embedding_model = EmbeddingModel()

    create_collection_if_not_exists(client)

    samples: list[dict[str, str | int]] = [
        {
            "text": "Employees receive 20 days of annual paid leave.",
            "chunk_index": 0,
            "word_count": 8,
            "document_id": "integration-test-leave",
            "filename": "leave_policy_test.txt",
        },
        {
            "text": "Employees must notify their manager when taking sick leave.",
            "chunk_index": 1,
            "word_count": 10,
            "document_id": "integration-test-leave",
            "filename": "leave_policy_test.txt",
        },
        {
            "text": "The solar system contains eight planets orbiting the Sun.",
            "chunk_index": 2,
            "word_count": 9,
            "document_id": "integration-test-science",
            "filename": "science_test.txt",
        },
    ]

    try:
        # Store the sample chunks in the real Qdrant collection.
        for chunk in samples:
            text = chunk["text"]
            assert isinstance(text, str)
            embedding = embedding_model.embed_text(text)
            store_chunk(client, chunk, embedding)

        # Search the real collection.
        results = semantic_search(
            query="How many days of annual paid leave do employees receive?",
            embedding_model=embedding_model,
            qdrant_client=client,
            collection_name=COLLECTION_NAME,
            top_k=3,
        )

        print("\nRetrieved chunks:")
        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. Score={result['similarity']:.4f} "
                f"| {result['text']}"
            )

        assert results, "Qdrant returned no results."
        assert results[0]["document_id"] == "integration-test-leave", (
            "The expected leave-policy document was not ranked first."
        )
        assert len(results) <= 3

        print("\nPASS: Real Qdrant semantic retrieval works.")

        # Verify document-level filtering against real Qdrant.
        filtered_results = semantic_search(
            query="What information is in this document?",
            embedding_model=embedding_model,
            qdrant_client=client,
            collection_name=COLLECTION_NAME,
            top_k=3,
            document_id="integration-test-science",
        )

        assert filtered_results, "Document filtering returned no results."
        assert all(
            result["document_id"] == "integration-test-science"
            for result in filtered_results
        ), "Document filtering returned results from another document."

        print("PASS: Document-level filtering works against real Qdrant.")

    finally:
        test_point_ids: list[models.ExtendedPointId] = [
            str(uuid5(
                NAMESPACE_URL,
                f"{chunk['document_id']}:{chunk['chunk_index']}",
            ))
            for chunk in samples
        ]

        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(
                points=test_point_ids,
            ),
            wait=True,
        )
        client.close()


if __name__ == "__main__":
    main()
