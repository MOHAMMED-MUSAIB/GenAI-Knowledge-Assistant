
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from qdrant_client.http import models

from src.embeddings.embedding_model import EmbeddingModel
from src.storage.qdrant_store import (
    COLLECTION_NAME,
    create_collection_if_not_exists,
    get_chunk_by_id,
    get_qdrant_client,
    store_chunk,
)


def main() -> None:
    document_id = "persistence-check-20261021"
    chunk_index = 0
    point_id = str(uuid5(
        NAMESPACE_URL,
        f"{document_id}:{chunk_index}",
    ))

    chunk: dict[str, str | int] = {
        "text": "Temporary persistence verification point.",
        "chunk_index": chunk_index,
        "word_count": 4,
        "document_id": document_id,
        "filename": "persistence_check.txt",
    }

    embedding = np.zeros(
        EmbeddingModel.EMBEDDING_DIMENSION,
        dtype=np.float32,
    )
    embedding[0] = 1.0

    client = get_qdrant_client()

    try:
        create_collection_if_not_exists(client)

        # Avoid overwriting an existing point.
        existing = get_chunk_by_id(client, point_id)
        if existing is not None:
            raise RuntimeError(
                "The temporary test point already exists. "
                "Stop and inspect it before continuing."
            )

        store_chunk(client, chunk, embedding)
        client.close()
        client = get_qdrant_client()

        stored = get_chunk_by_id(client, point_id)
        assert stored is not None
        assert stored["text"] == chunk["text"]

        print("PASS: Test point is stored and retrievable.")
        print("Next: restart Qdrant, then run the verification command.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
