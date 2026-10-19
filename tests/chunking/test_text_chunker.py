# Text chunker test plan
# 1. Empty and whitespace-only input returns no chunks.
# 2. Short text returns one chunk.
# 3. Long text produces multiple chunks.
# 4. Chunks target a maximum of 500 words.
# 5. Consecutive chunks aim for 50 words of overlap.
# 6. Sentence and paragraph boundaries are preferred.
# 7. Chunk indexes start at zero and increase sequentially.
# 8. Each chunk's word_count matches its text.
# 9. Invalid chunk_size raises ValueError.
# 10. Invalid overlap raises ValueError.
# 11. A sentence longer than chunk_size has defined behavior.


import pytest

from src.chunking.text_chunker import chunk_text, count_words, split_sentences


def test_empty_text_returns_empty_list() -> None:
    assert chunk_text("") == []


def test_whitespace_only_returns_empty_list() -> None:
    assert chunk_text("   \n\t  ") == []


def test_short_text_returns_one_chunk() -> None:
    result = chunk_text("First sentence. Second sentence.")

    assert len(result) == 1
    assert result[0]["text"] == "First sentence. Second sentence."
    assert result[0]["chunk_index"] == 0
    assert result[0]["word_count"] == 4


def test_chunk_indexes_are_sequential() -> None:
    text = " ".join(f"Sentence number {i} has several words." for i in range(1, 100))
    result = chunk_text(text, chunk_size=30, overlap=5)

    assert len(result) > 1
    assert [chunk["chunk_index"] for chunk in result] == list(range(len(result)))


def test_word_count_matches_chunk_text() -> None:
    text = " ".join(f"Sentence number {i} has several words." for i in range(1, 100))
    result = chunk_text(text, chunk_size=30, overlap=5)

    for chunk in result:
        assert chunk["word_count"] == count_words(str(chunk["text"]))


def test_invalid_chunk_size_raises_value_error() -> None:
    with pytest.raises(ValueError):
        chunk_text("Some text.", chunk_size=0)


def test_negative_overlap_raises_value_error() -> None:
    with pytest.raises(ValueError):
        chunk_text("Some text.", chunk_size=10, overlap=-1)


def test_overlap_equal_to_chunk_size_raises_value_error() -> None:
    with pytest.raises(ValueError):
        chunk_text("Some text.", chunk_size=10, overlap=10)


def test_split_sentences() -> None:
    assert split_sentences("First sentence. Second sentence!") == [
        "First sentence.",
        "Second sentence!",
    ]


def test_long_sentence_is_split_to_respect_chunk_size() -> None:
    text = " ".join(f"word{i}" for i in range(25))
    result = chunk_text(text, chunk_size=10, overlap=2)

    assert len(result) == 3
    assert all(int(chunk["word_count"]) <= 10 for chunk in result)


def test_chunks_share_overlap_words() -> None:
    sentences = [
        f"Sentence {i} contains words alpha beta gamma delta."
        for i in range(100)
    ]
    text = " ".join(sentences)

    result = chunk_text(text, chunk_size=30, overlap=5)

    assert len(result) > 1

    for previous, current in zip(result, result[1:]):
        previous_words = str(previous["text"]).split()
        current_words = str(current["text"]).split()

        shared_words = 0
        max_overlap = min(len(previous_words), len(current_words))

        for length in range(1, max_overlap + 1):
            if previous_words[-length:] == current_words[:length]:
                shared_words = length

        assert shared_words > 0, (
            "Consecutive chunks should share some words."
        )


def test_chunks_preserve_sentence_boundaries() -> None:
    text = "First sentence has four words. Second sentence has four words. Third sentence has four words."

    result = chunk_text(text, chunk_size=8, overlap=2)

    assert len(result) > 1
    for chunk in result:
        chunk_text_value = str(chunk["text"])
        assert not chunk_text_value.startswith(".")


def test_overlap_matches_requested_word_count() -> None:
    text = " ".join(
        f"Sentence {i} contains words alpha beta gamma delta."
        for i in range(100)
    )

    result = chunk_text(text, chunk_size=30, overlap=5)

    assert len(result) > 1

    for previous, current in zip(result, result[1:]):
        previous_words = str(previous["text"]).split()
        current_words = str(current["text"]).split()

        assert previous_words[-5:] == current_words[:5]


def test_chunking_progresses_when_sentence_boundary_is_too_early() -> None:
    text = (
        "A. This sentence contains many words "
        "beyond the chunk limit and keeps going."
    )

    chunks = chunk_text(text, chunk_size=5, overlap=4)

    assert len(chunks) > 1
    assert all(int(chunk["word_count"]) <= 5 for chunk in chunks)
    assert [chunk["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )


def test_split_sentences_with_whitespace_returns_empty_list() -> None:
    assert split_sentences("   ") == []


def test_chunking_preserves_all_source_words_in_order() -> None:
    text: str = (
        "The first sentence contains several words. "
        "The second sentence contains several more words. "
        "The third sentence completes the document."
    )

    chunks = chunk_text(text, chunk_size=6, overlap=2)

    source_words = text.split()
    reconstructed_words: list[str] = []

    for chunk in chunks:
        chunk_text_value = str(chunk["text"])
        chunk_words: list[str] = chunk_text_value.split()

        if not reconstructed_words:
            reconstructed_words.extend(chunk_words)
        else:
            reconstructed_words.extend(chunk_words[2:])

    assert reconstructed_words == source_words

def test_split_sentences_preserves_paragraph_content() -> None:
    text = "First paragraph has two sentences. Second sentence.\n\nSecond paragraph has one sentence."

    sentences = split_sentences(text)

    assert sentences == [
        "First paragraph has two sentences.",
        "Second sentence.",
        "Second paragraph has one sentence.",
    ]

def test_chunk_end_prefers_paragraph_boundary() -> None:
    text = (
        "One two three four five six.\n\n"
        "Seven eight nine ten eleven twelve."
    )

    chunks = chunk_text(text, chunk_size=7, overlap=1)

    first_chunk = chunks[0]["text"]
    assert isinstance(first_chunk, str)
    assert first_chunk == "One two three four five six."


def test_chunk_indexes_and_word_counts_are_correct() -> None:
    text = "One two three four five six seven eight nine ten."

    chunks = chunk_text(text, chunk_size=5, overlap=1)

    for expected_index, chunk in enumerate(chunks):
        chunk_text_value = chunk["text"]
        assert isinstance(chunk_text_value, str)

        assert chunk["chunk_index"] == expected_index
        assert chunk["word_count"] == len(chunk_text_value.split())


def test_chunk_text_includes_document_identity_metadata() -> None:
    chunks = chunk_text(
        "This is a document.",
        chunk_size=10,
        overlap=2,
        document_id="doc-123",
        filename="example.txt",
    )

    assert len(chunks) == 1
    assert chunks[0]["document_id"] == "doc-123"
    assert chunks[0]["filename"] == "example.txt"


def test_chunk_text_omits_unprovided_metadata() -> None:
    chunks = chunk_text(
        "This is a document.",
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 1
    assert "document_id" not in chunks[0]
    assert "filename" not in chunks[0]
