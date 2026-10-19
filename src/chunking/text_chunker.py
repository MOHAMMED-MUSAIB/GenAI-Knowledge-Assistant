
import re


def count_words(text: str) -> int:
    """Return the number of whitespace-separated words in text."""
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    """Split text into sentences while preserving paragraph content."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    sentences: list[str] = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        parts = re.split(r"(?<=[.!?])\s+", paragraph)

        for part in parts:
            sentence = part.strip()
            if sentence:
                sentences.append(sentence)

    return sentences


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    *,
    document_id: str | None = None,
    filename: str | None = None,
) -> list[dict[str, str | int]]:
    """Create paragraph-aware chunks with exact word overlap.

    Prefer paragraph boundaries, then sentence boundaries, when practical.
    Every chunk includes its index and word count. Document identity metadata
    is included when provided.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )

    if not text or not text.strip():
        return []

    # Record the word positions at the end of each paragraph and sentence.
    paragraphs = re.split(r"\n\s*\n", text.strip())
    words: list[str] = []
    paragraph_ends: list[int] = []
    sentence_ends: list[int] = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_start = len(words)
        paragraph_words = paragraph.split()
        words.extend(paragraph_words)

        if paragraph_words:
            paragraph_ends.append(len(words))

        parts = re.split(r"(?<=[.!?])\s+", paragraph)
        sentence_word_count = 0

        for part in parts:
            sentence_word_count += len(part.split())
            if part.strip():
                sentence_ends.append(
                    paragraph_start + sentence_word_count
                )

    if not words:
        return []

    chunks: list[dict[str, str | int]] = []
    start = 0

    while start < len(words):
        max_end = min(start + chunk_size, len(words))
        end = max_end

        if max_end < len(words):
            minimum_end = start + overlap

            # Prefer paragraph boundaries when one fits.
            valid_paragraph_ends = [
                boundary
                for boundary in paragraph_ends
                if minimum_end < boundary <= max_end
            ]

            if valid_paragraph_ends:
                end = max(valid_paragraph_ends)
            else:
                # Otherwise, prefer a sentence boundary.
                valid_sentence_ends = [
                    boundary
                    for boundary in sentence_ends
                    if minimum_end < boundary <= max_end
                ]

                if valid_sentence_ends:
                    end = max(valid_sentence_ends)

        chunk_words = words[start:end]
        chunk: dict[str, str | int] = {
            "text": " ".join(chunk_words),
            "chunk_index": len(chunks),
            "word_count": len(chunk_words),
        }

        if document_id is not None:
            chunk["document_id"] = document_id

        if filename is not None:
            chunk["filename"] = filename

        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks
