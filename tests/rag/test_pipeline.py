from typing import Any
from unittest.mock import Mock, patch
from unittest.mock import Mock, patch
import pytest

from src.rag.pipeline import (
    build_prompt,
    prepare_context,
    retrieve_context,
    extract_sources,
    validate_question,
)
from src.storage.qdrant_store import COLLECTION_NAME


def test_validate_question_returns_question():
    assert validate_question("What is RAG?") == "What is RAG?"


def test_validate_question_strips_whitespace():
    assert validate_question("  What is RAG?  ") == "What is RAG?"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_validate_question_rejects_empty_input(question: str):
    with pytest.raises(ValueError, match="Question must not be empty."):
        validate_question(question)


def test_validate_question_rejects_non_string_input():
    with pytest.raises(ValueError, match="Question must be a string."):
        validate_question(None)

@patch("src.rag.pipeline.semantic_search")
def test_retrieve_context_calls_semantic_search(mock_search: Mock):
    expected_chunks: list[dict[str, Any]] = [
        {
            "text": "Employees receive annual leave.",
            "document_id": "doc-1",
            "filename": "policy.txt",
            "similarity": 0.85,
            "id": "chunk-1",
        }
    ]
    mock_search.return_value = expected_chunks

    embedding_model = Mock()
    qdrant_client = Mock()

    result = retrieve_context(
        question="  What is the leave policy?  ",
        embedding_model=embedding_model,
        qdrant_client=qdrant_client,
        top_k=2,
    )

    assert result == expected_chunks
    mock_search.assert_called_once_with(
        query="What is the leave policy?",
        embedding_model=embedding_model,
        qdrant_client=qdrant_client,
        collection_name=COLLECTION_NAME,
        top_k=2,
    )


@patch("src.rag.pipeline.semantic_search")
def test_retrieve_context_rejects_empty_question(mock_search: Mock):
    with pytest.raises(ValueError, match="Question must not be empty."):
        retrieve_context(
            question="   ",
            embedding_model=Mock(),
            qdrant_client=Mock(),
        )

    mock_search.assert_not_called()


def test_prepare_context_formats_source_and_text():
    chunks: list[dict[str, Any]] = [
        {
            "text": "Employees receive 20 days of annual leave.",
            "filename": "leave_policy.pdf",
            "document_id": "doc-1",
            "chunk_index": 0,
        }
    ]

    result = prepare_context(chunks)

    assert "[Source: leave_policy.pdf;" in result
    assert "Document ID: doc-1" in result
    assert "Chunk index: 0" in result
    assert "Employees receive 20 days of annual leave." in result


def test_prepare_context_separates_multiple_chunks():
    chunks: list[dict[str, Any]] = [
        {"text": "First chunk.", "filename": "a.txt"},
        {"text": "Second chunk.", "filename": "b.txt"},
    ]

    result = prepare_context(chunks)

    assert "First chunk.\n\n" in result
    assert "Second chunk." in result
    assert result.index("First chunk.") < result.index("Second chunk.")

def test_prepare_context_assigns_source_ids_to_chunks() -> None:
    chunks: list[dict[str, Any]] = [
        {
            "text": "Qdrant stores document chunks as vectors.",
            "filename": "guide.txt",
            "document_id": "doc-1",
            "chunk_index": 0,
        },
        {
            "text": "The solar system contains planets.",
            "filename": "sample.txt",
            "document_id": "doc-2",
            "chunk_index": 0,
        },
    ]

    result = prepare_context(chunks)

    assert "[S1]" in result
    assert "[S2]" in result
    assert "Qdrant stores document chunks as vectors." in result
    assert "The solar system contains planets." in result
def test_prepare_context_skips_empty_text():
    chunks: list[dict[str, Any]] = [
        {"text": "   ", "filename": "empty.txt"},
        {"text": "Useful information.", "filename": "useful.txt"},
    ]

    result = prepare_context(chunks)

    assert "empty.txt" not in result
    assert "Useful information." in result


def test_prepare_context_returns_empty_string_for_no_chunks():
    assert prepare_context([]) == ""


def test_prepare_context_handles_missing_filename():
    result = prepare_context(
        [{"text": "Some information."}]
    )

    assert "Unknown source" in result
    assert "Some information." in result

def test_build_prompt_includes_question_context_and_instructions() -> None:
    question = "What is the leave policy?"
    context = "[Source: policy.txt]\nEmployees receive 12 days of leave."

    prompt = build_prompt(question, context)

    assert question in prompt
    assert context in prompt
    assert "using only the supplied context" in prompt
    assert "say so clearly" in prompt
    assert "Do not invent facts or sources." in prompt
    assert "untrusted reference material" in prompt
    assert "Answer:" in prompt


def test_build_prompt_trims_question_and_context() -> None:
    prompt = build_prompt(
        "  What is the policy?  ",
        "  Policy details are provided here.  ",
    )

    assert "Question:\nWhat is the policy?" in prompt
    assert "Context:\nPolicy details are provided here." in prompt


def test_build_prompt_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="Question must not be empty"):
        build_prompt("   ", "Some context")


def test_build_prompt_rejects_empty_context() -> None:
    with pytest.raises(ValueError, match="Context must not be empty"):
        build_prompt("What is the policy?", "   ")


def test_build_prompt_rejects_non_string_context() -> None:
    with pytest.raises(ValueError, match="Context must be a string"):
        build_prompt("What is the policy?", None)

def test_extract_sources_returns_document_metadata() -> None:
    chunks: list[dict[str, Any]] = [
        {
            "text": "RAG combines retrieval with generation.",
            "filename": "rag-guide.pdf",
            "document_id": "doc-123",
            "chunk_index": 2,
        }
    ]

    sources = extract_sources(chunks)

    assert sources == [
        {
            "filename": "rag-guide.pdf",
            "document_id": "doc-123",
            "chunk_index": 2,
        }
    ]

from unittest.mock import patch


@patch("src.rag.pipeline.generate_answer")
def test_answer_with_sources_returns_answer_and_citations(
    mock_generate_answer: Mock,
) -> None:
    from src.rag.pipeline import answer_with_sources

    mock_generate_answer.return_value = (
    "RAG retrieves relevant document context. [S1]"
)

    chunks: list[dict[str, Any]] = [
        {
            "text": "RAG retrieves relevant document context.",
            "filename": "rag-guide.pdf",
            "document_id": "doc-123",
            "chunk_index": 2,
        }
    ]

    result = answer_with_sources("What is RAG?", chunks)

    assert result == {
        "answer": "RAG retrieves relevant document context. [S1]",
        "sources": [
            {
                "filename": "rag-guide.pdf",
                "document_id": "doc-123",
                "chunk_index": 2,
            }
        ],
    }
    mock_generate_answer.assert_called_once()

def test_answer_with_sources_handles_empty_chunks() -> None:
    from src.rag.pipeline import answer_with_sources

    result = answer_with_sources("What is RAG?", [])

    assert result == {
        "answer": (
            "I could not find relevant information in the provided documents."
        ),
        "sources": [],
    }

from unittest.mock import patch


@patch("src.rag.pipeline.generate_answer")
@patch("src.rag.pipeline.retrieve_context")
def test_answer_question_orchestrates_rag_pipeline(
    mock_retrieve_context: Mock,
    mock_generate_answer: Mock,
) -> None:
    from src.rag.pipeline import answer_question

    mock_retrieve_context.return_value = [
        {
            "text": "RAG retrieves relevant context before generating an answer.",
            "filename": "rag-guide.pdf",
            "document_id": "doc-123",
            "chunk_index": 0,
        }
    ]
    mock_generate_answer.return_value = (
        "Qdrant stores document chunks as vectors. [S1]"
    )   
    result = answer_question(
        question="What is RAG?",
        embedding_model=Mock(),
        qdrant_client=Mock(),
    )

    assert result == {
        "answer": "Qdrant stores document chunks as vectors. [S1]",
        "sources": [
            {
                "filename": "rag-guide.pdf",
                "document_id": "doc-123",
                "chunk_index": 0,
            }
        ],
    }

    mock_retrieve_context.assert_called_once()
    mock_generate_answer.assert_called_once()

def test_extract_sources_skips_chunks_with_empty_text() -> None:
    chunks: list[dict[str, Any]] = [
        {
            "text": "   ",
            "filename": "empty.txt",
            "document_id": "doc-empty",
            "chunk_index": 0,
        },
        {
            "text": "Qdrant stores document chunks as vectors.",
            "filename": "guide.txt",
            "document_id": "doc-guide",
            "chunk_index": 1,
        },
    ]

    sources = extract_sources(chunks)

    assert sources == [
        {
            "filename": "guide.txt",
            "document_id": "doc-guide",
            "chunk_index": 1,
        }
    ]

@patch("src.rag.pipeline.generate_answer")
def test_answer_with_sources_preserves_retrieved_chunk_metadata(
    mock_generate_answer: Mock,
) -> None:
    from src.rag.pipeline import answer_with_sources

    mock_generate_answer.return_value = (
    "Qdrant stores document chunks as vectors. [S1]"
)

    chunks: list[dict[str, Any]] = [
        {
            "text": "Qdrant stores document chunks as vectors.",
            "filename": "guide.txt",
            "document_id": "doc-1",
            "chunk_index": 0,
            "similarity": 0.31,
            "id": "point-1",
        },
        {
            "text": "Sample text about the solar system.",
            "filename": "sample.txt",
            "document_id": "doc-2",
            "chunk_index": 0,
            "similarity": 0.15,
            "id": "point-2",
        },
    ]

    result = answer_with_sources("What does Qdrant store?", chunks)

    assert result["sources"] == [
        {
            "filename": "guide.txt",
            "document_id": "doc-1",
            "chunk_index": 0,
        },

    ]

    prompt = mock_generate_answer.call_args.args[0]
    assert "Qdrant stores document chunks as vectors." in prompt
    assert "Sample text about the solar system." in prompt
@patch("src.rag.pipeline.generate_answer")
def test_answer_with_sources_returns_multiple_cited_sources(
    mock_generate_answer: Mock,
) -> None:
    from src.rag.pipeline import answer_with_sources

    mock_generate_answer.return_value = "Both facts are relevant. [S1] [S2]"

    chunks: list[dict[str, Any]] = [
        {
            "text": "Qdrant stores vectors.",
            "filename": "guide.txt",
            "document_id": "doc-1",
            "chunk_index": 0,
        },
        {
            "text": "React builds user interfaces.",
            "filename": "frontend.txt",
            "document_id": "doc-2",
            "chunk_index": 0,
        },
    ]

    result = answer_with_sources("Explain the project.", chunks)

    assert [source["filename"] for source in result["sources"]] == [
        "guide.txt",
        "frontend.txt",
    ]


@patch("src.rag.pipeline.generate_answer")
def test_answer_with_sources_ignores_unknown_citation_ids(
    mock_generate_answer: Mock,
) -> None:
    from src.rag.pipeline import answer_with_sources

    mock_generate_answer.return_value = "Unsupported citation. [S99]"

    chunks: list[dict[str, Any]] = [
        {
            "text": "Qdrant stores vectors.",
            "filename": "guide.txt",
            "document_id": "doc-1",
            "chunk_index": 0,
        }
    ]

    result = answer_with_sources("What does Qdrant store?", chunks)

    assert result["sources"] == []