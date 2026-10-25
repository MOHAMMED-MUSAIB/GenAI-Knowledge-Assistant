
from unittest.mock import Mock, patch

import pytest

from src.rag.llm_client import create_llm_client, generate_answer


@patch("src.rag.llm_client.load_dotenv")
@patch("src.rag.llm_client.OpenAI")
def test_create_llm_client_uses_openrouter(
    mock_openai: Mock,
    mock_load_dotenv: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    create_llm_client()

    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )


@patch("src.rag.llm_client.load_dotenv")
def test_generate_answer_uses_free_router_by_default(
    mock_load_dotenv: Mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)

    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(
            message=Mock(
                content="RAG combines retrieval with generation."
            )
        )
    ]

    result = generate_answer("Explain RAG.", client=client)

    assert result == "RAG combines retrieval with generation."
    client.chat.completions.create.assert_called_once_with(
        model="openrouter/free",
        messages=[
            {"role": "user", "content": "Explain RAG."}
        ],
        temperature=0,
    )


@pytest.mark.parametrize("prompt", ["", "   ", "\n"])
def test_generate_answer_rejects_empty_prompt(prompt: str) -> None:
    with pytest.raises(ValueError, match="Prompt must not be empty"):
        generate_answer(prompt, client=Mock())


@patch("src.rag.llm_client.load_dotenv")
def test_generate_answer_rejects_empty_model_response(
    mock_load_dotenv: Mock,
) -> None:
    client = Mock()
    client.chat.completions.create.return_value.choices = [
        Mock(message=Mock(content="  "))
    ]
    with pytest.raises(
        RuntimeError,
        match="language model returned an empty answer",
    ):
        generate_answer("Explain RAG.", client=client)
def test_generate_answer_handles_api_error() -> None:
    from openai import APIError

    client = Mock()
    client.chat.completions.create.side_effect = APIError(
        message="Provider failed",
        request=Mock(),
        body=None,
    )

    with pytest.raises(
        RuntimeError,
        match="language model provider failed",
    ):
        generate_answer("Explain RAG.", client=client)

def test_generate_answer_handles_connection_error() -> None:
    from openai import APIConnectionError

    client = Mock()
    client.chat.completions.create.side_effect = APIConnectionError(
        request=Mock(),
    )

    with pytest.raises(
        RuntimeError,
        match="language model provider failed",
    ):
        generate_answer("Explain RAG.", client=client)
