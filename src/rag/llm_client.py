
import os

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openrouter/free"


def create_llm_client() -> OpenAI:
    """Create an OpenAI-compatible client for OpenRouter."""
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured.")

    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )


def generate_answer(
    prompt: object,
    client: OpenAI | None = None,
) -> str:
    """Generate an answer using OpenRouter's free-model router."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must not be empty.")

    if client is None:
        client = create_llm_client()

    load_dotenv()
    model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip()

    if not model:
        model = DEFAULT_MODEL

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=0,
        )
    except (APIError, APIConnectionError) as exc:
        raise RuntimeError(
            "The language model provider failed to generate an answer."
        ) from exc

    answer = response.choices[0].message.content

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The language model returned an empty answer.")

    return answer.strip()
