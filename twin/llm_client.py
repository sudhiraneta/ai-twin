"""
Unified LLM client — supports Anthropic, Ollama (free/local), OpenRouter, and OpenAI-compatible APIs.

Default: Ollama (free, runs locally with `ollama serve`)
Switch via LLM_PROVIDER env var: "ollama", "anthropic", "openrouter", "openai_compat"
"""

from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_BASE_URLS,
    ANTHROPIC_API_KEY,
    OPENROUTER_API_KEY,
    OPENAI_API_KEY,
)


def _create_anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _create_openai_client():
    from openai import OpenAI

    if LLM_PROVIDER == "ollama":
        return OpenAI(base_url=LLM_BASE_URLS["ollama"], api_key="ollama")
    elif LLM_PROVIDER == "openrouter":
        return OpenAI(base_url=LLM_BASE_URLS["openrouter"], api_key=OPENROUTER_API_KEY)
    elif LLM_PROVIDER == "openai_compat":
        return OpenAI(base_url=LLM_BASE_URLS.get("openai_compat"), api_key=OPENAI_API_KEY)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def chat_completion(system: str, messages: list[dict], max_tokens: int = 4096) -> str:
    """Send a chat completion request to the configured LLM provider.

    Args:
        system: System prompt string.
        messages: List of {"role": "user"/"assistant", "content": "..."} dicts.
        max_tokens: Max tokens in response.

    Returns:
        The assistant's response text.
    """
    if LLM_PROVIDER == "anthropic":
        client = _create_anthropic_client()
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    # All other providers use OpenAI-compatible API
    client = _create_openai_client()
    oai_messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=oai_messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
