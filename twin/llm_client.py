"""
Unified LLM client — supports Groq (fast+free), Anthropic, Ollama, OpenRouter, OpenAI-compat.

Default: Groq (free tier, ~500 tokens/sec, 10x faster than local Ollama)
Fallback: Ollama (free, local, no API key needed)
Switch via LLM_PROVIDER env var: "groq", "ollama", "anthropic", "openrouter", "openai_compat"
"""

from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_BASE_URLS,
    ANTHROPIC_API_KEY,
    OPENROUTER_API_KEY,
    OPENAI_API_KEY,
    GROQ_API_KEY,
)

# Reuse client across calls (connection pooling)
_client_cache: dict = {}


def _get_openai_client():
    """Get or create a cached OpenAI-compatible client."""
    if "openai" in _client_cache:
        return _client_cache["openai"]

    from openai import OpenAI

    if LLM_PROVIDER == "groq":
        client = OpenAI(base_url=LLM_BASE_URLS["groq"], api_key=GROQ_API_KEY)
    elif LLM_PROVIDER == "ollama":
        client = OpenAI(base_url=LLM_BASE_URLS["ollama"], api_key="ollama")
    elif LLM_PROVIDER == "openrouter":
        client = OpenAI(base_url=LLM_BASE_URLS["openrouter"], api_key=OPENROUTER_API_KEY)
    elif LLM_PROVIDER == "openai_compat":
        client = OpenAI(base_url=LLM_BASE_URLS.get("openai_compat"), api_key=OPENAI_API_KEY)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

    _client_cache["openai"] = client
    return client


def _get_anthropic_client():
    if "anthropic" in _client_cache:
        return _client_cache["anthropic"]
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    _client_cache["anthropic"] = client
    return client


def chat_completion(system: str, messages: list[dict], max_tokens: int = 1024) -> str:
    """Send a chat completion request to the configured LLM provider.

    Args:
        system: System prompt string.
        messages: List of {"role": "user"/"assistant", "content": "..."} dicts.
        max_tokens: Max tokens in response (default reduced to 1024 for speed).

    Returns:
        The assistant's response text.
    """
    if LLM_PROVIDER == "anthropic":
        client = _get_anthropic_client()
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    # Groq, Ollama, OpenRouter, OpenAI-compat — all use OpenAI API
    client = _get_openai_client()
    oai_messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=oai_messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
