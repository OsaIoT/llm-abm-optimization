"""
Client wrapper for chat-completion calls to OpenAI and Google Gemini.

Answers are deliberately short (``config.MAX_TOKENS``, 30 by default) and, for
models that can "think", the reasoning budget is kept to a minimum: a firm agent
has to reply with a single ``DECISION: (price, quality)`` line, with no
chain-of-thought.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .. import config

# --- Optional imports: only the provider you use needs to be installed ---
try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore


# --- Client caching ---
_openai_client: Any = None
_gemini_client: Any = None


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    if OpenAI is None:
        raise RuntimeError("The 'openai' package is not installed. Install it with: pip install openai")
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in a .env file or in the environment.")

    _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _get_gemini_client() -> Any:
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    if genai is None:
        raise RuntimeError("The 'google-genai' package is not installed. Install it with: pip install google-genai")
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Put it in a .env file or in the environment.")

    _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def generate_answer(messages: List[Dict[str, str]]) -> str:
    """Send ``messages`` to the configured LLM and return its (short) reply.

    Args:
        messages: Chat messages in the OpenAI format, i.e. dicts with ``role``
            (``"system"`` or ``"user"``) and ``content``.
    Returns:
        The stripped text of the model's reply.
    Raises:
        ValueError: If ``config.LLM_PROVIDER`` is neither ``"openai"`` nor ``"gemini"``.
    """
    provider = config.LLM_PROVIDER.strip().lower()

    if provider == "gemini":
        client = _get_gemini_client()

        system_instruction = None
        user_prompt = ""

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                user_prompt = msg["content"]

        generation_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.MAX_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        response = client.models.generate_content(
            model=config.LLM_MODEL,
            contents=user_prompt,
            config=generation_config,
        )
        return (response.text or "").strip()

    elif provider == "openai":
        client = _get_openai_client()

        # OpenAI reasoning models take different parameters than standard chat models.
        is_reasoning_model = config.LLM_MODEL.startswith(("o1", "o3", "gpt-5"))

        if is_reasoning_model:
            completion = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                max_completion_tokens=config.MAX_TOKENS,
                reasoning_effort="low",  # minimises the thinking budget
            )
        else:
            # Standard GPT models (gpt-4o-mini, gpt-3.5-turbo, ...)
            completion = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
            )

        content = completion.choices[0].message.content
        return (content or "").strip()

    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Use 'openai' or 'gemini'.")
