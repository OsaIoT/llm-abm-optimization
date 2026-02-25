"""
Client wrapper for interacting with LLM Chat Completions (OpenAI & Gemini).
Ottimizzato per MINIMIZZARE il reasoning e forzare output diretti e immediati.
"""

from __future__ import annotations
from typing import List, Dict, Any

from .. import config

# --- Optional Imports ---
try:
    from openai import OpenAI  # type: ignore
except ImportError:
    OpenAI = None  # type: ignore

try:
    # Usiamo il nuovo SDK google-genai (come nel tuo notebook)
    from google import genai # type: ignore
    from google.genai import types # type: ignore
except ImportError:
    genai = None # type: ignore


# --- Client Caching ---
_openai_client: Any = None
_gemini_client: Any = None


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    if OpenAI is None:
        raise RuntimeError("The 'openai' package is not installed. Install it with: pip install openai")
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY.startswith("INSERISCI"):
        raise RuntimeError("OPENAI_API_KEY is not set correctly in config.")

    _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client


def _get_gemini_client() -> Any:
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client

    if genai is None:
        raise RuntimeError("The 'google-genai' package is not installed. Install it with: pip install google-genai")
    if not config.GEMINI_API_KEY or config.GEMINI_API_KEY.startswith("INSERISCI"):
        raise RuntimeError("GEMINI_API_KEY is not set correctly in config.")

    # Inizializza il nuovo client
    _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def generate_answer(messages: List[Dict[str, str]]) -> str:
    """
    Invia i messaggi all'LLM forzando una risposta breve senza reasoning.
    """
    provider = config.LLM_PROVIDER.strip().lower()
    
    # Abbiamo bisogno di pochissimi token per "DECISION: (p, q)"
    MAX_TOKENS = 30 

    if provider == "gemini":
        client = _get_gemini_client()
        
        system_instruction = None
        user_prompt = ""

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                user_prompt = msg["content"]

        # Configurazione "Zero-Thinking" per Gemini
        generation_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            # Azzera il budget di ragionamento per i modelli come gemini-2.0-flash-thinking o 3.0
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )

        response = client.models.generate_content(
            model=config.LLM_MODEL,
            contents=user_prompt,
            config=generation_config
        )
        return (response.text or "").strip()

    elif provider == "openai":
            client = _get_openai_client()
            
            # Check if it's an OpenAI reasoning model (o1, o3, etc.)
            is_reasoning_model = config.LLM_MODEL.startswith(("o1", "o3", "gpt-5"))
            
            if is_reasoning_model:
                completion = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    max_completion_tokens=config.MAX_TOKENS,
                    reasoning_effort="low" # Minimizes the thinking budget for OpenAI
                )
            else:
                # Standard GPT models (gpt-4o-mini, gpt-3.5-turbo, etc.)
                completion = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    messages=messages,
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=config.MAX_TOKENS
                )
                
            content = completion.choices[0].message.content
            return (content or "").strip()

    else:
        raise ValueError(f"Provider LLM non supportato: '{provider}'. Usa 'openai' o 'gemini'.")