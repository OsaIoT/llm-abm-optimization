"""The provider wrapper, exercised with stub clients (no network)."""

from types import SimpleNamespace

import pytest

from llm_abm_ga import config
from llm_abm_ga.llm import client


class StubOpenAI:
    def __init__(self):
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="  DECISION: (0.5, 0.5)  ")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def use_openai(monkeypatch, model, max_tokens=30):
    stub = StubOpenAI()
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_MODEL", model)
    monkeypatch.setattr(config, "MAX_TOKENS", max_tokens)
    monkeypatch.setattr(client, "_get_openai_client", lambda: stub)
    return stub


def test_standard_openai_models_use_temperature_and_max_tokens(monkeypatch):
    stub = use_openai(monkeypatch, "gpt-4o-mini", max_tokens=17)
    assert client.generate_answer([{"role": "user", "content": "hi"}]) == "DECISION: (0.5, 0.5)"
    assert stub.kwargs["temperature"] == config.LLM_TEMPERATURE
    assert stub.kwargs["max_tokens"] == 17
    assert "reasoning_effort" not in stub.kwargs


def test_reasoning_models_use_the_reasoning_parameters(monkeypatch):
    stub = use_openai(monkeypatch, "gpt-5-mini", max_tokens=17)
    client.generate_answer([{"role": "user", "content": "hi"}])
    assert stub.kwargs["max_completion_tokens"] == 17
    assert stub.kwargs["reasoning_effort"] == "low"
    assert "temperature" not in stub.kwargs


def test_gemini_honours_the_configured_token_limit(monkeypatch):
    seen = {}

    class StubGemini:
        models = SimpleNamespace(
            generate_content=lambda **kw: seen.update(kw) or SimpleNamespace(text="DECISION: (0.5, 0.5)")
        )

    monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(config, "LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setattr(config, "MAX_TOKENS", 17)
    monkeypatch.setattr(client, "_get_gemini_client", lambda: StubGemini())
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    assert client.generate_answer(messages) == "DECISION: (0.5, 0.5)"
    assert seen["config"].max_output_tokens == 17
    assert seen["contents"] == "usr"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        client.generate_answer([])


@pytest.mark.parametrize(
    "provider, openai_key, gemini_key, ok",
    [
        ("openai", "k", None, True),
        ("openai", None, "k", False),
        ("gemini", None, "k", True),
        ("gemini", "k", None, False),
        ("nope", "k", "k", False),
    ],
)
def test_require_api_key_checks_only_the_selected_provider(monkeypatch, provider, openai_key, gemini_key, ok):
    monkeypatch.setattr(config, "LLM_PROVIDER", provider)
    monkeypatch.setattr(config, "OPENAI_API_KEY", openai_key)
    monkeypatch.setattr(config, "GEMINI_API_KEY", gemini_key)
    if ok:
        config.require_api_key()
    else:
        with pytest.raises(SystemExit):
            config.require_api_key()
