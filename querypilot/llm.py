"""LLM clients. The agent only needs `complete(system, messages) -> str`."""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Protocol

import httpx


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict]) -> str: ...


class OpenAICompatibleLLM:
    """Any OpenAI-compatible chat endpoint: OpenAI, vLLM, or Ollama (http://localhost:11434/v1)."""

    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1", api_key: str | None = None,
                 timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, system: str, messages: list[dict]) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": system}, *messages]}
        r = httpx.post(f"{self.base_url}/chat/completions", json=body, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


class AnthropicLLM:
    """Claude via the Anthropic SDK (pip install 'querypilot[anthropic]')."""

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 1024):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: list[dict]) -> str:
        resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, temperature=0,
                                           system=system, messages=messages)
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


class ScriptedLLM:
    """Deterministic stand-in for tests: replays canned responses or calls a function."""

    def __init__(self, responses: list[str] | Callable[[str, list[dict]], str]):
        self._fn = responses if callable(responses) else None
        self._queue = None if callable(responses) else list(responses)
        self.calls: list[list[dict]] = []

    def complete(self, system: str, messages: list[dict]) -> str:
        self.calls.append(messages)
        if self._fn:
            return self._fn(system, messages)
        if not self._queue:
            raise RuntimeError("ScriptedLLM ran out of responses")
        return self._queue.pop(0)


def from_env() -> LLM:
    """Builds a client from QUERYPILOT_LLM (anthropic | openai | ollama) and QUERYPILOT_MODEL."""
    kind = os.environ.get("QUERYPILOT_LLM", "ollama")
    model = os.environ.get("QUERYPILOT_MODEL")
    if kind == "anthropic":
        if not model:
            raise SystemExit("set QUERYPILOT_MODEL to a Claude model id")
        return AnthropicLLM(model)
    if kind == "openai":
        return OpenAICompatibleLLM(model or "gpt-4o-mini", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                                   os.environ.get("OPENAI_API_KEY"))
    if kind == "ollama":
        return OpenAICompatibleLLM(model or "qwen2.5-coder:7b",
                                   os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"))
    raise SystemExit(f"unknown QUERYPILOT_LLM={kind}")
