"""Pluggable LLM interface.

Generation is *grounded extractive* by default: the answer is built from
cited trusted passages, no model call at all. A generative LLM is an
optional post-step that may only rewrite the extractive draft — it can
never add claims, and its output is re-checked by the grounding check,
falling back to the extractive draft on any failure.

``OpenAILLM`` activates only if ``OPENAI_API_KEY`` is set (and only when
the caller opts in with ``RAG_USE_LLM=1``). No key is ever required, and
no key is ever committed. No other provider is wired in.
"""

from __future__ import annotations

import os
from typing import Protocol


class LLM(Protocol):
    def generate(self, system: str, context: str, question: str) -> str | None: ...


class NullLLM:
    """No LLM: extractive answers only."""

    def generate(self, system: str, context: str, question: str) -> str | None:
        return None


class OpenAILLM:
    """Minimal OpenAI chat client over httpx (no openai package needed)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, system: str, context: str, question: str) -> str | None:
        import httpx

        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": f"Context (trusted passages only):\n{context}\n\nQuestion: {question}\nAnswer using only the context, keeping every [doc_id] citation. If the context does not support an answer, say so.",
                        },
                    ],
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return None


def get_llm() -> LLM:
    """Return an LLM only when explicitly opted in AND a key is present."""
    if os.environ.get("RAG_USE_LLM") != "1":
        return NullLLM()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return NullLLM()
    return OpenAILLM(key)
