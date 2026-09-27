"""LLM client abstractions.

MiniGraphRAG talks to language models through the tiny ``LLMClient`` protocol
below, so the indexing/query pipeline never depends on a specific provider.

Two implementations ship with the project:

* :class:`OpenAICompatibleLLM` — hits any OpenAI-compatible ``/chat/completions``
  endpoint (OpenAI, vLLM, Ollama, llama.cpp server, ...). Configured purely
  through environment variables so no secrets ever live in code.
* :class:`FakeLLM` — deterministic, offline, scripted responses. Used by the
  test-suite, the ``--llm fake`` CLI demo, and anyone who wants to try the
  pipeline without an API key.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol

import requests

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class LLMClient(Protocol):
    """Minimal interface the pipeline needs from a language model."""

    def complete(self, prompt: str) -> str:
        """Return the model's raw text completion for ``prompt``."""
        ...


# ---------------------------------------------------------------------------
# OpenAI-compatible client
# ---------------------------------------------------------------------------


class OpenAICompatibleLLM:
    """Client for any OpenAI-compatible chat-completions endpoint.

    Configuration comes from the environment:

    * ``LLM_BASE_URL`` — e.g. ``https://api.openai.com/v1``
      (default) or ``http://localhost:11434/v1`` for Ollama.
    * ``LLM_API_KEY`` — bearer token. May be empty for local servers.
    * ``LLM_MODEL`` — model name, e.g. ``gpt-4o-mini`` (default).
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout

    def complete(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Deterministic fake for tests / demos
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "this", "that", "these", "those", "with", "from", "and", "for",
    "are", "was", "were", "has", "have", "had", "its", "our", "their",
    "each", "both", "such", "when", "where", "which", "who", "what",
    "why", "how", "not", "but", "all", "any", "can", "will", "may",
    "into", "over", "under", "between", "during", "after", "before",
    "founded", "based", "led",
}

_MULTIWORD_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*)+)\b")
_SINGLEWORD_RE = re.compile(r"\b([A-Z][A-Za-z0-9]{3,})\b")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _guess_type(name: str) -> str:
    upper = name.upper()
    if any(k in upper for k in ("CORP", "INC", "LLC", "LTD", "CO.")):
        return "Organization"
    if "PROJECT" in upper:
        return "Project"
    if upper.endswith("DB") or "DATABASE" in upper:
        return "Technology"
    if upper.endswith("AI") and len(name.split()) == 2:
        return "Product"
    if len(name.split()) >= 2:
        return "Person"
    return "Concept"


def _fake_extract(chunk: str) -> str:
    """Heuristic stand-in for LLM extraction: capitalized phrases -> entities.

    Deterministic and offline. Emits the same JSON schema the real prompt
    asks for, so the rest of the pipeline is exercised identically.
    """
    sentences = [s.strip() for s in _SENTENCE_RE.split(chunk) if s.strip()]
    entities: dict[str, dict] = {}
    triples: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    for sent in sentences:
        found: list[str] = []
        for m in _MULTIWORD_RE.finditer(sent):
            name = " ".join(m.group(1).split())
            if name not in found:
                found.append(name)
        for m in _SINGLEWORD_RE.finditer(sent):
            name = m.group(1)
            # skip tokens already covered by a multi-word phrase
            if any(name in phrase.split() for phrase in found):
                continue
            if name.lower() in _STOPWORDS:
                continue
            if name not in found:
                found.append(name)
        for name in found:
            entities.setdefault(
                name,
                {"name": name, "type": _guess_type(name), "description": sent},
            )
        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                pair = (found[i], found[j])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                triples.append(
                    {
                        "head": found[i],
                        "relation": "related_to",
                        "tail": found[j],
                        "description": sent,
                    }
                )
    return json.dumps({"entities": list(entities.values()), "triples": triples})


def _fake_summarize(prompt: str) -> str:
    names = re.findall(r"-\s+(.+?)\s+\(", prompt)
    names = names or ["(no entities listed)"]
    shown = ", ".join(names[:12])
    return (
        "Community summary: this cluster centers on "
        f"{shown}. The entities are densely interlinked, suggesting they "
        "belong to the same functional area of the documented domain."
    )


def _clean_echo(text: str) -> str:
    text = re.sub(r"\*+", "", text)  # drop markdown bold
    text = re.sub(r"^#\s+", "", text)  # drop markdown headings
    return " ".join(text.split())


def _fake_answer(prompt: str) -> str:
    # Only look at the numbered context section, never the instructions above it.
    ctx = prompt.split("Retrieved context:\n", 1)[1] if "Retrieved context:\n" in prompt else prompt
    ctx = ctx.split("\n\nQuestion:", 1)[0]
    blocks = re.findall(r"\[(\d+)\]\s*(.*?)(?=\n\[\d+\]|\Z)", ctx, re.S)
    blocks = [(n, _clean_echo(t)) for n, t in blocks if t.strip()]
    if not blocks:
        return "I could not find relevant information in the provided context."
    seen: set[str] = set()
    numbers: list[str] = []
    for n, _ in blocks:
        if n not in seen:
            seen.add(n)
            numbers.append(n)
    cites = " ".join(f"[{n}]" for n in numbers[:4])
    first = " ".join(blocks[0][1].split()[:40])
    answer = f"Based on the retrieved context {cites}: {first}"
    if len(blocks) > 1:
        second = " ".join(blocks[1][1].split()[:30])
        answer += f" Additionally, {second}"
    return answer


class FakeLLM:
    """Deterministic offline LLM used by tests and the ``--llm fake`` demo.

    It routes on the task markers embedded in the prompts built by
    :mod:`minigraphrag.extract`, :mod:`minigraphrag.graph` and
    :mod:`minigraphrag.answer`, so prompt changes only require updating the
    markers in one place.
    """

    def complete(self, prompt: str) -> str:
        if "TASK: EXTRACT_ENTITIES" in prompt:
            chunk = prompt.split("CHUNK:\n", 1)[1] if "CHUNK:\n" in prompt else prompt
            return _fake_extract(chunk)
        if "TASK: SUMMARIZE_COMMUNITY" in prompt:
            return _fake_summarize(prompt)
        if "TASK: ANSWER_QUESTION" in prompt:
            return _fake_answer(prompt)
        return "FakeLLM received an unrecognized prompt."
