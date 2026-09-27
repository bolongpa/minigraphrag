"""Answer generation with citations.

The prompt packs three things:

1. the top-k retrieved chunks, numbered ``[1]..[k]``,
2. community summaries for the communities touched by the question's
   entities (the GraphRAG "global context" that helps multi-hop questions),
3. an instruction to answer *only* from the provided context and to cite
   every factual claim as ``[n]``.

Citations are parsed back out of the model's response and mapped to the
source chunks, so the returned :class:`Answer` always carries machine-
checkable provenance — the thing that makes RAG answers trustworthy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .graph import Community
from .llm import LLMClient
from .retrieve import RetrievedChunk

ANSWER_PROMPT = """TASK: ANSWER_QUESTION

Answer the question using ONLY the context below. Every factual claim in your
answer must end with a citation like [1] or [2] referring to the numbered
context chunk it came from. If the context does not contain the answer, say so
explicitly instead of guessing.

Community summaries (high-level context):
{community_context}

Retrieved context:
{retrieved_context}

Question: {question}

Answer with [n] citations:
"""


@dataclass
class Citation:
    number: int
    chunk_id: str
    chunk_text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)


def build_answer_prompt(
    question: str,
    retrieved: list[RetrievedChunk],
    community_summaries: list[str],
) -> str:
    context_lines = [
        f"[{i + 1}] {chunk.text} (source: {chunk.chunk_id})"
        for i, chunk in enumerate(retrieved)
    ]
    return ANSWER_PROMPT.format(
        community_context="\n".join(f"- {s}" for s in community_summaries) or "(none)",
        retrieved_context="\n\n".join(context_lines),
        question=question,
    )


def parse_citations(answer_text: str, retrieved: list[RetrievedChunk]) -> list[Citation]:
    """Map ``[n]`` markers in the answer back to the retrieved chunks."""
    citations: list[Citation] = []
    seen: set[int] = set()
    for match in re.finditer(r"\[(\d+)\]", answer_text):
        number = int(match.group(1))
        if number in seen or not 1 <= number <= len(retrieved):
            continue
        seen.add(number)
        chunk = retrieved[number - 1]
        citations.append(Citation(number=number, chunk_id=chunk.chunk_id, chunk_text=chunk.text))
    return citations


def answer_question(
    question: str,
    retrieved: list[RetrievedChunk],
    communities: list[Community],
    llm: LLMClient,
) -> Answer:
    """Generate a cited answer from retrieved chunks + community summaries."""
    summaries = [c.summary for c in communities if c.summary]
    prompt = build_answer_prompt(question, retrieved, summaries[:3])
    text = llm.complete(prompt).strip()
    return Answer(text=text, citations=parse_citations(text, retrieved))
