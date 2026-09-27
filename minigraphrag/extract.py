"""Document chunking and LLM-based entity/relation extraction.

The indexing pipeline splits each document into overlapping chunks and asks
the LLM to pull out ``(head, relation, tail)`` triples plus typed entity
descriptions. Output parsing is deliberately defensive: real models love to
wrap JSON in markdown fences, trail commas, or fall back to ad-hoc formats,
and indexing a 10k-document corpus should not die on chunk #7,431.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import LLMClient

EXTRACTION_PROMPT = """TASK: EXTRACT_ENTITIES

You are an information extraction system. Read the text chunk below and extract
all salient entities and the relationships between them.

Return ONLY a JSON object with this exact schema (no markdown fences, no prose):

{{
  "entities": [
    {{"name": "Entity Name", "type": "Person|Organization|Product|Project|Technology|Location|Concept", "description": "one-sentence description grounded in the chunk"}}
  ],
  "triples": [
    {{"head": "Entity Name", "relation": "short snake_case relation", "tail": "Entity Name", "description": "one-sentence evidence from the chunk"}}
  ]
}}

Rules:
- Use the exact entity name strings in triples as in the entities list.
- Keep relations short and specific (e.g. "leads", "founded", "powers", "reports_to").
- Do not invent entities or facts not present in the chunk.

CHUNK:
{chunk}
"""


@dataclass
class Entity:
    name: str
    type: str = "Concept"
    description: str = ""


@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    description: str = ""


@dataclass
class ChunkExtraction:
    chunk_id: str
    entities: list[Entity] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split ``text`` into overlapping character chunks on sentence boundaries."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        if current and current_len + len(sent) > max_chars:
            chunks.append(" ".join(current))
            # carry overlap into the next chunk
            carried: list[str] = []
            carried_len = 0
            for s in reversed(current):
                if carried_len + len(s) > overlap:
                    break
                carried.insert(0, s)
                carried_len += len(s)
            current, current_len = carried, carried_len
        current.append(sent)
        current_len += len(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        # drop the opening fence line (``` or ```json) and the closing fence
        lines = raw.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    return raw.strip()


def _coerce(obj: dict) -> tuple[list[Entity], list[Triple]]:
    entities = [
        Entity(
            name=str(e.get("name", "")).strip(),
            type=str(e.get("type", "Concept")).strip() or "Concept",
            description=str(e.get("description", "")).strip(),
        )
        for e in obj.get("entities", [])
        if isinstance(e, dict) and str(e.get("name", "")).strip()
    ]
    triples = [
        Triple(
            head=str(t.get("head", "")).strip(),
            relation=str(t.get("relation", "related_to")).strip() or "related_to",
            tail=str(t.get("tail", "")).strip(),
            description=str(t.get("description", "")).strip(),
        )
        for t in obj.get("triples", [])
        if isinstance(t, dict) and str(t.get("head", "")).strip() and str(t.get("tail", "")).strip()
    ]
    return entities, triples


def parse_extraction_output(raw: str) -> tuple[list[Entity], list[Triple]]:
    """Parse LLM extraction output into entities and triples.

    Tries, in order:
    1. strict ``json.loads`` (after stripping markdown fences),
    2. extracting the largest ``{...}`` JSON span via regex,
    3. a line-based ``head | relation | tail`` fallback.

    Never raises on malformed input — returns empty lists instead, so one bad
    chunk cannot poison a whole indexing run.
    """
    text = _strip_fences(raw)
    candidates = [text]
    match = re.search(r"\{.*\}", text, re.S)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return _coerce(obj)

    # Fallback: "head | relation | tail" lines, e.g. "Maya Chen | founded | Acme Corp"
    entities: dict[str, Entity] = {}
    triples: list[Triple] = []
    for line in text.splitlines():
        parts = [p.strip().strip("`\"'") for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            head, relation, tail = parts
            for name in (head, tail):
                entities.setdefault(name, Entity(name=name))
            triples.append(Triple(head=head, relation=relation, tail=tail))
    return list(entities.values()), triples


def extract_from_chunk(chunk: str, chunk_id: str, llm: LLMClient) -> ChunkExtraction:
    """Run one chunk through the LLM extractor and parse the result."""
    raw = llm.complete(EXTRACTION_PROMPT.format(chunk=chunk))
    entities, triples = parse_extraction_output(raw)
    return ChunkExtraction(chunk_id=chunk_id, entities=entities, triples=triples)
