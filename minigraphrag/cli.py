"""Command-line interface.

Examples
--------
Index the bundled sample docs::

    python -m minigraphrag index data/*.md --out index.pkl

Ask a question against the saved index::

    python -m minigraphrag ask "Who mentors the engineer leading Project Phoenix?" \\
        --index index.pkl

Use a real model instead of the deterministic fake::

    export LLM_BASE_URL=https://api.openai.com/v1 LLM_API_KEY=sk-... LLM_MODEL=gpt-4o-mini
    python -m minigraphrag index data/*.md --out index.pkl --llm openai
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from .llm import FakeLLM, LLMClient, OpenAICompatibleLLM
from .pipeline import MiniGraphRAG


def _make_llm(kind: str) -> LLMClient:
    if kind == "fake":
        return FakeLLM()
    if kind == "openai":
        return OpenAICompatibleLLM()
    raise ValueError(f"unknown --llm {kind!r}; expected 'fake' or 'openai'")


def cmd_index(args: argparse.Namespace) -> int:
    paths: list[str] = []
    for pattern in args.files:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        print("no input files matched", file=sys.stderr)
        return 1
    documents = [Path(p).read_text(encoding="utf-8") for p in paths]
    print(f"indexing {len(documents)} document(s) with llm={args.llm} ...")
    rag = MiniGraphRAG(_make_llm(args.llm)).index(documents)
    rag.save(args.out)
    stats = rag.stats()
    print(f"saved index to {args.out}")
    print(
        "stats: "
        + ", ".join(f"{k}={v}" for k, v in stats.items())
    )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    rag = MiniGraphRAG.load(args.index, _make_llm(args.llm))
    answer, retrieved = rag.query(args.question, top_k=args.top_k)
    print(f"\nQ: {args.question}\n")
    print(f"A: {answer.text}\n")
    if answer.citations:
        print("Citations:")
        for c in answer.citations:
            snippet = " ".join(c.chunk_text.split()[:30])
            print(f"  [{c.number}] {c.chunk_id}: {snippet}...")
    if args.verbose:
        print("\nRetrieved chunks:")
        for r in retrieved:
            print(f"  {r.chunk_id} score={r.score:.3f} via={r.provenance}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="minigraphrag", description="Minimal GraphRAG over your docs.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="build a graph index from markdown files")
    p_index.add_argument("files", nargs="+", help="input files (globs allowed), e.g. data/*.md")
    p_index.add_argument("--out", default="index.pkl", help="where to save the index")
    p_index.add_argument("--llm", default="fake", choices=["fake", "openai"])
    p_index.set_defaults(func=cmd_index)

    p_ask = sub.add_parser("ask", help="ask a question against a saved index")
    p_ask.add_argument("question", help="the question to answer")
    p_ask.add_argument("--index", default="index.pkl", help="saved index from the index command")
    p_ask.add_argument("--llm", default="fake", choices=["fake", "openai"])
    p_ask.add_argument("--top-k", type=int, default=5)
    p_ask.add_argument("--verbose", action="store_true", help="also print retrieval details")
    p_ask.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
