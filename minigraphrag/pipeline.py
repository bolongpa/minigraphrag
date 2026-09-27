"""The end-to-end MiniGraphRAG pipeline.

``MiniGraphRAG`` owns the full lifecycle:

* :meth:`index` — chunk documents, extract entities/relations with the LLM,
  build the knowledge graph, detect communities, summarize them, and fit the
  TF-IDF retriever.
* :meth:`query` — hybrid retrieval (lexical + graph expansion), community
  selection, and cited answer generation.
* :meth:`save` / :meth:`load` — pickle persistence of the whole index. The
  LLM client itself is *not* pickled (API keys must never land in an index
  file); pass one to :meth:`load` instead.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from .answer import Answer, answer_question
from .extract import ChunkExtraction, chunk_text, extract_from_chunk
from .graph import Community, build_graph, detect_communities, summarize_communities
from .llm import LLMClient
from .retrieve import RetrievedChunk, Retriever


class MiniGraphRAG:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.chunks: dict[str, str] = {}
        self.extractions: list[ChunkExtraction] = []
        self.graph: nx.Graph = nx.Graph()
        self.communities: list[Community] = []
        self.retriever: Retriever | None = None

    # -- indexing --------------------------------------------------------
    def index(self, documents: list[str]) -> "MiniGraphRAG":
        """Build the graph index from raw document texts."""
        self.chunks = {}
        for doc_idx, doc in enumerate(documents):
            for chunk_idx, chunk in enumerate(chunk_text(doc)):
                cid = f"doc{doc_idx}-chunk{chunk_idx}"
                self.chunks[cid] = chunk

        self.extractions = [
            extract_from_chunk(text, cid, self.llm)
            for cid, text in self.chunks.items()
        ]
        self.graph = build_graph(self.extractions)
        self.communities = detect_communities(self.graph)
        summarize_communities(self.graph, self.communities, self.llm)
        self.retriever = Retriever.build(self.chunks, self.graph)
        return self

    # -- querying --------------------------------------------------------
    def _relevant_communities(self, question: str, limit: int = 2) -> list[Community]:
        assert self.retriever is not None
        entities = self.retriever.link_entities(question)
        wanted: set[int] = set()
        for name in entities:
            cid = self.graph.nodes[name].get("community")
            if cid is not None:
                wanted.add(cid)
        picked = [c for c in self.communities if c.id in wanted][:limit]
        if not picked:
            picked = self.communities[:limit]
        return picked

    def query(self, question: str, top_k: int = 5) -> tuple[Answer, list[RetrievedChunk]]:
        """Answer ``question``; returns the answer plus retrieved chunks."""
        if self.retriever is None:
            raise RuntimeError("index() must be called before query().")
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        if not retrieved:
            return Answer(text="I could not find relevant information in the indexed documents."), []
        communities = self._relevant_communities(question)
        answer = answer_question(question, retrieved, communities, self.llm)
        return answer, retrieved

    # -- persistence -----------------------------------------------------
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["llm"] = None  # never persist clients (or their credentials)
        return state

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @classmethod
    def load(cls, path: str | Path, llm: LLMClient) -> "MiniGraphRAG":
        with open(path, "rb") as f:
            obj: MiniGraphRAG = pickle.load(f)
        obj.llm = llm
        return obj

    # -- introspection ---------------------------------------------------
    def stats(self) -> dict:
        return {
            "documents": len({cid.split("-chunk")[0] for cid in self.chunks}),
            "chunks": len(self.chunks),
            "entities": self.graph.number_of_nodes(),
            "relations": self.graph.number_of_edges(),
            "communities": len(self.communities),
        }
