"""Hybrid retrieval: lexical search + graph neighborhood expansion.

Naive RAG ranks chunks by embedding/keyword similarity to the question and
stops there. That misses facts that are *connected* to the matched chunks but
phrased differently — the classic multi-hop failure.

MiniGraphRAG does two passes and merges them:

1. **Lexical pass** — TF-IDF (scikit-learn) cosine similarity between the
   question and every chunk. This is the "vector-ish" baseline; TF-IDF is the
   default so the project runs with zero downloads and zero API keys, and it
   can be swapped for dense embeddings later (see README design notes).
2. **Graph pass** — link entity mentions in the question to graph nodes
   (substring match, longest names first), then expand one hop along the
   graph and collect the chunks those neighboring nodes/edges came from.
   Chunks reached this way get a score bonus.

The merged ranking is returned with provenance labels so answers can cite
exactly which chunks each claim came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    provenance: str  # e.g. "lexical" or "graph:Priya Nair"


@dataclass
class Retriever:
    chunks: dict[str, str]
    graph: nx.Graph
    vectorizer: TfidfVectorizer = field(repr=False)
    chunk_matrix: object = field(repr=False)  # scipy sparse matrix
    chunk_ids: list[str] = field(repr=False)

    @classmethod
    def build(cls, chunks: dict[str, str], graph: nx.Graph) -> "Retriever":
        chunk_ids = sorted(chunks.keys())
        texts = [chunks[cid] for cid in chunk_ids]
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts)
        return cls(
            chunks=chunks,
            graph=graph,
            vectorizer=vectorizer,
            chunk_matrix=matrix,
            chunk_ids=chunk_ids,
        )

    # -- entity linking ----------------------------------------------------
    def link_entities(self, question: str) -> list[str]:
        """Match question substrings against graph node names.

        Longest names first so "Project Phoenix" wins over "Phoenix".
        """
        lowered = question.lower()
        matched: list[str] = []
        for name in sorted(self.graph.nodes, key=len, reverse=True):
            if len(name) < 3:
                continue
            if name.lower() in lowered and not any(name in m for m in matched):
                matched.append(name)
        return matched

    # -- graph expansion ---------------------------------------------------
    def _graph_chunk_scores(self, entities: list[str], hops: int = 1) -> dict[str, tuple[float, str]]:
        """chunk_id -> (bonus, provenance label) from entity neighborhoods.

        The provenance label names the *query* entity whose neighborhood
        reached the chunk, e.g. ``graph:Project Phoenix``.
        """
        scores: dict[str, tuple[float, str]] = {}
        for entity in entities:
            if entity not in self.graph:
                continue
            frontier = {entity}
            visited = {entity}
            for _ in range(hops):
                next_frontier: set[str] = set()
                for name in frontier:
                    for neighbor in self.graph.neighbors(name):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
                frontier = next_frontier
            for name in visited:
                node = self.graph.nodes[name]
                for cid in node.get("chunks", set()):
                    bonus, _ = scores.get(cid, (0.0, ""))
                    scores[cid] = (bonus + 1.0, f"graph:{entity}")
                for _, _, edge in self.graph.edges(name, data=True):
                    for cid in edge.get("chunks", set()):
                        bonus, _ = scores.get(cid, (0.0, ""))
                        scores[cid] = (bonus + 0.5, f"graph:{entity}")
        return scores

    # -- public API --------------------------------------------------------
    def retrieve(self, question: str, top_k: int = 5, graph_bonus: float = 0.35) -> list[RetrievedChunk]:
        """Return the top-k chunks for ``question`` with provenance labels."""
        q_vec = self.vectorizer.transform([question])
        sims = cosine_similarity(q_vec, self.chunk_matrix)[0]

        entities = self.link_entities(question)
        graph_scores = self._graph_chunk_scores(entities)

        ranked: list[RetrievedChunk] = []
        for idx, cid in enumerate(self.chunk_ids):
            score = float(sims[idx])
            provenance = "lexical"
            if cid in graph_scores:
                bonus, label = graph_scores[cid]
                score += graph_bonus * bonus
                provenance = f"lexical+{label}" if score > 0 else label
            ranked.append(
                RetrievedChunk(chunk_id=cid, text=self.chunks[cid], score=score, provenance=provenance)
            )
        ranked.sort(key=lambda r: r.score, reverse=True)
        return [r for r in ranked[:top_k] if r.score > 0]
