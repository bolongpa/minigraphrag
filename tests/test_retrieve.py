"""Tests for hybrid retrieval (lexical + graph expansion)."""

from minigraphrag.extract import ChunkExtraction, Entity, Triple
from minigraphrag.graph import build_graph
from minigraphrag.retrieve import Retriever

CHUNKS = {
    "doc0-chunk0": "Maya Chen founded Acme Corp in 2019 and serves as its CEO.",
    "doc1-chunk0": "Raj Patel leads Project Phoenix, the platform modernization effort.",
    "doc2-chunk0": "Priya Nair mentors Raj Patel on storage engine internals.",
}


def _retriever():
    exts = [
        ChunkExtraction(
            chunk_id="doc0-chunk0",
            entities=[Entity(name="Maya Chen"), Entity(name="Acme Corp")],
            triples=[Triple(head="Maya Chen", relation="founded", tail="Acme Corp")],
        ),
        ChunkExtraction(
            chunk_id="doc1-chunk0",
            entities=[Entity(name="Raj Patel"), Entity(name="Project Phoenix")],
            triples=[Triple(head="Raj Patel", relation="leads", tail="Project Phoenix")],
        ),
        ChunkExtraction(
            chunk_id="doc2-chunk0",
            entities=[Entity(name="Priya Nair"), Entity(name="Raj Patel")],
            triples=[Triple(head="Priya Nair", relation="mentors", tail="Raj Patel")],
        ),
    ]
    return Retriever.build(CHUNKS, build_graph(exts))


def test_retrieval_finds_direct_chunk():
    retriever = _retriever()
    results = retriever.retrieve("Who is the CEO of Acme Corp?")
    assert results, "expected at least one result"
    assert results[0].chunk_id == "doc0-chunk0"
    assert "lexical" in results[0].provenance


def test_graph_expansion_surfaces_connected_chunk():
    retriever = _retriever()
    # "Raj Patel" appears in doc1 lexically; doc2 is only connected via the graph edge.
    results = retriever.retrieve("Who mentors Raj Patel?")
    ids = [r.chunk_id for r in results]
    assert "doc2-chunk0" in ids
    doc2 = next(r for r in results if r.chunk_id == "doc2-chunk0")
    assert "graph:" in doc2.provenance


def test_link_entities_prefers_longest_names():
    retriever = _retriever()
    assert "Project Phoenix" in retriever.link_entities("Tell me about Project Phoenix")


def test_retrieve_top_k_and_no_network():
    retriever = _retriever()
    results = retriever.retrieve("Acme Corp", top_k=2)
    assert len(results) <= 2
    assert all(r.score > 0 for r in results)
