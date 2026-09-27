"""End-to-end tests: index -> query -> citations, plus save/load."""

import pytest

from minigraphrag.answer import parse_citations
from minigraphrag.llm import FakeLLM
from minigraphrag.pipeline import MiniGraphRAG

DOCS = [
    "Maya Chen founded Acme Corp in 2019. She serves as the CEO of Acme Corp.",
    "Project Phoenix is led by Raj Patel. Project Phoenix migrates pipelines onto NimbusDB.",
    "NimbusDB is the distributed database of Acme Corp. Priya Nair leads NimbusDB development "
    "and mentors Raj Patel on storage internals.",
]


@pytest.fixture()
def rag():
    return MiniGraphRAG(FakeLLM()).index(DOCS)


def test_end_to_end_query_produces_citations(rag):
    answer, retrieved = rag.query("Who mentors the engineer leading Project Phoenix?")
    assert retrieved, "expected retrieved chunks"
    assert answer.citations, "expected citations in the answer"
    for citation in answer.citations:
        assert citation.chunk_id in {r.chunk_id for r in retrieved}
    # the evidence for the multi-hop answer must be retrievable
    assert any("Priya Nair" in r.text for r in retrieved)


def test_query_before_index_raises():
    with pytest.raises(RuntimeError):
        MiniGraphRAG(FakeLLM()).query("anything")


def test_query_with_no_match_returns_graceful_answer(rag):
    answer, retrieved = rag.query("xylophone quantum zebras")
    assert retrieved == []
    assert "could not find" in answer.text.lower()


def test_save_load_roundtrip(rag, tmp_path):
    path = tmp_path / "index.pkl"
    rag.save(path)
    loaded = MiniGraphRAG.load(path, FakeLLM())
    assert loaded.stats() == rag.stats()
    answer, _ = loaded.query("Who founded Acme Corp?")
    assert answer.citations


def test_stats_shape(rag):
    stats = rag.stats()
    assert stats["documents"] == 3
    assert stats["chunks"] >= 3
    assert stats["entities"] > 0
    assert stats["relations"] > 0
    assert stats["communities"] > 0


def test_parse_citations_ignores_out_of_range():
    from minigraphrag.retrieve import RetrievedChunk

    retrieved = [RetrievedChunk(chunk_id="c0", text="hello", score=1.0, provenance="lexical")]
    citations = parse_citations("claims [1] and [99]", retrieved)
    assert [c.number for c in citations] == [1]


def test_index_pickle_does_not_contain_llm(rag, tmp_path):
    path = tmp_path / "index.pkl"
    rag.save(path)
    raw = path.read_bytes()
    assert b"FakeLLM" not in raw
