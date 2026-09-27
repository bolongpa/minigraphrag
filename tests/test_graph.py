"""Tests for graph construction and community detection."""

from minigraphrag.extract import ChunkExtraction, Entity, Triple
from minigraphrag.graph import build_graph, detect_communities


def _sample_extractions():
    return [
        ChunkExtraction(
            chunk_id="doc0-chunk0",
            entities=[
                Entity(name="Maya Chen", type="Person", description="CEO of Acme Corp"),
                Entity(name="Acme Corp", type="Organization", description="a software company"),
            ],
            triples=[Triple(head="Maya Chen", relation="founded", tail="Acme Corp")],
        ),
        ChunkExtraction(
            chunk_id="doc1-chunk0",
            entities=[
                Entity(name="Raj Patel", type="Person", description="leads Project Phoenix"),
                Entity(name="Project Phoenix", type="Project", description="modernization effort"),
                Entity(name="Priya Nair", type="Person", description="mentors Raj Patel"),
            ],
            triples=[
                Triple(head="Raj Patel", relation="leads", tail="Project Phoenix"),
                Triple(head="Priya Nair", relation="mentors", tail="Raj Patel"),
            ],
        ),
    ]


def test_build_graph_nodes_and_edges():
    graph = build_graph(_sample_extractions())
    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 3
    assert graph.nodes["Maya Chen"]["type"] == "Person"
    assert graph["Maya Chen"]["Acme Corp"]["relation"] == "founded"
    assert graph.nodes["Raj Patel"]["chunks"] == {"doc1-chunk0"}


def test_build_graph_merges_duplicate_mentions():
    exts = _sample_extractions()
    exts.append(
        ChunkExtraction(
            chunk_id="doc2-chunk0",
            entities=[Entity(name="Maya Chen", type="Person", description="founder")],
            triples=[],
        )
    )
    graph = build_graph(exts)
    assert graph.nodes["Maya Chen"]["chunks"] == {"doc0-chunk0", "doc2-chunk0"}
    assert "founder" in graph.nodes["Maya Chen"]["description"]


def test_detect_communities_covers_all_nodes():
    graph = build_graph(_sample_extractions())
    communities = detect_communities(graph)
    covered = {n for c in communities for n in c.node_names}
    assert covered == set(graph.nodes)
    assert len(communities) == 2  # two disconnected components
    assert all(graph.nodes[n]["community"] == c.id for c in communities for n in c.node_names)


def test_isolated_node_becomes_singleton_community():
    exts = _sample_extractions()
    exts.append(ChunkExtraction(chunk_id="doc3-chunk0", entities=[Entity(name="Lonely Entity")]))
    graph = build_graph(exts)
    communities = detect_communities(graph)
    singletons = [c for c in communities if c.node_names == ["Lonely Entity"]]
    assert len(singletons) == 1
