"""Knowledge-graph construction, community detection, and summarization.

Triples extracted per chunk become a :mod:`networkx` graph:

* **nodes** — entities, with ``type``, ``description``, and the set of chunk
  ids that mention them;
* **edges** — relations, with ``relation``, ``description``, and source chunks.

Communities are found with greedy modularity maximization
(:func:`networkx.algorithms.community.greedy_modularity_communities`), then
each community is summarized by the LLM. Those summaries are what let the
query path answer "global" questions (e.g. "what are the main themes?") that
no single chunk covers — the core GraphRAG idea.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from .extract import ChunkExtraction
from .llm import LLMClient

COMMUNITY_PROMPT = """TASK: SUMMARIZE_COMMUNITY

You are summarizing one community of a knowledge graph built from a document
corpus. Write a concise 3-5 sentence summary capturing what this community is
about, the key entities, and how they relate. Ground everything in the facts
below; do not invent new facts.

Entities and relations in this community:
{facts}
"""


@dataclass
class Community:
    id: int
    node_names: list[str]
    summary: str = ""


def _merge_description(existing: str, new: str) -> str:
    if not new:
        return existing
    if not existing:
        return new
    if new in existing:
        return existing
    return f"{existing} {new}"


def build_graph(extractions: list[ChunkExtraction]) -> nx.Graph:
    """Build a knowledge graph from per-chunk extractions."""
    graph: nx.Graph = nx.Graph()
    for ext in extractions:
        for ent in ext.entities:
            if graph.has_node(ent.name):
                node = graph.nodes[ent.name]
                node["description"] = _merge_description(node.get("description", ""), ent.description)
                if not node.get("type") or node["type"] == "Concept":
                    node["type"] = ent.type
                node.setdefault("chunks", set()).add(ext.chunk_id)
            else:
                graph.add_node(
                    ent.name,
                    type=ent.type,
                    description=ent.description,
                    chunks={ext.chunk_id},
                )
        for triple in ext.triples:
            for name in (triple.head, triple.tail):
                if not graph.has_node(name):
                    graph.add_node(name, type="Concept", description="", chunks={ext.chunk_id})
            if graph.has_edge(triple.head, triple.tail):
                edge = graph[triple.head][triple.tail]
                edge["description"] = _merge_description(edge.get("description", ""), triple.description)
                edge.setdefault("chunks", set()).add(ext.chunk_id)
                # keep the most recently seen relation label if unset
                edge.setdefault("relation", triple.relation)
            else:
                graph.add_edge(
                    triple.head,
                    triple.tail,
                    relation=triple.relation,
                    description=triple.description,
                    chunks={ext.chunk_id},
                )
            # the triple's endpoints were both observed in this chunk
            graph.nodes[triple.head].setdefault("chunks", set()).add(ext.chunk_id)
            graph.nodes[triple.tail].setdefault("chunks", set()).add(ext.chunk_id)
    return graph


def detect_communities(graph: nx.Graph) -> list[Community]:
    """Partition the graph into communities via greedy modularity.

    Isolated nodes (no edges) each become their own singleton community so no
    entity is ever dropped from the summary layer.
    """
    communities: list[Community] = []
    if graph.number_of_nodes() == 0:
        return communities
    subgraph = graph.subgraph([n for n in graph.nodes if graph.degree(n) > 0])
    comm_id = 0
    if subgraph.number_of_nodes() > 0:
        for members in greedy_modularity_communities(subgraph):
            names = sorted(members)
            communities.append(Community(id=comm_id, node_names=names))
            for name in names:
                graph.nodes[name]["community"] = comm_id
            comm_id += 1
    for node in graph.nodes:
        if graph.degree(node) == 0:
            graph.nodes[node]["community"] = comm_id
            communities.append(Community(id=comm_id, node_names=[node]))
            comm_id += 1
    return communities


def _community_facts(graph: nx.Graph, community: Community, max_facts: int = 40) -> str:
    lines: list[str] = []
    for name in community.node_names:
        node = graph.nodes[name]
        desc = node.get("description", "")
        lines.append(f"- {name} ({node.get('type', 'Concept')}): {desc}".rstrip(": "))
        if len(lines) >= max_facts:
            break
    for head, tail in graph.edges(community.node_names):
        if head in community.node_names and tail in community.node_names:
            edge = graph[head][tail]
            lines.append(f"- {head} --[{edge.get('relation', 'related_to')}]--> {tail}")
            if len(lines) >= max_facts:
                break
    return "\n".join(lines[:max_facts])


def summarize_communities(
    graph: nx.Graph, communities: list[Community], llm: LLMClient
) -> list[Community]:
    """Ask the LLM to summarize each community; mutates ``summary`` in place."""
    for community in communities:
        facts = _community_facts(graph, community)
        prompt = COMMUNITY_PROMPT.format(facts=facts)
        community.summary = llm.complete(prompt).strip()
    return communities
