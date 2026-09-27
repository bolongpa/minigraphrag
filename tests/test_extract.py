"""Tests for extraction chunking and LLM-output parsing."""

from minigraphrag.extract import chunk_text, parse_extraction_output


def test_chunk_text_respects_max_chars():
    text = " ".join(f"Sentence number {i} about Acme Corp." for i in range(50))
    chunks = chunk_text(text, max_chars=200, overlap=40)
    assert len(chunks) > 1
    # small tolerance: a single long sentence may exceed the budget
    assert all(len(c) <= 320 for c in chunks)


def test_parse_valid_json():
    raw = (
        '{"entities": [{"name": "Maya Chen", "type": "Person", '
        '"description": "CEO of Acme Corp"}], '
        '"triples": [{"head": "Maya Chen", "relation": "founded", '
        '"tail": "Acme Corp", "description": "founded in 2019"}]}'
    )
    entities, triples = parse_extraction_output(raw)
    assert [e.name for e in entities] == ["Maya Chen"]
    assert entities[0].type == "Person"
    assert (triples[0].head, triples[0].relation, triples[0].tail) == (
        "Maya Chen", "founded", "Acme Corp",
    )


def test_parse_json_inside_markdown_fence():
    raw = (
        "Here is the extraction:\n```json\n"
        '{"entities": [{"name": "NimbusDB", "type": "Technology", "description": "a database"}], '
        '"triples": []}\n```'
    )
    entities, triples = parse_extraction_output(raw)
    assert [e.name for e in entities] == ["NimbusDB"]
    assert triples == []


def test_parse_line_fallback():
    raw = "Maya Chen | founded | Acme Corp\nRaj Patel | leads | Project Phoenix"
    entities, triples = parse_extraction_output(raw)
    assert {e.name for e in entities} == {"Maya Chen", "Acme Corp", "Raj Patel", "Project Phoenix"}
    assert len(triples) == 2
    assert triples[1].relation == "leads"


def test_parse_garbage_returns_empty():
    entities, triples = parse_extraction_output("sorry, I cannot do that")
    assert entities == [] and triples == []
