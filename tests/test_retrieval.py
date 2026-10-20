def test_every_chunk_has_a_page(vector_store):
    page_map = vector_store._load_page_map()
    assert page_map and all(page is not None for page in page_map.values())

def test_keyword_search_finds_relevant_passages(vector_store):
    hits = vector_store._keyword_search("primary PCI fibrinolysis STEMI", n_results=5)
    assert len(hits) == 5
    assert any("fibrinolysis" in h["text"].lower() for h in hits)
    assert all(h["page"] for h in hits)

def test_keyword_scores_never_claim_high_confidence(vector_store):
    hits = vector_store._keyword_search("troponin", n_results=5)
    assert all(0 < h["score"] <= 0.8 for h in hits)

def test_search_falls_back_when_openai_is_down(vector_store, openai_down):
    from agents.doc_checker import check_citations
    citations = check_citations("door-to-balloon time", n_results=3)
    assert len(citations) == 3
    assert all(c["page"] != "Unknown" for c in citations)
