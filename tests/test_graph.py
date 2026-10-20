def test_graph_routes_citation_questions_without_claude(vector_store, openai_down):
    from agents.graph import run_cardiobot
    result = run_cardiobot("show me the source for troponin rule-out")
    assert result["type"] == "citation"
    assert result["citations"]

def test_graph_redirects_dosing_questions():
    from agents.graph import run_cardiobot
    result = run_cardiobot("heparin dose for 80 kg")
    assert result["type"] == "dosing_redirect"

def test_graph_sends_clinical_questions_to_claude(vector_store, openai_down, monkeypatch):
    import agents.claude_agent as claude_agent
    monkeypatch.setattr(claude_agent, "ask_claude", lambda *a, **k: "Grounded answer")
    from agents.graph import run_cardiobot
    result = run_cardiobot("What is the treatment for STEMI?")
    assert result["type"] == "rag"
    assert result["answer"] == "Grounded answer"
    assert result["sources"]
