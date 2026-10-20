from tests.conftest import Unavailable

def test_dosing_falls_back_to_claude(vector_store, openai_down, monkeypatch):
    import agents.dosing_agent as dosing_agent
    monkeypatch.setattr(dosing_agent, "ask_claude", lambda *a, **k: "Claude guidance")
    result = dosing_agent.get_dosing_guidance("heparin", 70)
    assert result["guidance"] == "Claude guidance"
    assert result["model"] == dosing_agent.CLAUDE_MODEL

def test_dosing_still_works_with_every_ai_provider_down(openai_down, monkeypatch):
    import agents.dosing_agent as dosing_agent

    def fail(*args, **kwargs):
        raise Unavailable("Error code: 529 - overloaded")

    monkeypatch.setattr(dosing_agent, "ask_claude", fail)
    result = dosing_agent.get_dosing_guidance("heparin", 100)
    assert result["calculated_dose"] == "5000.0 units (max dose)"
    assert "temporarily unavailable" in result["guidance"]
    assert "offline" in result["model"]
