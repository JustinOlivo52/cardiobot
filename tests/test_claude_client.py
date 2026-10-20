from types import SimpleNamespace
import pytest
import agents.claude_client as claude_client
from utils.logger import CardioAPIError

def fake_response(stop_reason, blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)

def test_reads_text_and_skips_thinking_blocks(monkeypatch):
    blocks = [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="Aspirin and a P2Y12 inhibitor."),
    ]
    monkeypatch.setattr(claude_client.client.beta.messages, "create",
                        lambda **kw: fake_response("end_turn", blocks))
    assert claude_client.ask_claude("sys", [{"role": "user", "content": "q"}]) == "Aspirin and a P2Y12 inhibitor."

def test_refusal_raises_a_clear_error(monkeypatch):
    monkeypatch.setattr(claude_client.client.beta.messages, "create",
                        lambda **kw: fake_response("refusal", []))
    with pytest.raises(CardioAPIError, match="declined"):
        claude_client.ask_claude("sys", [{"role": "user", "content": "q"}])

def test_request_uses_configured_model_and_effort(monkeypatch):
    sent = {}
    def capture(**kwargs):
        sent.update(kwargs)
        return fake_response("end_turn", [SimpleNamespace(type="text", text="ok")])
    monkeypatch.setattr(claude_client.client.beta.messages, "create", capture)
    claude_client.ask_claude("sys", [{"role": "user", "content": "q"}], effort="medium")
    assert sent["model"] == claude_client.CLAUDE_MODEL
    assert sent["output_config"] == {"effort": "medium"}
