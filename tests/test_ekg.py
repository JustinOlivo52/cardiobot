import base64
import io
import pytest
from PIL import Image
import agents.ekg_agent as ekg_agent
from tests.conftest import Unavailable
from tools.image_tool import prepare_for_vision, MAX_VISION_EDGE
from utils.logger import CardioImageError

class FakeUpload(io.BytesIO):
    """Mimics a Streamlit UploadedFile."""
    def __init__(self, data, media_type):
        super().__init__(data)
        self.type = media_type

def make_image(size, fmt="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format=fmt)
    return buffer.getvalue()

def test_large_photo_is_shrunk_and_converted_to_jpeg():
    encoded, media_type = prepare_for_vision(FakeUpload(make_image((4032, 3024)), "image/png"))
    image = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert media_type == "image/jpeg"
    assert max(image.size) == MAX_VISION_EDGE

def test_bmp_is_converted_for_providers_that_reject_it():
    _, media_type = prepare_for_vision(FakeUpload(make_image((800, 600), "BMP"), "image/bmp"))
    assert media_type == "image/jpeg"

def test_upload_can_be_read_after_streamlit_displays_it():
    upload = FakeUpload(make_image((800, 600)), "image/png")
    upload.read()  # st.image consumes the buffer before the button is clicked
    encoded, _ = prepare_for_vision(upload)
    assert encoded

@pytest.fixture
def readers(monkeypatch):
    calls = []
    def install(claude, gemini):
        def wrap(name, behavior):
            def read(encoded, media_type):
                calls.append(name)
                if isinstance(behavior, Exception):
                    raise behavior
                return behavior
            return read
        monkeypatch.setattr(ekg_agent, "READERS", {
            "claude": ("claude-model", wrap("claude", claude)),
            "gemini": ("gemini-model", wrap("gemini", gemini)),
        })
        return calls
    return install

def upload():
    return FakeUpload(make_image((800, 600)), "image/png")

def test_claude_reads_first(readers):
    calls = readers("Sinus rhythm", "unused")
    result = ekg_agent.interpret_ekg(upload())
    assert result["model"] == "claude-model"
    assert calls == ["claude"]

def test_falls_back_to_gemini_when_claude_fails(readers):
    calls = readers(Unavailable("Error code: 529 - overloaded"), "Sinus rhythm")
    result = ekg_agent.interpret_ekg(upload())
    assert result["model"] == "gemini-model"
    assert calls == ["claude", "gemini"]

def test_empty_reply_counts_as_a_failure(readers):
    readers("   ", "Sinus rhythm")
    assert ekg_agent.interpret_ekg(upload())["model"] == "gemini-model"

def test_clear_error_when_every_provider_fails(readers):
    readers(Unavailable("overloaded"), Unavailable("429 RESOURCE_EXHAUSTED"))
    with pytest.raises(CardioImageError, match="RESOURCE_EXHAUSTED"):
        ekg_agent.interpret_ekg(upload())

def test_claude_request_includes_the_image(monkeypatch):
    sent = {}
    monkeypatch.setattr(ekg_agent, "ask_claude", lambda system, messages, **kw: sent.update(messages=messages) or "ok")
    ekg_agent._read_with_claude("abc123", "image/jpeg")
    image_block = sent["messages"][0]["content"][0]
    assert image_block["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "abc123"}
