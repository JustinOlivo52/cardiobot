import pytest
from utils.logger import friendly_error

@pytest.mark.parametrize("raw, expected", [
    ("Error code: 429 - You exceeded your current quota", "usage limit was reached"),
    ("Error code: 401 - invalid x-api-key", "API key was rejected"),
    ("Connection error.", "temporarily unreachable"),
    ("something odd", "unexpected error"),
])
def test_friendly_error_explains_the_cause(raw, expected):
    message = friendly_error(Exception(raw), "Citation search")
    assert message.startswith("Citation search is temporarily unavailable")
    assert expected in message

def test_friendly_error_hides_raw_details():
    message = friendly_error(Exception("Error code: 401 - sk-secret-key-123"), "Consult")
    assert "sk-secret" not in message
