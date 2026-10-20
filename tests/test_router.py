import pytest
from agents.router import route_query

@pytest.mark.parametrize("query, route", [
    ("heparin dose for 80 kg", "dosing"),
    ("how much amiodarone in arrest", "dosing"),
    ("can you interpret this ekg strip", "ekg"),
    ("show me the source for troponin timing", "citation"),
    # General clinical questions should get an answer, not a redirect
    ("What medications are given in STEMI?", "rag"),
    ("Treatment for afib with RVR", "rag"),
    ("What is the recommended door-to-balloon time?", "rag"),
    # Whole-word matching: "kg" inside "background" must not trigger dosing
    ("background on NSTEMI", "rag"),
])
def test_route_query(query, route):
    assert route_query(query) == route
