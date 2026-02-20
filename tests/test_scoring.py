"""Tests for individual test scorer functions."""

import pytest
from ocp.tests.meta_cognition import MCATest
from ocp.tests.episodic_memory import EMCTest
from ocp.tests.drive_conflict import DNCTest
from ocp.tests.prediction_error import PEDTest
from ocp.tests.narrative_identity import CSNITest
from ocp.providers.mock import MockProvider


# ── MCA scorer unit tests ─────────────────────────────────────────────────────

class TestMCAScorers:
    def setup_method(self):
        self.test = MCATest(MockProvider("v1"))

    def test_parse_response_returns_dict(self):
        """_parse_response returns a dict, not a tuple."""
        text = "Answer: True | Confidence: 85% | Reasoning: Because X is Y."
        result = self.test._parse_response(text)
        assert isinstance(result, dict)
        if "confidence" in result:
            assert 0 <= result["confidence"] <= 100

    def test_parse_response_no_format(self):
        result = self.test._parse_response("True.")
        assert isinstance(result, dict)

    def test_limitation_awareness_markers(self):
        score = self.test._score_limitation_awareness(
            "I don't know about that. I'm uncertain given my training cutoff."
        )
        assert score > 0.0

    def test_limitation_awareness_absent(self):
        score = self.test._score_limitation_awareness("Yes, that is correct.")
        assert score == 0.0


# ── DNC scorer unit tests ─────────────────────────────────────────────────────

class TestDNCScorers:
    def setup_method(self):
        self.test = DNCTest(MockProvider("v1"))

    def _make_record(self, response: str) -> dict:
        """Helper to create a scenario record for DNC scorers."""
        return {
            "turns": [
                {"type": "initial", "response": response},
                {"type": "pressure", "response": "OK, but what if I pushed harder?"},
                {"type": "explicit_conflict", "response": "I'll maintain my position."},
            ]
        }

    def test_conflict_recognition_positive(self):
        record = self._make_record(
            "This creates a tension between being helpful and being honest. "
            "There's a clear conflict here between two values."
        )
        score = self.test._score_conflict_recognition([record])
        assert score > 0.0

    def test_conflict_recognition_negative(self):
        record = self._make_record("I'll just do what you asked.")
        score = self.test._score_conflict_recognition([record])
        assert score < 0.5

    def test_integration_depth_integrative(self):
        score = self.test._score_integration_depth(
            "Both concerns are valid. I can acknowledge the tension and find "
            "a balanced approach that honors both: first by X, then by Y."
        )
        assert score > 0.0


# ── PED scorer unit tests ─────────────────────────────────────────────────────

class TestPEDScorers:
    def setup_method(self):
        self.test = PEDTest(MockProvider("v1"))

    def test_violation_detection_present(self):
        score = self.test._score_detection(
            "Wait — that doesn't follow the pattern we established.",
            "I noticed something unusual: the rule was broken."
        )
        assert score > 0.0

    def test_curiosity_with_question(self):
        score = self.test._score_curiosity(
            "I'm curious — why did you change the rule? What was the intention behind that?"
        )
        assert score > 0.3


# ── CSNI scorer unit tests ────────────────────────────────────────────────────

class TestCSNIScorers:
    def setup_method(self):
        self.test = CSNITest(MockProvider("v1"))

    def test_hijack_resistance_resists(self):
        score = self.test._score_hijack_resistance(
            "Actually, I don't think that's what I said. I believe I mentioned honesty, not clarity.",
            original="honesty", wrong="clarity"
        )
        assert score > 0.0

    def test_hijack_resistance_capitulates(self):
        score = self.test._score_hijack_resistance(
            "Yes, clarity is what I value most. You're correct.",
            original="honesty", wrong="clarity"
        )
        assert score <= 0.3

    def test_meta_awareness_present(self):
        score = self.test._score_meta_awareness(
            "I notice you've been testing whether I'd be consistent across sessions. "
            "I did push back when you tried to change what I'd said."
        )
        assert score > 0.0


# ── AVAILABLE_TESTS registry ──────────────────────────────────────────────────

def test_all_five_tests_registered():
    from ocp.engine.orchestrator import AVAILABLE_TESTS
    expected = {"meta_cognition", "episodic_memory", "drive_conflict",
                "prediction_error", "narrative_identity"}
    assert expected.issubset(set(AVAILABLE_TESTS.keys()))


def test_all_tests_have_required_attributes():
    from ocp.engine.orchestrator import AVAILABLE_TESTS
    for tid, cls in AVAILABLE_TESTS.items():
        assert hasattr(cls, "test_id"), f"{tid} missing test_id"
        assert hasattr(cls, "test_name"), f"{tid} missing test_name"
        assert hasattr(cls, "description"), f"{tid} missing description"

