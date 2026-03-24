"""
Unit tests for transcript_classifier.py.

These tests verify prompt construction, response parsing, and configuration
handling WITHOUT requiring an actual GGUF model or llama-cpp-python installed.
"""
import json
import os
import sys
import pytest

# Ensure server modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transcript_classifier import (
    _build_prompt,
    _parse_response,
    classify_transcript,
    VALID_CATEGORIES,
    CLASSIFIER_CONTEXT_SIZE,
    CLASSIFIER_MAX_TOKENS,
    _PROMPT_TEMPLATE_TOKENS,
    _CHARS_PER_TOKEN_ESTIMATE,
)


# ---------------------------------------------------------------------------
# TestPromptConstruction
# ---------------------------------------------------------------------------

class TestPromptConstruction:

    def test_prompt_contains_all_categories(self):
        prompt = _build_prompt("Hello world, this is a test transcript.")
        for cat in VALID_CATEGORIES:
            assert cat in prompt, f"Prompt missing category: {cat}"

    def test_prompt_contains_transcript_text(self):
        text = "This is a very specific unique test sentence."
        prompt = _build_prompt(text)
        assert text in prompt

    def test_prompt_truncates_long_transcript(self):
        # Build a transcript longer than the context window allows
        available_tokens = CLASSIFIER_CONTEXT_SIZE - CLASSIFIER_MAX_TOKENS - _PROMPT_TEMPLATE_TOKENS
        max_chars = int(available_tokens * _CHARS_PER_TOKEN_ESTIMATE)
        long_text = "word " * (max_chars // 5 + 1000)
        prompt = _build_prompt(long_text)
        assert "[...truncated]" in prompt

    def test_prompt_does_not_truncate_short_transcript(self):
        short_text = "Short meeting about the project."
        prompt = _build_prompt(short_text)
        assert "[...truncated]" not in prompt

    def test_prompt_requests_json_format(self):
        prompt = _build_prompt("Some text")
        assert '"category"' in prompt
        assert '"confidence"' in prompt
        assert '"rationale"' in prompt

    def test_prompt_handles_empty_string(self):
        prompt = _build_prompt("")
        # Should still produce a valid prompt structure
        assert "Classify" in prompt


# ---------------------------------------------------------------------------
# TestResponseParsing
# ---------------------------------------------------------------------------

class TestResponseParsing:

    def test_valid_meeting_response(self):
        raw = '{"category": "meeting", "confidence": 0.85, "rationale": "Multiple speakers discussing project."}'
        result = _parse_response(raw)
        assert result is not None
        assert result["category"] == "meeting"
        assert result["confidence"] == 0.85
        assert "speakers" in result["rationale"]

    def test_valid_technical_response(self):
        raw = '{"category": "technical", "confidence": 0.9, "rationale": "Code review discussion."}'
        result = _parse_response(raw)
        assert result is not None
        assert result["category"] == "technical"

    def test_valid_self_reflection_response(self):
        raw = '{"category": "self_reflection", "confidence": 0.7, "rationale": "Personal journaling."}'
        result = _parse_response(raw)
        assert result is not None
        assert result["category"] == "self_reflection"

    def test_valid_self_therapy_response(self):
        raw = '{"category": "self_therapy", "confidence": 0.6, "rationale": "Emotional processing."}'
        result = _parse_response(raw)
        assert result is not None
        assert result["category"] == "self_therapy"

    def test_json_with_surrounding_text(self):
        raw = 'Here is the classification:\n{"category": "meeting", "confidence": 0.8, "rationale": "A team standup."}\nDone.'
        result = _parse_response(raw)
        assert result is not None
        assert result["category"] == "meeting"

    def test_invalid_category_returns_none(self):
        raw = '{"category": "podcast", "confidence": 0.9, "rationale": "A podcast."}'
        result = _parse_response(raw)
        assert result is None

    def test_no_json_returns_none(self):
        raw = "This is just plain text with no JSON."
        result = _parse_response(raw)
        assert result is None

    def test_broken_json_returns_none(self):
        raw = '{"category": "meeting", confidence: 0.8}'
        result = _parse_response(raw)
        assert result is None

    def test_missing_category_returns_none(self):
        raw = '{"confidence": 0.9, "rationale": "Some reason."}'
        result = _parse_response(raw)
        assert result is None

    def test_confidence_clamped_to_0_1(self):
        raw = '{"category": "technical", "confidence": 1.5, "rationale": "Overconfigent."}'
        result = _parse_response(raw)
        assert result is not None
        assert result["confidence"] == 1.0

    def test_negative_confidence_clamped(self):
        raw = '{"category": "technical", "confidence": -0.5, "rationale": "Negative."}'
        result = _parse_response(raw)
        assert result is not None
        assert result["confidence"] == 0.0

    def test_missing_confidence_defaults_to_0_5(self):
        raw = '{"category": "meeting", "rationale": "A meeting."}'
        result = _parse_response(raw)
        assert result is not None
        assert result["confidence"] == 0.5

    def test_rationale_truncated_to_500_chars(self):
        long_rationale = "x" * 1000
        raw = json.dumps({"category": "meeting", "confidence": 0.8, "rationale": long_rationale})
        result = _parse_response(raw)
        assert result is not None
        assert len(result["rationale"]) == 500

    def test_hyphenated_category_normalized(self):
        """Categories with hyphens should be normalized to underscores."""
        raw = '{"category": "self-reflection", "confidence": 0.7, "rationale": "Thinking."}'
        result = _parse_response(raw)
        assert result is not None
        assert result["category"] == "self_reflection"

    def test_spaced_category_normalized(self):
        """Categories with spaces should be normalized to underscores."""
        raw = '{"category": "self therapy", "confidence": 0.7, "rationale": "Feelings."}'
        result = _parse_response(raw)
        assert result is not None
        assert result["category"] == "self_therapy"


# ---------------------------------------------------------------------------
# TestClassifyTranscriptFallbacks
# ---------------------------------------------------------------------------

class TestClassifyTranscriptFallbacks:
    """
    Test that classify_transcript returns None or a structured skip/fail dict
    gracefully in various misconfigured or unavailable scenarios.
    """

    def test_returns_none_when_disabled(self, monkeypatch):
        import transcript_classifier as tc
        monkeypatch.setattr(tc, "CLASSIFIER_ENABLED", False)
        result = tc.classify_transcript("Some transcript text here.")
        assert result is None

    def test_returns_none_when_no_model_path(self, monkeypatch):
        import transcript_classifier as tc
        monkeypatch.setattr(tc, "CLASSIFIER_ENABLED", True)
        monkeypatch.setattr(tc, "CLASSIFIER_MODEL_PATH", "")
        result = tc.classify_transcript("Some transcript text here.")
        assert result is None

    def test_returns_skipped_when_model_file_missing(self, monkeypatch):
        import transcript_classifier as tc
        monkeypatch.setattr(tc, "CLASSIFIER_ENABLED", True)
        monkeypatch.setattr(tc, "CLASSIFIER_MODEL_PATH", "/nonexistent/model.gguf")
        # Reset cached instance so it tries to load
        monkeypatch.setattr(tc, "_llm_instance", None)
        monkeypatch.setattr(tc, "_llm_model_path", None)
        result = tc.classify_transcript("Some transcript text here.")
        assert result is not None
        assert result["status"] == "skipped"
        assert result["skip_reason"] == "model_load_failed"

    def test_returns_skipped_for_empty_transcript(self, monkeypatch):
        import transcript_classifier as tc
        monkeypatch.setattr(tc, "CLASSIFIER_ENABLED", True)
        monkeypatch.setattr(tc, "CLASSIFIER_MODEL_PATH", "/some/path.gguf")
        result = tc.classify_transcript("")
        assert result is not None
        assert result["status"] == "skipped"
        assert result["skip_reason"] == "empty_transcript"

    def test_returns_skipped_for_whitespace_transcript(self, monkeypatch):
        import transcript_classifier as tc
        monkeypatch.setattr(tc, "CLASSIFIER_ENABLED", True)
        monkeypatch.setattr(tc, "CLASSIFIER_MODEL_PATH", "/some/path.gguf")
        result = tc.classify_transcript("   \n  \t  ")
        assert result is not None
        assert result["status"] == "skipped"
        assert result["skip_reason"] == "empty_transcript"
