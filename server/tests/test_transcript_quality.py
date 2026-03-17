"""
Pure unit tests for server/transcript_quality.py and server/clean_transcript.py.

These tests run without any server, Redis, or GPU.  They verify that the
quality analysis module classifies transcripts correctly, stays conservative,
and never flags valid multilingual or rough-but-real speech.

They also verify that the clean transcript generator follows the design rules:
  - never invents content
  - only excludes high-confidence noise
  - marks uncertain spans with a prefix but preserves original text
  - merges adjacent segments correctly
  - produces correct paragraph breaks

Usage:
    cd server
    python -m pytest tests/test_transcript_quality.py -v
"""
import sys
import os

# Allow importing from the server directory even when pytest is invoked from
# the repo root or from server/tests/.
_SERVER_DIR = os.path.join(os.path.dirname(__file__), "..")
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from transcript_quality import (
    analyze_transcript_quality,
    ANALYSIS_VERSION,
    GAP_SUSPICIOUS_S,
    GAP_LIKELY_MISSING_S,
    _check_low_information,
    _check_phonetic_hallucination,
    _check_syntax_collapse,
    _check_segmentation_error,
    _parse_continuity_hints,
    _find_upstream_evidence_near_gap,
)
from clean_transcript import (
    build_clean_transcript,
    GENERATOR_VERSION,
    MERGE_GAP_S,
    PARAGRAPH_GAP_S,
    _UNCERTAIN_PREFIX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seg(start_ms: int, end_ms: int, text: str, speaker: str = "SPEAKER_00") -> dict:
    """Build a segment dict in API format (start_ms / end_ms in ms)."""
    return {"start_ms": start_ms, "end_ms": end_ms, "text": text, "speaker": speaker}


def seg_s(start: float, end: float, text: str) -> dict:
    """Build a segment dict in worker format (start / end in seconds)."""
    return {"start": start, "end": end, "text": text}


def issues_of_type(report: dict, issue_type: str) -> list:
    return [i for i in report["issues"] if i["type"] == issue_type]


# ---------------------------------------------------------------------------
# Low-information noise classifier
# ---------------------------------------------------------------------------

class TestLowInformationClassifier:
    def test_empty_text(self):
        flagged, _ = _check_low_information("")
        assert flagged

    def test_ellipsis_only(self):
        for t in ["...", ". . .", "\u2026", "...\u2026", "---", "???"]:
            flagged, reason = _check_low_information(t)
            assert flagged, f"Expected ellipsis-only to be flagged: {t!r}"
            assert reason

    def test_known_filler_patterns(self):
        for t in ["Thank you.", "thank you", "...", "[music]", "[silence]", "(no audio)"]:
            flagged, reason = _check_low_information(t)
            assert flagged, f"Expected known filler to be flagged: {t!r}"

    def test_real_speech_not_flagged(self):
        for t in [
            "Hello, how are you?",
            "Да, это хорошо.",          # Russian — valid multilingual
            "Je suis fatigué.",           # French — valid multilingual
            "No.",                        # Very short but real word
            "OK let's go",               # Casual speech
            "Shit, I forgot my keys.",    # Profanity — still real speech
        ]:
            flagged, _ = _check_low_information(t)
            assert not flagged, f"Real speech should not be flagged: {t!r}"

    def test_single_nonalpha_char(self):
        flagged, _ = _check_low_information("?")
        assert flagged
        flagged2, _ = _check_low_information("a")
        assert not flagged2


# ---------------------------------------------------------------------------
# Phonetic hallucination classifier
# ---------------------------------------------------------------------------

class TestPhoneticHallucinationClassifier:
    def test_repeated_word_pattern(self):
        for t in ["go go go go", "yes yes yes yes yes", "ching ching ching ching"]:
            flagged, reason = _check_phonetic_hallucination(t, duration_s=1.0)
            assert flagged, f"Repeated word should be flagged: {t!r}"
            assert reason

    def test_implausible_speech_rate(self):
        # 20 words in 0.2 s — physically impossible
        long_text = " ".join(["word"] * 20)
        flagged, reason = _check_phonetic_hallucination(long_text, duration_s=0.2)
        assert flagged
        assert "rate" in reason.lower() or "implausible" in reason.lower()

    def test_normal_speech_rate_not_flagged(self):
        # 10 words in 3 s — perfectly normal
        flagged, _ = _check_phonetic_hallucination(
            "This is a normal sentence of about ten words.", duration_s=3.0
        )
        assert not flagged

    def test_multilingual_not_flagged(self):
        # "Нет нет" (No no in Russian) — only 2 repetitions, fine
        flagged, _ = _check_phonetic_hallucination("Нет нет, это неправильно.", 1.5)
        assert not flagged

    def test_three_repetitions_not_flagged(self):
        # "no no no" — emphatic but plausible speech, threshold is 4
        flagged, _ = _check_phonetic_hallucination("no no no", 1.0)
        assert not flagged

    def test_four_repetitions_flagged(self):
        flagged, _ = _check_phonetic_hallucination("no no no no", 1.0)
        assert flagged


# ---------------------------------------------------------------------------
# Syntax collapse classifier
# ---------------------------------------------------------------------------

class TestSyntaxCollapseClassifier:
    def test_high_repetition_flagged(self):
        # "the the the the the" — 5 of 5 words are "the"
        flagged, reason = _check_syntax_collapse("the the the the the the")
        assert flagged
        assert "the" in reason

    def test_normal_text_not_flagged(self):
        for t in [
            "I went to the store to buy some milk.",
            "The quick brown fox jumps over the lazy dog.",
            "Я думаю что это хорошая идея.",
        ]:
            flagged, _ = _check_syntax_collapse(t)
            assert not flagged, f"Normal text should not be flagged: {t!r}"

    def test_short_segment_skipped(self):
        # Less than SYNTAX_MIN_WORDS — not enough data to analyse
        flagged, _ = _check_syntax_collapse("hello world")
        assert not flagged

    def test_moderate_repetition_not_flagged(self):
        # "the" appears 3/10 words (30%) — below 60% threshold
        text = "I went to the store and the baker gave the bread."
        flagged, _ = _check_syntax_collapse(text)
        assert not flagged


# ---------------------------------------------------------------------------
# Segmentation error classifier
# ---------------------------------------------------------------------------

class TestSegmentationErrorClassifier:
    def test_inverted_timestamps(self):
        s = seg(5000, 2000, "hello")
        flagged, reason = _check_segmentation_error(s, None)
        assert flagged
        assert "inverted" in reason.lower()

    def test_overlap_with_previous(self):
        prev = seg(0, 5000, "first")
        cur = seg(4000, 7000, "second")  # starts 1000ms before prev ends
        flagged, reason = _check_segmentation_error(cur, prev)
        assert flagged
        assert "overlap" in reason.lower()

    def test_small_rounding_overlap_tolerated(self):
        # 30ms overlap — within 50ms tolerance
        prev = seg(0, 5000, "first")
        cur = seg(4980, 7000, "second")
        flagged, _ = _check_segmentation_error(cur, prev)
        assert not flagged

    def test_normal_segment_not_flagged(self):
        prev = seg(0, 3000, "first")
        cur = seg(3500, 6000, "second")
        flagged, _ = _check_segmentation_error(cur, prev)
        assert not flagged


# ---------------------------------------------------------------------------
# Full analyze_transcript_quality
# ---------------------------------------------------------------------------

class TestAnalyzeTranscriptQuality:

    def test_empty_segments_returns_clean_report(self):
        report = analyze_transcript_quality([])
        assert report["segment_count"] == 0
        assert report["flagged_segment_count"] == 0
        assert report["issues"] == []
        assert report["suspected_missing_intervals"] == []
        assert report["reading_text"] == ""
        assert report["analysis_version"] == ANALYSIS_VERSION

    def test_clean_segments_not_flagged(self):
        segments = [
            seg(0, 2000, "Good morning, how are you?"),
            seg(2500, 5000, "I'm doing well, thank you."),
            seg(5500, 8000, "Let's start the meeting."),
        ]
        report = analyze_transcript_quality(segments)
        assert report["flagged_segment_count"] == 0
        assert report["issues"] == []
        assert report["segment_count"] == 3

    def test_multilingual_not_flagged(self):
        """Valid multilingual speech must never be flagged."""
        segments = [
            seg(0, 2000, "Hello, nice to meet you."),
            seg(2500, 4500, "Привет, рад познакомиться."),  # Russian
            seg(5000, 7000, "Bonjour, enchanté."),           # French
            seg(7500, 9500, "Hola, mucho gusto."),           # Spanish
        ]
        report = analyze_transcript_quality(segments)
        assert report["flagged_segment_count"] == 0, (
            f"Multilingual segments should not be flagged. Issues: {report['issues']}"
        )

    def test_colloquial_rough_speech_not_flagged(self):
        """Abrupt, colloquial, or vulgar speech is real speech — must not be rewritten."""
        segments = [
            seg(0, 1000, "Shit."),
            seg(1500, 3000, "Yeah, whatever."),
            seg(3500, 5000, "I dunno, man."),
            seg(5500, 7000, "No way."),
        ]
        report = analyze_transcript_quality(segments)
        assert report["flagged_segment_count"] == 0, (
            f"Colloquial speech should not be flagged. Issues: {report['issues']}"
        )

    def test_short_meaningful_utterances_not_filtered(self):
        """Short but semantically valid utterances must survive quality filtering."""
        segments = [
            seg(0, 500, "Yes."),
            seg(1000, 1500, "No."),
            seg(2000, 2800, "OK."),
            seg(3000, 4000, "Wait."),
        ]
        report = analyze_transcript_quality(segments)
        # "OK." and "Yes." are real speech (single alpha word)
        noise_issues = issues_of_type(report, "low_information_noise")
        # None of these single-word utterances should be flagged as noise
        flagged_texts = {i["text"] for i in noise_issues}
        for text in ["Yes.", "No.", "OK.", "Wait."]:
            assert text not in flagged_texts, (
                f"Short meaningful utterance {text!r} was incorrectly flagged as noise"
            )

    def test_low_information_noise_detected(self):
        segments = [
            seg(0, 2000, "This is real speech."),
            seg(2500, 3000, "..."),           # junk
            seg(3500, 4000, "Thank you."),    # known filler
            seg(4500, 6500, "More real content here."),
        ]
        report = analyze_transcript_quality(segments)
        noise = issues_of_type(report, "low_information_noise")
        assert len(noise) == 2, f"Expected 2 noise issues, got {noise}"

    def test_reading_text_excludes_noise(self):
        segments = [
            seg(0, 2000, "This is real speech."),
            seg(2500, 3000, "..."),
            seg(3500, 5000, "More real content here."),
        ]
        report = analyze_transcript_quality(segments)
        assert "..." not in report["reading_text"]
        assert "This is real speech." in report["reading_text"]
        assert "More real content here." in report["reading_text"]

    def test_reading_text_keeps_hallucination_segments(self):
        """
        phonetic_hallucination is NOT excluded from reading_text —
        false-positive exclusion of real speech is worse than keeping a noisy segment.
        """
        segments = [
            seg(0, 2000, "Let's go."),
            seg(2500, 3500, "go go go go go"),  # hallucination flag
            seg(4000, 6000, "We are ready."),
        ]
        report = analyze_transcript_quality(segments)
        hallu = issues_of_type(report, "phonetic_hallucination")
        assert len(hallu) == 1
        # But it should still appear in reading_text
        assert "go go go go go" in report["reading_text"]

    def test_phonetic_hallucination_detected(self):
        segments = [
            seg(0, 2000, "The meeting starts now."),
            seg(2500, 3500, "ching ching ching ching ching"),
        ]
        report = analyze_transcript_quality(segments)
        hallu = issues_of_type(report, "phonetic_hallucination")
        assert len(hallu) >= 1

    def test_syntax_collapse_detected(self):
        segments = [
            seg(0, 2000, "Normal sentence here."),
            # "the" repeats 7/10 times — above 60% threshold
            seg(2500, 5000, "the the the the the the the way to go now"),
        ]
        report = analyze_transcript_quality(segments)
        collapse = issues_of_type(report, "syntax_collapse")
        assert len(collapse) >= 1

    def test_missing_audio_continuity_small_gap_not_flagged(self):
        """4 s gap is below GAP_SUSPICIOUS_S (5 s) — should not be flagged."""
        segments = [
            seg(0, 2000, "First sentence."),
            seg(6000, 8000, "Second sentence after 4s gap."),  # 4s gap
        ]
        report = analyze_transcript_quality(segments)
        continuity = issues_of_type(report, "missing_audio_continuity")
        assert len(continuity) == 0

    def test_missing_audio_continuity_large_gap_flagged(self):
        """Gap clearly above GAP_SUSPICIOUS_S must produce a continuity issue."""
        gap_s = GAP_SUSPICIOUS_S + 10.0
        segments = [
            seg(0, 2000, "First sentence."),
            seg(int((2.0 + gap_s) * 1000), int((2.0 + gap_s + 2.0) * 1000),
                "Second sentence after big gap."),
        ]
        report = analyze_transcript_quality(segments)
        continuity = issues_of_type(report, "missing_audio_continuity")
        assert len(continuity) >= 1
        assert len(report["suspected_missing_intervals"]) >= 1

    def test_likely_missing_audio_large_gap(self):
        """Gap > GAP_LIKELY_MISSING_S should be classified 'likely_missing_audio'."""
        gap_s = GAP_LIKELY_MISSING_S + 5.0
        segments = [
            seg(0, 2000, "Before the gap."),
            seg(int((2.0 + gap_s) * 1000), int((2.0 + gap_s + 3.0) * 1000),
                "After the likely missing audio."),
        ]
        report = analyze_transcript_quality(segments)
        intervals = report["suspected_missing_intervals"]
        assert intervals[0]["assessment"] == "likely_missing_audio"

    def test_segmentation_error_inverted_timestamps(self):
        segments = [
            seg(0, 2000, "Normal."),
            seg(5000, 3000, "Inverted timestamps."),  # end < start
        ]
        report = analyze_transcript_quality(segments)
        seg_errors = issues_of_type(report, "segmentation_error")
        assert len(seg_errors) >= 1

    def test_continuity_metadata(self):
        segments = [
            seg(0, 2000, "First."),
            seg(2500, 4500, "Second."),
        ]
        report = analyze_transcript_quality(segments, chunk_count=5, total_audio_ms=8000)
        c = report["continuity"]
        assert c["chunk_count_at_finalize"] == 5
        assert c["declared_audio_ms"] == 8000
        assert c["gap_count"] == 0
        import pytest
        assert c["transcribed_duration_s"] == pytest.approx(4.0, abs=0.01)

    def test_accepts_worker_format_seconds(self):
        """Worker format uses start/end in seconds — must be handled correctly."""
        segments = [
            seg_s(0.0, 2.0, "Hello."),
            seg_s(2.5, 4.5, "World."),
        ]
        report = analyze_transcript_quality(segments)
        assert report["segment_count"] == 2
        assert report["flagged_segment_count"] == 0

    def test_canonical_text_not_dropped(self):
        """
        All non-noise text should appear in the reading_text in the correct order.
        """
        segments = [
            seg(0, 1000, "Alpha"),
            seg(1500, 2500, "Beta"),
            seg(3000, 4000, "Gamma"),
        ]
        report = analyze_transcript_quality(segments)
        reading = report["reading_text"]
        # All three words must be present and in order
        idx_a = reading.index("Alpha")
        idx_b = reading.index("Beta")
        idx_g = reading.index("Gamma")
        assert idx_a < idx_b < idx_g

    def test_report_structure_complete(self):
        """Quality report must contain all required top-level keys."""
        report = analyze_transcript_quality([seg(0, 2000, "Test.")])
        required_keys = {
            "analysis_version", "analyzed_at",
            "segment_count", "flagged_segment_count", "clean_segment_count",
            "issues", "suspected_missing_intervals", "continuity",
            "source_integrity", "reading_text",
        }
        missing = required_keys - set(report.keys())
        assert not missing, f"Missing keys in quality report: {missing}"

    def test_continuity_structure_complete(self):
        report = analyze_transcript_quality([seg(0, 2000, "Test.")])
        continuity_keys = {
            "total_span_s", "transcribed_duration_s",
            "gap_count", "large_gap_count",
            "chunk_count_at_finalize", "declared_audio_ms",
        }
        missing = continuity_keys - set(report["continuity"].keys())
        assert not missing, f"Missing continuity keys: {missing}"

    def test_source_integrity_structure_no_hints(self):
        """source_integrity must be present even without continuity hints."""
        report = analyze_transcript_quality([seg(0, 2000, "Test.")])
        si = report["source_integrity"]
        assert si is not None
        assert si["hints_available"] is False
        assert si["session_degraded"] is False
        assert si["total_dropped_frames"] == 0

    def test_issue_structure_complete(self):
        """Every issue must have all required fields."""
        segments = [seg(0, 2000, "...")]  # noise segment → will produce an issue
        report = analyze_transcript_quality(segments)
        assert report["issues"]
        for issue in report["issues"]:
            for key in ("type", "confidence", "segment_index", "start_ms", "end_ms",
                        "text", "reason", "upstream_damage_suspected"):
                assert key in issue, f"Issue missing key '{key}': {issue}"

    def test_multiple_issue_types_combined(self):
        """A session with mixed problems should produce multiple issue types."""
        big_gap_start = int((2.0 + GAP_SUSPICIOUS_S + 5.0) * 1000)
        segments = [
            seg(0, 2000, "Normal start of meeting."),
            seg(2500, 3000, "..."),                              # low_information_noise
            seg(big_gap_start, big_gap_start + 2000,
                "After a big gap."),                             # missing_audio_continuity
            seg(big_gap_start + 3000, big_gap_start + 5000,
                "la la la la la la la la la la la la"),          # syntax_collapse
        ]
        report = analyze_transcript_quality(segments)
        types_found = {i["type"] for i in report["issues"]}
        assert "low_information_noise" in types_found
        assert "missing_audio_continuity" in types_found
        assert "syntax_collapse" in types_found


# ---------------------------------------------------------------------------
# Continuity hints: _parse_continuity_hints and _find_upstream_evidence_near_gap
# ---------------------------------------------------------------------------

class TestContinuityHintsParsing:

    def test_no_hints_returns_safe_defaults(self):
        chunk_map, si = _parse_continuity_hints(None)
        assert chunk_map == {}
        assert si["hints_available"] is False
        assert si["session_degraded"] is False
        assert si["total_dropped_frames"] == 0
        assert si["chunks_with_decode_failure"] == 0
        assert si["chunks_with_gaps"] == 0
        assert si["integrity_note"] is None

    def test_empty_hints_returns_safe_defaults(self):
        chunk_map, si = _parse_continuity_hints({})
        assert si["hints_available"] is False

    def test_per_chunk_decode_failure_counted(self):
        hints = {
            "per_chunk": [
                {"chunk_index": 0, "chunk_started_ms": 0, "chunk_duration_ms": 5000,
                 "decode_failure": True, "dropped_frames": 0, "gap_before_ms": 0},
                {"chunk_index": 1, "chunk_started_ms": 5000, "chunk_duration_ms": 5000,
                 "decode_failure": False, "dropped_frames": 10, "gap_before_ms": 0},
            ]
        }
        chunk_map, si = _parse_continuity_hints(hints)
        assert si["hints_available"] is True
        assert si["chunks_with_decode_failure"] == 1
        assert 0 in chunk_map
        assert chunk_map[0]["decode_failure"] is True
        assert 5000 in chunk_map
        assert chunk_map[5000]["dropped_frames"] == 10

    def test_session_integrity_parsed(self):
        hints = {
            "session_integrity": {
                "session_degraded": True,
                "total_dropped_frames": 42,
                "integrity_note": "BLE signal unstable",
            }
        }
        _, si = _parse_continuity_hints(hints)
        assert si["session_degraded"] is True
        assert si["total_dropped_frames"] == 42
        assert si["integrity_note"] == "BLE signal unstable"

    def test_gap_before_significant_counted(self):
        hints = {
            "per_chunk": [
                {"chunk_index": 2, "chunk_started_ms": 10000, "chunk_duration_ms": 5000,
                 "gap_before_ms": 2000},  # > 500ms threshold
                {"chunk_index": 3, "chunk_started_ms": 15000, "chunk_duration_ms": 5000,
                 "gap_before_ms": 100},   # below threshold
            ]
        }
        _, si = _parse_continuity_hints(hints)
        assert si["chunks_with_gaps"] == 1


class TestUpstreamEvidenceSearch:

    def test_no_chunk_map_returns_none(self):
        result = _find_upstream_evidence_near_gap(5000, {})
        assert result is None

    def test_decode_failure_near_gap_detected(self):
        chunk_map = {
            4000: {  # started_ms=4000, end_ms=4000+5000=9000
                "chunk_index": 1,
                "started_ms": 4000,
                "end_ms": 9000,
                "decode_failure": True,
                "dropped_frames": 0,
                "gap_before_ms": 0,
                "source_degraded": False,
            }
        }
        # gap_start_ms=8000 is within tolerance of end_ms=9000
        result = _find_upstream_evidence_near_gap(8000, chunk_map)
        assert result is not None
        assert "decode failure" in result

    def test_dropped_frames_near_gap_detected(self):
        chunk_map = {
            5000: {
                "chunk_index": 2,
                "started_ms": 5000,
                "end_ms": 10000,
                "decode_failure": False,
                "dropped_frames": 7,
                "gap_before_ms": 0,
                "source_degraded": False,
            }
        }
        result = _find_upstream_evidence_near_gap(9500, chunk_map)
        assert result is not None
        assert "dropped 7 frames" in result

    def test_clean_chunk_near_gap_returns_none(self):
        chunk_map = {
            5000: {
                "chunk_index": 2,
                "started_ms": 5000,
                "end_ms": 10000,
                "decode_failure": False,
                "dropped_frames": 0,
                "gap_before_ms": 0,
                "source_degraded": False,
            }
        }
        result = _find_upstream_evidence_near_gap(9500, chunk_map)
        assert result is None

    def test_chunk_far_from_gap_returns_none(self):
        chunk_map = {
            0: {
                "chunk_index": 0,
                "started_ms": 0,
                "end_ms": 5000,
                "decode_failure": True,
                "dropped_frames": 10,
                "gap_before_ms": 0,
                "source_degraded": False,
            }
        }
        # gap at 60_000ms, chunk ends at 5000ms — far beyond tolerance
        result = _find_upstream_evidence_near_gap(60_000, chunk_map)
        assert result is None


class TestContinuityHintsIntegration:
    """Integration tests: continuity hints passed to analyze_transcript_quality."""

    def test_hints_improve_confidence_on_gap_issue(self):
        """Decode failure near a gap should upgrade confidence from low to medium."""
        gap_ms = int((GAP_SUSPICIOUS_S + 10.0) * 1000)  # 15s gap — > suspicious threshold
        segments = [
            seg(0, 2000, "Before gap."),
            seg(2000 + gap_ms, 2000 + gap_ms + 2000, "After gap."),
        ]
        # A chunk that ends right at the gap start, with decode failure
        continuity_hints = {
            "per_chunk": [
                {"chunk_index": 0, "chunk_started_ms": 0, "chunk_duration_ms": 2000,
                 "decode_failure": False, "dropped_frames": 0, "gap_before_ms": 0},
                {"chunk_index": 1, "chunk_started_ms": 2000, "chunk_duration_ms": gap_ms,
                 "decode_failure": True, "dropped_frames": 5, "gap_before_ms": 0},
            ]
        }

        report_no_hints = analyze_transcript_quality(segments)
        report_with_hints = analyze_transcript_quality(segments, continuity_hints=continuity_hints)

        # Both should flag missing_audio_continuity
        continuity_no = issues_of_type(report_no_hints, "missing_audio_continuity")
        continuity_with = issues_of_type(report_with_hints, "missing_audio_continuity")
        assert len(continuity_no) >= 1
        assert len(continuity_with) >= 1

        # With hints: upstream_damage_suspected should be True
        assert continuity_with[0]["upstream_damage_suspected"] is True
        # Without hints: upstream_damage_suspected should be False
        assert continuity_no[0]["upstream_damage_suspected"] is False

    def test_source_integrity_present_with_hints(self):
        hints = {
            "session_integrity": {
                "session_degraded": True,
                "total_dropped_frames": 100,
                "integrity_note": "poor BLE"
            }
        }
        report = analyze_transcript_quality([seg(0, 2000, "Test.")], continuity_hints=hints)
        si = report["source_integrity"]
        assert si["hints_available"] is True
        assert si["session_degraded"] is True
        assert si["total_dropped_frames"] == 100
        assert si["integrity_note"] == "poor BLE"

    def test_no_hints_upstream_damage_false(self):
        """Without hints, no issue should have upstream_damage_suspected=True."""
        gap_ms = int((GAP_SUSPICIOUS_S + 10.0) * 1000)
        segments = [
            seg(0, 2000, "Before."),
            seg(2000 + gap_ms, 2000 + gap_ms + 2000, "After."),
        ]
        report = analyze_transcript_quality(segments)
        for issue in report["issues"]:
            assert issue.get("upstream_damage_suspected") is False


# ---------------------------------------------------------------------------
# Clean transcript generator
# ---------------------------------------------------------------------------

class TestBuildCleanTranscript:

    def test_empty_input(self):
        result = build_clean_transcript([])
        assert result["clean_text"] == ""
        assert result["paragraphs"] == []
        assert result["excluded_count"] == 0
        assert result["uncertainty_count"] == 0
        assert result["merge_count"] == 0
        assert result["generator"] == GENERATOR_VERSION

    def test_clean_segments_pass_through(self):
        segs = [
            seg(0, 2000, "Good morning."),
            seg(5000, 7000, "How are you?"),
        ]
        result = build_clean_transcript(segs)
        assert "Good morning." in result["clean_text"]
        assert "How are you?" in result["clean_text"]
        assert result["excluded_count"] == 0
        assert result["uncertainty_count"] == 0

    def test_noise_excluded_from_clean_text(self):
        """High-confidence noise must be excluded from clean_text."""
        segs = [
            seg(0, 2000, "Real speech here."),
            seg(2500, 3000, "..."),           # noise — index 1 in quality report
            seg(3500, 5500, "More real speech."),
        ]
        # Build a quality report that flags index 1 as low_information_noise/high
        quality_report = {
            "issues": [
                {
                    "type": "low_information_noise",
                    "confidence": "high",
                    "segment_index": 1,
                    "text": "...",
                    "reason": "ellipsis",
                }
            ]
        }
        result = build_clean_transcript(segs, quality_report)
        assert "..." not in result["clean_text"]
        assert "Real speech here." in result["clean_text"]
        assert "More real speech." in result["clean_text"]
        assert result["excluded_count"] == 1
        assert 1 in result["excluded_segment_indices"]

    def test_low_confidence_noise_not_excluded(self):
        """Low-confidence issues must NOT cause exclusion — conservative rule."""
        segs = [
            seg(0, 2000, "Maybe noisy but keep."),
        ]
        quality_report = {
            "issues": [
                {
                    "type": "low_information_noise",
                    "confidence": "low",   # low confidence — must not exclude
                    "segment_index": 0,
                    "text": "Maybe noisy but keep.",
                    "reason": "uncertain",
                }
            ]
        }
        result = build_clean_transcript(segs, quality_report)
        assert "Maybe noisy but keep." in result["clean_text"]
        assert result["excluded_count"] == 0

    def test_hallucination_preserved_with_uncertain_prefix(self):
        """phonetic_hallucination segments are preserved but prefixed [uncertain]."""
        segs = [
            seg(0, 2000, "Normal speech."),
            seg(3000, 4000, "go go go go go"),
        ]
        quality_report = {
            "issues": [
                {
                    "type": "phonetic_hallucination",
                    "confidence": "medium",
                    "segment_index": 1,
                    "text": "go go go go go",
                    "reason": "repeated word",
                }
            ]
        }
        result = build_clean_transcript(segs, quality_report)
        # Original text preserved
        assert "go go go go go" in result["clean_text"]
        # Uncertainty marker prepended
        assert _UNCERTAIN_PREFIX in result["clean_text"]
        assert result["uncertainty_count"] == 1
        assert result["excluded_count"] == 0
        # Marker captured in uncertainty_markers list
        assert any(m["original_text"] == "go go go go go" for m in result["uncertainty_markers"])

    def test_syntax_collapse_preserved_with_uncertain_prefix(self):
        segs = [
            seg(0, 2000, "Normal."),
            seg(3000, 5000, "the the the the the the the"),
        ]
        quality_report = {
            "issues": [
                {
                    "type": "syntax_collapse",
                    "confidence": "medium",
                    "segment_index": 1,
                    "text": "the the the the the the the",
                    "reason": "high repetition",
                }
            ]
        }
        result = build_clean_transcript(segs, quality_report)
        assert "the the the the the the the" in result["clean_text"]
        assert _UNCERTAIN_PREFIX in result["clean_text"]
        assert result["uncertainty_count"] == 1

    def test_low_confidence_hallucination_not_marked(self):
        """Low-confidence issues must NOT be marked uncertain."""
        segs = [seg(0, 2000, "Maybe real.")]
        quality_report = {
            "issues": [
                {
                    "type": "phonetic_hallucination",
                    "confidence": "low",   # low confidence — no marker
                    "segment_index": 0,
                    "text": "Maybe real.",
                    "reason": "borderline",
                }
            ]
        }
        result = build_clean_transcript(segs, quality_report)
        assert _UNCERTAIN_PREFIX not in result["clean_text"]
        assert result["uncertainty_count"] == 0

    def test_adjacent_segments_merged(self):
        """Segments within MERGE_GAP_S of each other should be merged into one span."""
        gap_ms = int(MERGE_GAP_S * 1000 * 0.5)  # well within threshold
        segs = [
            seg(0, 1000, "First part"),
            seg(1000 + gap_ms, 2000 + gap_ms, "second part"),
        ]
        result = build_clean_transcript(segs)
        assert result["merge_count"] == 1
        # Both parts in the same paragraph (merged)
        assert len(result["paragraphs"]) == 1
        assert "First part" in result["paragraphs"][0]["text"]
        assert "second part" in result["paragraphs"][0]["text"]

    def test_large_gap_creates_paragraph_break(self):
        """Gap > PARAGRAPH_GAP_S must produce a new paragraph."""
        gap_ms = int(PARAGRAPH_GAP_S * 1000 * 2)  # well above threshold
        segs = [
            seg(0, 1000, "First paragraph content."),
            seg(1000 + gap_ms, 2000 + gap_ms, "Second paragraph content."),
        ]
        result = build_clean_transcript(segs)
        assert len(result["paragraphs"]) == 2
        assert result["paragraphs"][0]["text"] == "First paragraph content."
        assert result["paragraphs"][1]["text"] == "Second paragraph content."
        assert "\n\n" in result["clean_text"]

    def test_no_quality_report_is_safe(self):
        """Passing quality_report=None must not raise or exclude anything."""
        segs = [
            seg(0, 1000, "Alpha."),
            seg(5000, 6000, "Beta."),
        ]
        result = build_clean_transcript(segs, quality_report=None)
        assert "Alpha." in result["clean_text"]
        assert "Beta." in result["clean_text"]
        assert result["excluded_count"] == 0
        assert result["uncertainty_count"] == 0

    def test_output_structure_complete(self):
        result = build_clean_transcript([seg(0, 2000, "Test.")])
        required = {
            "clean_text", "paragraphs", "excluded_segment_indices",
            "uncertainty_markers", "merge_count", "excluded_count",
            "uncertainty_count", "generator", "generated_at",
        }
        missing = required - set(result.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_worker_format_seconds_handled(self):
        segs = [
            seg_s(0.0, 2.0, "Hello."),
            seg_s(5.0, 7.0, "World."),
        ]
        result = build_clean_transcript(segs)
        assert "Hello." in result["clean_text"]
        assert "World." in result["clean_text"]

    def test_content_never_invented(self):
        """clean_text must be a strict subset of the input texts."""
        segs = [
            seg(0, 2000, "Actual content here."),
            seg(10000, 12000, "More content."),
        ]
        result = build_clean_transcript(segs)
        clean = result["clean_text"]
        # No new words appear that aren't in the original texts (minus the uncertain prefix)
        clean_stripped = clean.replace(_UNCERTAIN_PREFIX, "")
        input_words = set("Actual content here. More content.".lower().split())
        output_words = set(clean_stripped.lower().split())
        invented = output_words - input_words
        assert not invented, f"Invented words in clean_text: {invented}"

    def test_multilingual_content_preserved(self):
        """Multilingual speech must pass through untouched."""
        segs = [
            seg(0, 2000, "Hello."),
            seg(3000, 5000, "Привет."),   # Russian
            seg(6000, 8000, "Bonjour."),  # French
        ]
        result = build_clean_transcript(segs)
        assert "Привет." in result["clean_text"]
        assert "Bonjour." in result["clean_text"]


# ---------------------------------------------------------------------------
# Backward compat: existing fields must be unaffected
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_analysis_version_constant(self):
        assert isinstance(ANALYSIS_VERSION, str)
        # Version bumped to 1.1 to reflect continuity hints addition
        assert ANALYSIS_VERSION == "1.1"

    def test_empty_input_does_not_raise(self):
        report = analyze_transcript_quality([])
        assert report is not None

    def test_none_text_handled_gracefully(self):
        # Segment with no 'text' key
        segments = [{"start_ms": 0, "end_ms": 2000}]
        report = analyze_transcript_quality(segments)
        assert report["segment_count"] == 1

    def test_old_call_signature_still_works(self):
        """Calling without continuity_hints must work identically to before."""
        segs = [
            seg(0, 2000, "Hello."),
            seg(2500, 4500, "World."),
        ]
        report = analyze_transcript_quality(segs, chunk_count=2, total_audio_ms=5000)
        assert report["segment_count"] == 2
        assert report["flagged_segment_count"] == 0
        # source_integrity always present, hints_available=False
        assert report["source_integrity"]["hints_available"] is False


# ---------------------------------------------------------------------------
# Allow `pytest` marker import even if pytest is not installed at import time
# ---------------------------------------------------------------------------
try:
    import pytest
except ImportError:
    pass
