"""
Transcript quality analysis for DevKit2 BLE audio.

Conservative classification of suspicious transcript spans.

Design principles:
- Prefer "mark uncertain" over "rewrite confidently"
- Do NOT flag multilingual switching as an error
- Do NOT flag rough or colloquial speech as invalid
- Do NOT rewrite text, even when confident about corruption
- Preserve all recognized content in canonical text
- Only flag issues where the evidence is strong and machine-detectable
- reading_text excludes only the highest-confidence noise segments;
  everything else stays in both reading and technical views

Issue types produced:
  segmentation_error         — timing inconsistency (overlap, inverted timestamps)
  missing_audio_continuity   — large gap between consecutive segments
  phonetic_hallucination     — implausible speech rate or known junk patterns
  syntax_collapse            — single token dominates a segment at high frequency
  low_information_noise      — ellipsis-only, known Whisper filler, empty text

Continuity hints (Android-provided metadata):
  The optional continuity_hints parameter lets the caller pass chunk-level and
  session-level audio integrity metadata from the Android app.  When present,
  the analysis uses it to:
    - Upgrade confidence on missing_audio_continuity issues when a nearby chunk
      boundary reports decode failures, dropped frames, or explicit gap hints
    - Add upstream_damage_suspected=True to affected issues
    - Emit a source_integrity summary in the report

  The analysis degrades gracefully when continuity_hints is absent or partial:
  all thresholds and logic remain identical.

Analysis version: 1.1
"""
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

ANALYSIS_VERSION = "1.1"

# ---------------------------------------------------------------------------
# Thresholds (tunable constants — not env vars; these are analysis heuristics)
# ---------------------------------------------------------------------------

# Gaps between consecutive segments larger than this are flagged suspicious.
# 5 s is generous — natural spoken pauses are usually < 2 s.
GAP_SUSPICIOUS_S: float = 5.0

# Gaps larger than this are classified as "likely_missing_audio"
# rather than just "possible_silence". 30 s is very conservative.
GAP_LIKELY_MISSING_S: float = 30.0

# Minimum words in a segment before syntax_collapse analysis is attempted.
# Short segments are too noisy to evaluate for structural repetition.
SYNTAX_MIN_WORDS: int = 6

# A single token must account for at least this fraction of the segment's words
# to trigger a syntax_collapse flag.
SYNTAX_REPEAT_FRACTION: float = 0.60

# Repeated-word hallucination: same word N+ times in a row (e.g. "go go go go").
# Set to 4 to avoid flagging legitimate emphatic speech ("no no no").
HALLU_REPEAT_MIN: int = 4

# Implausible speech rate: segment has more than this many words in < 0.5 s.
# Normal fast speech is ~5 words/s; 15 words in 0.5 s = 30 words/s — impossible.
HALLU_MAX_WORDS_IN_HALF_SEC: int = 15

# When correlating chunk boundaries to segment gaps, a chunk boundary within
# this many ms of the gap start is considered "nearby".
_CHUNK_GAP_TOLERANCE_MS: int = 2000

# A chunk reporting gap_before_ms larger than this is considered a meaningful gap
# (not just normal BLE packet timing jitter).
_CHUNK_GAP_BEFORE_SIGNIFICANT_MS: int = 500

# ---------------------------------------------------------------------------
# Known Whisper filler / hallucination patterns (lower-cased, exact match).
# Conservative list: only patterns that are essentially never real wearable
# speech. Multilingual content, informal language, and profanity are NOT here.
# ---------------------------------------------------------------------------
_KNOWN_FILLER_LOWER: frozenset = frozenset({
    "thank you.",
    "thank you",
    "thanks.",
    "thanks for watching.",
    "thanks for watching",
    "...",
    ". . .",
    "\u2026",            # …
    "subtitles by",
    "subtitle by",
    "subscribed",
    "please subscribe",
    "www.",
    "[music]",
    "[applause]",
    "[laughter]",
    "[silence]",
    "[noise]",
    "[no audio]",
    "(no audio)",
    "(silence)",
    "(music)",
})

# Matches segments whose entire content is punctuation / whitespace / ellipsis.
_ELLIPSIS_ONLY_RE = re.compile(
    r'^[\s\.\,\;\:\!\?\/\-\(\)\[\]\u2026\u2018\u2019\u201c\u201d\u2014\u2013\*]+$'
)

# "word word word word" (HALLU_REPEAT_MIN or more consecutive identical tokens).
_REPEATED_WORD_RE = re.compile(
    r'^(\w+)(\s+\1){' + str(HALLU_REPEAT_MIN - 1) + r',}[\.\!\?]?$',
    re.IGNORECASE | re.UNICODE,
)


# ---------------------------------------------------------------------------
# Segment key accessors
# Handles both start_ms/end_ms (API format) and start/end in seconds (worker).
# ---------------------------------------------------------------------------

def _start_ms(seg: Dict) -> int:
    if "start_ms" in seg:
        return int(seg["start_ms"])
    return int(seg.get("start", 0) * 1000)


def _end_ms(seg: Dict) -> int:
    if "end_ms" in seg:
        return int(seg["end_ms"])
    return int(seg.get("end", 0) * 1000)


def _duration_s(seg: Dict) -> float:
    return (_end_ms(seg) - _start_ms(seg)) / 1000.0


# ---------------------------------------------------------------------------
# Continuity hints helpers
# ---------------------------------------------------------------------------

def _parse_continuity_hints(continuity_hints: Optional[Dict]) -> Tuple[
    Dict[int, Dict],   # chunk_boundary_map: started_ms → hint
    Dict,              # session_integrity summary
]:
    """
    Parse continuity_hints into:
      - chunk_boundary_map: maps chunk start time (ms) to integrity data
      - session_integrity: dict summarising session-level degradation

    Both return values are safe to use when continuity_hints is None or incomplete.
    """
    if not continuity_hints:
        return {}, {
            "hints_available": False,
            "session_degraded": False,
            "total_dropped_frames": 0,
            "chunks_with_decode_failure": 0,
            "chunks_with_gaps": 0,
            "integrity_note": None,
        }

    per_chunk: List[Dict] = continuity_hints.get("per_chunk", [])
    chunk_map: Dict[int, Dict] = {}
    chunks_with_decode_failure = 0
    chunks_with_gaps = 0

    for ch in per_chunk:
        started_ms = int(ch.get("chunk_started_ms", 0))
        duration_ms = int(ch.get("chunk_duration_ms", 0))
        decode_fail = bool(ch.get("decode_failure", False))
        dropped = int(ch.get("dropped_frames", 0))
        gap_before = int(ch.get("gap_before_ms", 0))
        degraded = bool(ch.get("source_degraded", False))

        if decode_fail:
            chunks_with_decode_failure += 1
        if gap_before > _CHUNK_GAP_BEFORE_SIGNIFICANT_MS:
            chunks_with_gaps += 1

        chunk_map[started_ms] = {
            "chunk_index": ch.get("chunk_index", -1),
            "started_ms": started_ms,
            "end_ms": started_ms + duration_ms,
            "dropped_frames": dropped,
            "decode_failure": decode_fail,
            "gap_before_ms": gap_before,
            "source_degraded": degraded,
        }

    # Read session-level integrity from the hints dict itself,
    # or from a nested "session_integrity" sub-dict (both conventions supported).
    si = continuity_hints.get("session_integrity") or continuity_hints
    session_degraded = bool(si.get("session_degraded", False))
    total_dropped = int(si.get("total_dropped_frames", 0))
    integrity_note = si.get("integrity_note") or None

    session_integrity = {
        "hints_available": True,
        "session_degraded": session_degraded,
        "total_dropped_frames": total_dropped,
        "chunks_with_decode_failure": chunks_with_decode_failure,
        "chunks_with_gaps": chunks_with_gaps,
        "integrity_note": integrity_note,
    }

    return chunk_map, session_integrity


def _find_upstream_evidence_near_gap(
    gap_start_ms: int,
    chunk_map: Dict[int, Dict],
) -> Optional[str]:
    """
    Look for a chunk with damage indicators whose boundary is near the gap.

    Returns a human-readable evidence string if found, else None.
    The caller uses this to add upstream_damage_suspected=True and upgrade
    confidence on missing_audio_continuity issues.
    """
    if not chunk_map:
        return None

    for started_ms, hint in chunk_map.items():
        chunk_end_ms = hint["end_ms"]

        # "Near" means the chunk end, start, or span overlaps the gap ±tolerance
        near = (
            abs(chunk_end_ms - gap_start_ms) < _CHUNK_GAP_TOLERANCE_MS
            or abs(started_ms - gap_start_ms) < _CHUNK_GAP_TOLERANCE_MS
            or (started_ms < gap_start_ms < chunk_end_ms + _CHUNK_GAP_TOLERANCE_MS)
        )
        if not near:
            continue

        reasons = []
        if hint["decode_failure"]:
            reasons.append(
                f"chunk {hint['chunk_index']} reported decode failure"
            )
        if hint["dropped_frames"] > 0:
            reasons.append(
                f"chunk {hint['chunk_index']} dropped {hint['dropped_frames']} frames"
            )
        if hint["gap_before_ms"] > _CHUNK_GAP_BEFORE_SIGNIFICANT_MS:
            reasons.append(
                f"chunk {hint['chunk_index']} reported "
                f"{hint['gap_before_ms']}ms gap before it"
            )
        if hint["source_degraded"]:
            reasons.append(f"chunk {hint['chunk_index']} marked source_degraded")

        if reasons:
            return "; ".join(reasons)

    return None


# ---------------------------------------------------------------------------
# Per-segment classifiers — each returns (triggered: bool, reason: str)
# ---------------------------------------------------------------------------

def _check_low_information(text: str) -> Tuple[bool, str]:
    """High-confidence junk detection: ellipsis-only, known fillers, empty."""
    t = text.strip()
    if not t:
        return True, "Empty segment text"
    if _ELLIPSIS_ONLY_RE.match(t):
        return True, "Segment contains only punctuation or ellipsis characters"
    if t.lower() in _KNOWN_FILLER_LOWER:
        return True, f"Known Whisper filler/hallucination pattern: {t!r}"
    # Single non-alphabetic token (punctuation, number, etc.)
    if len(t) == 1 and not t.isalpha():
        return True, "Single non-alphabetic character"
    return False, ""


def _check_phonetic_hallucination(text: str, duration_s: float) -> Tuple[bool, str]:
    """Conservative: only flag very clear acoustic implausibility."""
    t = text.strip()
    if not t:
        return False, ""
    # Repeated single word: "go go go go go"
    if _REPEATED_WORD_RE.match(t):
        return True, f"Single word repeated {HALLU_REPEAT_MIN}+ times consecutively"
    # Physically impossible speech rate
    words = t.split()
    if (
        len(words) > HALLU_MAX_WORDS_IN_HALF_SEC
        and 0 < duration_s < 0.5
    ):
        return True, (
            f"Implausible speech rate: {len(words)} words in {duration_s:.2f}s "
            f"(max plausible ~{HALLU_MAX_WORDS_IN_HALF_SEC} words / 0.5s)"
        )
    return False, ""


def _check_syntax_collapse(text: str) -> Tuple[bool, str]:
    """Medium-confidence: a single token dominates the segment by frequency."""
    t = text.strip()
    words = t.split()
    if len(words) < SYNTAX_MIN_WORDS:
        return False, ""
    freq: Dict[str, int] = {}
    for w in words:
        wn = w.lower().strip(".,!?;:'\"-")
        if wn:
            freq[wn] = freq.get(wn, 0) + 1
    if not freq:
        return False, ""
    most_common = max(freq, key=freq.get)  # type: ignore[arg-type]
    count = freq[most_common]
    if count / len(words) > SYNTAX_REPEAT_FRACTION:
        return True, (
            f"Token '{most_common}' accounts for {count}/{len(words)} "
            f"words ({count / len(words):.0%}) in segment"
        )
    return False, ""


def _check_segmentation_error(
    seg: Dict, prev_seg: Optional[Dict]
) -> Tuple[bool, str]:
    """High-confidence timing consistency check."""
    s = _start_ms(seg)
    e = _end_ms(seg)
    if e < s:
        return True, f"Inverted timestamps: start={s}ms end={e}ms"
    if prev_seg is not None:
        prev_end = _end_ms(prev_seg)
        overlap_ms = prev_end - s
        if overlap_ms > 50:   # 50 ms tolerance for rounding errors
            return True, f"Overlaps previous segment by {overlap_ms}ms"
    return False, ""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_transcript_quality(
    segments: List[Dict],
    chunk_count: int = 0,
    total_audio_ms: int = 0,
    continuity_hints: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Analyze a finalized transcript segment list and return a structured quality
    report.  Input segments may use start_ms/end_ms (ms ints) or start/end
    (seconds floats) — both are handled transparently.

    Does NOT modify input.  Does NOT rewrite any text.

    Args:
        segments:           list of segment dicts
        chunk_count:        number of audio chunks at finalize time
        total_audio_ms:     total declared audio duration from chunk metadata
        continuity_hints:   optional dict from Android app with chunk-level and
                            session-level audio integrity metadata.  Structure:
                            {
                              "per_chunk": [
                                {
                                  "chunk_index": int,
                                  "chunk_started_ms": int,
                                  "chunk_duration_ms": int,
                                  "dropped_frames": int,   (default 0)
                                  "decode_failure": bool,  (default false)
                                  "gap_before_ms": int,    (default 0)
                                  "source_degraded": bool, (default false)
                                }
                              ],
                              "session_integrity": {          (optional)
                                "session_degraded": bool,
                                "total_dropped_frames": int,
                                "integrity_note": str,
                              }
                            }
                            Absent or null: analysis proceeds without hints,
                            source_integrity.hints_available = False.

    Returns dict with keys:
        analysis_version, analyzed_at,
        segment_count, flagged_segment_count, clean_segment_count,
        issues:                    list of issue dicts
        suspected_missing_intervals: list of gap dicts
        continuity:                aggregate timing stats
        source_integrity:          upstream damage summary (hints_available=False if absent)
        reading_text:              canonical text with high-confidence noise removed
    """
    # Parse continuity hints
    chunk_map, source_integrity = _parse_continuity_hints(continuity_hints)

    # Work on a deterministically ordered copy
    ordered: List[Dict] = sorted(segments, key=_start_ms)

    issues: List[Dict[str, Any]] = []
    suspected_missing_intervals: List[Dict[str, Any]] = []
    flagged_indices: Set[int] = set()

    for i, seg in enumerate(ordered):
        text = seg.get("text", "").strip()
        s_ms = _start_ms(seg)
        e_ms = _end_ms(seg)
        dur_s = _duration_s(seg)
        prev = ordered[i - 1] if i > 0 else None

        # ----------------------------------------------------------------
        # 1. Segmentation timing errors
        # ----------------------------------------------------------------
        is_seg_err, seg_err_reason = _check_segmentation_error(seg, prev)
        if is_seg_err:
            issues.append({
                "type": "segmentation_error",
                "confidence": "high",
                "segment_index": i,
                "start_ms": s_ms,
                "end_ms": e_ms,
                "text": text,
                "reason": seg_err_reason,
                "upstream_damage_suspected": False,
            })
            flagged_indices.add(i)

        # ----------------------------------------------------------------
        # 2. Missing audio continuity (gap analysis)
        #    Note: this classifies the *interval*, not the segment's text.
        #    We do NOT add the segment to flagged_indices — it may be
        #    perfectly valid speech that follows a missing chunk.
        # ----------------------------------------------------------------
        if prev is not None:
            gap_s = (s_ms - _end_ms(prev)) / 1000.0
            if gap_s > GAP_SUSPICIOUS_S:
                assessment = (
                    "likely_missing_audio"
                    if gap_s > GAP_LIKELY_MISSING_S
                    else "possible_silence_or_missing_audio"
                )

                # Check if chunk-level damage evidence is near this gap
                upstream_evidence = _find_upstream_evidence_near_gap(
                    gap_start_ms=s_ms,
                    chunk_map=chunk_map,
                )
                upstream_suspected = upstream_evidence is not None

                # Upgrade confidence when chunk damage confirms the gap
                if upstream_suspected:
                    confidence = "high" if gap_s > GAP_LIKELY_MISSING_S else "medium"
                else:
                    confidence = (
                        "medium" if gap_s > GAP_LIKELY_MISSING_S else "low"
                    )

                gap_reason = (
                    f"Gap of {gap_s:.1f}s before this segment "
                    f"({assessment.replace('_', ' ')})"
                )
                if upstream_evidence:
                    gap_reason += f". Upstream damage evidence: {upstream_evidence}"

                suspected_missing_intervals.append({
                    "after_segment_index": i - 1,
                    "gap_start_ms": _end_ms(prev),
                    "gap_end_ms": s_ms,
                    "gap_s": round(gap_s, 2),
                    "assessment": assessment,
                    "upstream_damage_suspected": upstream_suspected,
                })
                issues.append({
                    "type": "missing_audio_continuity",
                    "confidence": confidence,
                    "segment_index": i,
                    "start_ms": s_ms,
                    "end_ms": e_ms,
                    "text": text,
                    "reason": gap_reason,
                    "gap_s": round(gap_s, 2),
                    "upstream_damage_suspected": upstream_suspected,
                })

        # ----------------------------------------------------------------
        # 3. Low information noise (high-confidence junk)
        # ----------------------------------------------------------------
        is_noise, noise_reason = _check_low_information(text)
        if is_noise:
            issues.append({
                "type": "low_information_noise",
                "confidence": "high",
                "segment_index": i,
                "start_ms": s_ms,
                "end_ms": e_ms,
                "text": text,
                "reason": noise_reason,
                "upstream_damage_suspected": False,
            })
            flagged_indices.add(i)
            # No further checks on junk text — results would be meaningless
            continue

        # ----------------------------------------------------------------
        # 4. Phonetic hallucination
        # ----------------------------------------------------------------
        is_hallu, hallu_reason = _check_phonetic_hallucination(text, dur_s)
        if is_hallu:
            issues.append({
                "type": "phonetic_hallucination",
                "confidence": "medium",
                "segment_index": i,
                "start_ms": s_ms,
                "end_ms": e_ms,
                "text": text,
                "reason": hallu_reason,
                "upstream_damage_suspected": False,
            })
            flagged_indices.add(i)
            continue

        # ----------------------------------------------------------------
        # 5. Syntax collapse
        # ----------------------------------------------------------------
        is_collapsed, collapse_reason = _check_syntax_collapse(text)
        if is_collapsed:
            issues.append({
                "type": "syntax_collapse",
                "confidence": "medium",
                "segment_index": i,
                "start_ms": s_ms,
                "end_ms": e_ms,
                "text": text,
                "reason": collapse_reason,
                "upstream_damage_suspected": False,
            })
            flagged_indices.add(i)

    # ----------------------------------------------------------------
    # reading_text: full text minus *only* high-confidence noise segments.
    # We intentionally do NOT exclude phonetic_hallucination or
    # syntax_collapse — false-positive exclusion of real speech is worse
    # than including a slightly noisy token.  The distinction between
    # reading_text and text gives the client the choice.
    # ----------------------------------------------------------------
    noise_only_indices: Set[int] = {
        iss["segment_index"]
        for iss in issues
        if iss["type"] == "low_information_noise" and iss["confidence"] == "high"
    }
    reading_parts = [
        ordered[i].get("text", "").strip()
        for i in range(len(ordered))
        if i not in noise_only_indices
    ]
    reading_text = " ".join(p for p in reading_parts if p)

    # ----------------------------------------------------------------
    # Continuity aggregates
    # ----------------------------------------------------------------
    transcribed_ms = sum(
        max(0, _end_ms(s) - _start_ms(s)) for s in ordered
    )
    total_span_ms = 0
    if ordered:
        total_span_ms = _end_ms(ordered[-1]) - _start_ms(ordered[0])

    gap_count = len(suspected_missing_intervals)
    large_gap_count = sum(
        1 for iv in suspected_missing_intervals
        if iv["gap_s"] > GAP_LIKELY_MISSING_S
    )

    return {
        "analysis_version": ANALYSIS_VERSION,
        "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "segment_count": len(ordered),
        "flagged_segment_count": len(flagged_indices),
        "clean_segment_count": len(ordered) - len(flagged_indices),
        "issues": issues,
        "suspected_missing_intervals": suspected_missing_intervals,
        "continuity": {
            "total_span_s": round(total_span_ms / 1000.0, 2),
            "transcribed_duration_s": round(transcribed_ms / 1000.0, 2),
            "gap_count": gap_count,
            "large_gap_count": large_gap_count,
            "chunk_count_at_finalize": chunk_count,
            "declared_audio_ms": total_audio_ms,
        },
        "source_integrity": source_integrity,
        "reading_text": reading_text,
    }
