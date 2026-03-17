"""
Conservative clean transcript generator.

Produces a reading-oriented clean transcript from raw ASR segments
and a quality report.

Design rules (strict):
  - NEVER invent or rewrite content
  - NEVER silently remove speech that might be real
  - Only exclude segments classified as high-confidence low_information_noise
  - Preserve all other segments, including flagged ones
  - Flagged-but-preserved segments get a lightweight [uncertain] prefix;
    the original text is left completely unchanged
  - Adjacent segments within MERGE_GAP_S are joined into a single span
    (same utterance — avoids over-fragmentation)
  - Gaps > PARAGRAPH_GAP_S produce a paragraph break
  - Partial/provisional transcripts must NOT use this module;
    it is intended for final, fully-processed sessions only

Generator version: conservative_v1
"""
import time
from typing import Any, Dict, List, Optional, Set

GENERATOR_VERSION = "conservative_v1"

# Adjacent segments closer than this (gap between end of prev and start of next)
# are merged into a single text span.  0.5 s matches a typical inter-word pause.
MERGE_GAP_S: float = 0.5

# Gap larger than this triggers a paragraph break.
# 3 s heuristic for "enough silence to signal a new thought / speaker turn".
PARAGRAPH_GAP_S: float = 3.0

# Issue types where segments are preserved but prefixed with an uncertainty marker.
# We mark these because the evidence is strong enough to surface to the reader,
# but not strong enough to justify deletion.
#
# NOT included: missing_audio_continuity (that flags a gap, not the text)
#               segmentation_error       (timing issue; text may be fine)
_MARK_UNCERTAIN_TYPES: frozenset = frozenset({
    "phonetic_hallucination",
    "syntax_collapse",
})

# Prefix prepended to uncertain-but-preserved segments.
# Short and clearly machine-generated so the reader knows it is a flag, not text.
_UNCERTAIN_PREFIX = "[uncertain] "


# ---------------------------------------------------------------------------
# Segment accessors — accept both start_ms/end_ms (ms) and start/end (seconds)
# ---------------------------------------------------------------------------

def _seg_start_s(seg: Dict) -> float:
    if "start_ms" in seg:
        return seg["start_ms"] / 1000.0
    return float(seg.get("start", 0))


def _seg_end_s(seg: Dict) -> float:
    if "end_ms" in seg:
        return seg["end_ms"] / 1000.0
    return float(seg.get("end", 0))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_clean_transcript(
    segments: List[Dict],
    quality_report: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Build a reading-oriented clean transcript from ASR segments.

    Args:
        segments:       Raw segment dicts.  May use start/end (float, seconds)
                        or start_ms/end_ms (int, milliseconds) plus "text".
                        Speaker labels are preserved if present.
        quality_report: Output of analyze_transcript_quality() — used to
                        identify excluded and uncertain segments.
                        Passing None is safe: no segments are excluded or marked.

    Returns a dict:
        clean_text              — full reading text (paragraphs joined by \\n\\n)
        paragraphs              — list of {"text": str, "segment_indices": [int]}
        excluded_segment_indices — original indices of noise-excluded segments
        uncertainty_markers     — list of {"segment_index", "type", "original_text"}
        merge_count             — number of merge operations performed
        excluded_count          — count of excluded segments
        uncertainty_count       — count of uncertainty-marked segments
        generator               — generator version string
        generated_at            — ISO timestamp
    """
    if not segments:
        return {
            "clean_text": "",
            "paragraphs": [],
            "excluded_segment_indices": [],
            "uncertainty_markers": [],
            "merge_count": 0,
            "excluded_count": 0,
            "uncertainty_count": 0,
            "generator": GENERATOR_VERSION,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    # Sort by start time, preserving original indices
    ordered = sorted(enumerate(segments), key=lambda iv: _seg_start_s(iv[1]))

    # ----------------------------------------------------------------
    # Build exclusion set and uncertainty set from quality report
    # ----------------------------------------------------------------
    excluded_indices: Set[int] = set()
    uncertain_indices: Dict[int, str] = {}  # orig_idx → issue type

    if quality_report:
        for issue in quality_report.get("issues", []):
            orig_idx = issue.get("segment_index")
            if orig_idx is None:
                continue
            itype = issue.get("type", "")
            confidence = issue.get("confidence", "low")

            # Only exclude high-confidence noise — conservative threshold
            if itype == "low_information_noise" and confidence == "high":
                excluded_indices.add(orig_idx)

            # Mark uncertain (but keep) for hallucination/collapse with
            # at least medium confidence — low confidence is not shown to reader
            elif itype in _MARK_UNCERTAIN_TYPES and confidence in ("medium", "high"):
                if orig_idx not in excluded_indices:
                    uncertain_indices[orig_idx] = itype

    # ----------------------------------------------------------------
    # Build paragraphs
    # ----------------------------------------------------------------
    paragraphs: List[Dict[str, Any]] = []
    current_parts: List[str] = []
    current_indices: List[int] = []
    prev_end_s: Optional[float] = None
    merge_count = 0
    uncertainty_markers: List[Dict[str, Any]] = []

    for orig_idx, seg in ordered:
        # Skip high-confidence noise
        if orig_idx in excluded_indices:
            continue

        text = seg.get("text", "").strip()
        if not text:
            continue

        start_s = _seg_start_s(seg)
        end_s = _seg_end_s(seg)

        # Apply uncertainty prefix if flagged
        display_text = text
        if orig_idx in uncertain_indices:
            display_text = _UNCERTAIN_PREFIX + text
            uncertainty_markers.append({
                "segment_index": orig_idx,
                "type": uncertain_indices[orig_idx],
                "original_text": text,
            })

        if prev_end_s is None:
            # First segment — open first paragraph
            current_parts.append(display_text)
            current_indices.append(orig_idx)

        else:
            gap_s = start_s - prev_end_s

            if gap_s > PARAGRAPH_GAP_S:
                # New paragraph
                if current_parts:
                    paragraphs.append({
                        "text": " ".join(current_parts),
                        "segment_indices": list(current_indices),
                    })
                current_parts = [display_text]
                current_indices = [orig_idx]

            elif gap_s <= MERGE_GAP_S:
                # Merge: same utterance run
                if current_parts:
                    current_parts[-1] = current_parts[-1] + " " + display_text
                    current_indices.append(orig_idx)
                    merge_count += 1
                else:
                    current_parts.append(display_text)
                    current_indices.append(orig_idx)

            else:
                # Moderate gap: same paragraph, separate part
                current_parts.append(display_text)
                current_indices.append(orig_idx)

        prev_end_s = end_s

    # Flush final paragraph
    if current_parts:
        paragraphs.append({
            "text": " ".join(current_parts),
            "segment_indices": list(current_indices),
        })

    clean_text = "\n\n".join(p["text"] for p in paragraphs)

    return {
        "clean_text": clean_text,
        "paragraphs": paragraphs,
        "excluded_segment_indices": sorted(excluded_indices),
        "uncertainty_markers": uncertainty_markers,
        "merge_count": merge_count,
        "excluded_count": len(excluded_indices),
        "uncertainty_count": len(uncertainty_markers),
        "generator": GENERATOR_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
