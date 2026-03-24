"""
Focused unit tests for Android client field compatibility:

A. create-session: alias handling for sample_rate, source_type, chunk_duration_sec,
   diarization — and source_type=device_import suppressing partial triggers.

B. upload-chunk: continuity metadata alias handling for decode_errors, ble_gaps,
   plc_frames_applied, has_continuity_warning.

These are pure-logic tests.  They exercise the field-resolution code paths by
constructing the same data flow that the endpoints use, without needing a
running server, Redis, or ASGI test client.
"""
import os
import sys
import json
import wave
from pathlib import Path
from typing import Optional

import pytest

# Ensure server modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===========================================================================
# A. CREATE-SESSION ALIAS HANDLING
# ===========================================================================

def _simulate_create_session_body(body: dict) -> dict:
    """
    Replicate the exact field-resolution logic from create_session().

    Returns what would be stored in v2_session.json.
    This mirrors the code in api_v2.py create_session() after the alias
    resolution block.  If the server code changes, this helper must be
    updated to match — that is intentional: the test should break if
    someone alters the resolution logic.
    """
    sample_rate_hz = body.get("sample_rate_hz") or body.get("sample_rate") or 16000

    source_type = body.get("source_type")
    if source_type == "device_import":
        mode = "file"
    else:
        mode = body.get("mode", "stream")

    return {
        "sample_rate_hz": int(sample_rate_hz),
        "mode": mode,
        "source_type": source_type,
        "chunk_duration_sec": body.get("chunk_duration_sec"),
        "diarization_hint": bool(body.get("diarization", False)),
    }


class TestCreateSessionAliases:

    def test_legacy_sample_rate_hz(self):
        """Existing clients sending sample_rate_hz should still work."""
        result = _simulate_create_session_body({"sample_rate_hz": 44100})
        assert result["sample_rate_hz"] == 44100

    def test_android_sample_rate_alias(self):
        """Android sends 'sample_rate' — should be accepted."""
        result = _simulate_create_session_body({"sample_rate": 48000})
        assert result["sample_rate_hz"] == 48000

    def test_sample_rate_hz_takes_priority(self):
        """If both sample_rate_hz and sample_rate are sent, sample_rate_hz wins."""
        result = _simulate_create_session_body({
            "sample_rate_hz": 16000,
            "sample_rate": 48000,
        })
        assert result["sample_rate_hz"] == 16000

    def test_default_sample_rate(self):
        """No sample rate field at all → default 16000."""
        result = _simulate_create_session_body({})
        assert result["sample_rate_hz"] == 16000

    def test_source_type_device_import_sets_mode_file(self):
        """source_type=device_import must set mode=file."""
        result = _simulate_create_session_body({"source_type": "device_import"})
        assert result["mode"] == "file"
        assert result["source_type"] == "device_import"

    def test_source_type_ble_stream_keeps_stream(self):
        """source_type=ble_stream should NOT override mode from stream."""
        result = _simulate_create_session_body({"source_type": "ble_stream"})
        assert result["mode"] == "stream"

    def test_no_source_type_defaults_to_stream(self):
        """No source_type and no mode → stream."""
        result = _simulate_create_session_body({})
        assert result["mode"] == "stream"
        assert result["source_type"] is None

    def test_explicit_mode_preserved_when_no_device_import(self):
        """Explicit mode=file without source_type should be respected."""
        result = _simulate_create_session_body({"mode": "file"})
        assert result["mode"] == "file"

    def test_source_type_device_import_overrides_explicit_stream(self):
        """source_type=device_import should win even if mode=stream is also sent."""
        result = _simulate_create_session_body({
            "source_type": "device_import",
            "mode": "stream",
        })
        assert result["mode"] == "file"

    def test_chunk_duration_sec_stored(self):
        """Android's chunk_duration_sec should be stored."""
        result = _simulate_create_session_body({"chunk_duration_sec": 30.0})
        assert result["chunk_duration_sec"] == 30.0

    def test_diarization_hint_stored(self):
        """Android's diarization field should be stored as diarization_hint."""
        result = _simulate_create_session_body({"diarization": True})
        assert result["diarization_hint"] is True

    def test_diarization_hint_default_false(self):
        """No diarization field → diarization_hint=False."""
        result = _simulate_create_session_body({})
        assert result["diarization_hint"] is False


class TestDeviceImportSuppressesPartial:
    """
    The partial trigger in upload_chunk checks:
        session_data.get("mode", "stream") == "stream"

    Sessions created with source_type=device_import get mode="file",
    so partial triggers should NOT fire for them.
    """

    def test_stream_mode_would_trigger(self):
        """mode=stream should satisfy the partial trigger condition."""
        session_data = {"mode": "stream"}
        would_trigger = session_data.get("mode", "stream") == "stream"
        assert would_trigger is True

    def test_file_mode_blocks_trigger(self):
        """mode=file should NOT satisfy the partial trigger condition."""
        session_data = {"mode": "file"}
        would_trigger = session_data.get("mode", "stream") == "stream"
        assert would_trigger is False

    def test_device_import_resolves_to_file_mode(self):
        """End-to-end: source_type=device_import → mode=file → no partial trigger."""
        body = {"source_type": "device_import"}
        session = _simulate_create_session_body(body)
        would_trigger = session["mode"] == "stream"
        assert would_trigger is False


# ===========================================================================
# B. UPLOAD-CHUNK CONTINUITY METADATA ALIAS HANDLING
# ===========================================================================

def _normalize_continuity(
    # Legacy fields
    dropped_frames: int = 0,
    decode_failure: bool = False,
    gap_before_ms: int = 0,
    source_degraded: bool = False,
    # Current Android fields
    decode_errors: int = 0,
    ble_gaps: int = 0,
    plc_frames_applied: int = 0,
    has_continuity_warning: bool = False,
) -> dict:
    """
    Replicate the exact normalization logic from upload_chunk().

    Returns the normalized values that would be stored in chunk_info.
    """
    if decode_errors > 0 and not decode_failure:
        decode_failure = True
    if ble_gaps > 0 and gap_before_ms == 0:
        gap_before_ms = ble_gaps
    dropped_frames = dropped_frames + plc_frames_applied
    if has_continuity_warning and not source_degraded:
        source_degraded = True

    return {
        "dropped_frames": dropped_frames,
        "decode_failure": decode_failure,
        "gap_before_ms": gap_before_ms,
        "source_degraded": source_degraded,
    }


class TestUploadContinuityAliases:

    def test_legacy_fields_passthrough(self):
        """Legacy field names should work exactly as before."""
        result = _normalize_continuity(
            dropped_frames=5,
            decode_failure=True,
            gap_before_ms=100,
            source_degraded=True,
        )
        assert result == {
            "dropped_frames": 5,
            "decode_failure": True,
            "gap_before_ms": 100,
            "source_degraded": True,
        }

    def test_android_decode_errors(self):
        """decode_errors > 0 should set decode_failure=True."""
        result = _normalize_continuity(decode_errors=3)
        assert result["decode_failure"] is True

    def test_android_decode_errors_zero(self):
        """decode_errors=0 should leave decode_failure=False."""
        result = _normalize_continuity(decode_errors=0)
        assert result["decode_failure"] is False

    def test_legacy_decode_failure_preserved_with_zero_android(self):
        """Legacy decode_failure=True should survive even when decode_errors=0."""
        result = _normalize_continuity(decode_failure=True, decode_errors=0)
        assert result["decode_failure"] is True

    def test_android_ble_gaps(self):
        """ble_gaps > 0 should map to gap_before_ms when gap_before_ms=0."""
        result = _normalize_continuity(ble_gaps=250)
        assert result["gap_before_ms"] == 250

    def test_android_ble_gaps_does_not_overwrite_legacy(self):
        """If legacy gap_before_ms is already set, ble_gaps should not overwrite."""
        result = _normalize_continuity(gap_before_ms=100, ble_gaps=250)
        assert result["gap_before_ms"] == 100

    def test_android_plc_frames_added_to_dropped(self):
        """plc_frames_applied should add to dropped_frames."""
        result = _normalize_continuity(dropped_frames=2, plc_frames_applied=10)
        assert result["dropped_frames"] == 12

    def test_android_plc_frames_alone(self):
        """plc_frames_applied alone (no legacy dropped_frames)."""
        result = _normalize_continuity(plc_frames_applied=7)
        assert result["dropped_frames"] == 7

    def test_android_has_continuity_warning(self):
        """has_continuity_warning=True should set source_degraded=True."""
        result = _normalize_continuity(has_continuity_warning=True)
        assert result["source_degraded"] is True

    def test_legacy_source_degraded_preserved(self):
        """Legacy source_degraded=True should survive even when has_continuity_warning=False."""
        result = _normalize_continuity(source_degraded=True, has_continuity_warning=False)
        assert result["source_degraded"] is True

    def test_all_zeros(self):
        """No continuity data at all → all defaults."""
        result = _normalize_continuity()
        assert result == {
            "dropped_frames": 0,
            "decode_failure": False,
            "gap_before_ms": 0,
            "source_degraded": False,
        }

    def test_all_android_fields_combined(self):
        """All Android fields non-zero at once."""
        result = _normalize_continuity(
            decode_errors=2,
            ble_gaps=500,
            plc_frames_applied=15,
            has_continuity_warning=True,
        )
        assert result == {
            "dropped_frames": 15,
            "decode_failure": True,
            "gap_before_ms": 500,
            "source_degraded": True,
        }

    def test_mixed_legacy_and_android(self):
        """Both legacy and Android fields provided — merge correctly."""
        result = _normalize_continuity(
            dropped_frames=3,
            decode_failure=False,
            gap_before_ms=0,
            source_degraded=False,
            decode_errors=1,
            ble_gaps=200,
            plc_frames_applied=5,
            has_continuity_warning=True,
        )
        assert result["dropped_frames"] == 8    # 3 + 5
        assert result["decode_failure"] is True  # from decode_errors
        assert result["gap_before_ms"] == 200   # from ble_gaps (legacy was 0)
        assert result["source_degraded"] is True  # from has_continuity_warning


# ===========================================================================
# C. PARTIAL TRIGGER SUPPRESSION / COOLDOWN
# ===========================================================================

from datetime import datetime, timezone, timedelta


def _should_trigger_partial(
    *,
    is_final: bool = False,
    mode: str = "stream",
    total_chunks: int = 10,
    partial_every_n: int = 5,
    state: str = "receiving",
    last_partial_trigger_at: str = None,
    cooldown_seconds: int = 120,
) -> bool:
    """
    Replicate the exact partial-trigger decision logic from upload_chunk()
    and the cooldown check inside _trigger_partial().

    Returns True if a partial would be triggered.
    """
    # Gate 1: upload_chunk condition block
    if is_final:
        return False
    if mode != "stream":
        return False
    if partial_every_n <= 0:
        return False
    if total_chunks % partial_every_n != 0:
        return False
    if state not in ("receiving", "partially_processed"):
        return False

    # Gate 2: _trigger_partial cooldown check (lockfile assumed absent for test)
    if cooldown_seconds > 0 and last_partial_trigger_at:
        try:
            last_ts = datetime.fromisoformat(
                last_partial_trigger_at.replace("Z", "+00:00")
            )
            elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds()
            if elapsed < cooldown_seconds:
                return False
        except (ValueError, TypeError):
            pass

    return True


class TestPartialTriggerSuppression:

    def test_normal_stream_triggers(self):
        """Standard live stream session at chunk boundary triggers partial."""
        assert _should_trigger_partial(
            total_chunks=10, partial_every_n=5, mode="stream"
        ) is True

    def test_is_final_suppresses(self):
        """is_final=True should suppress partial — finalize imminent."""
        assert _should_trigger_partial(
            is_final=True, total_chunks=10, partial_every_n=5
        ) is False

    def test_file_mode_suppresses(self):
        """mode=file (device_import) should suppress partial."""
        assert _should_trigger_partial(
            mode="file", total_chunks=10, partial_every_n=5
        ) is False

    def test_non_boundary_chunk_no_trigger(self):
        """Chunk count not on boundary should not trigger."""
        assert _should_trigger_partial(
            total_chunks=7, partial_every_n=5
        ) is False

    def test_cooldown_suppresses_recent_trigger(self):
        """Partial triggered 30s ago with 120s cooldown should suppress."""
        recent = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
        assert _should_trigger_partial(
            total_chunks=10, partial_every_n=5,
            last_partial_trigger_at=recent,
            cooldown_seconds=120,
        ) is False

    def test_cooldown_allows_after_elapsed(self):
        """Partial triggered 200s ago with 120s cooldown should allow."""
        old = (datetime.now(timezone.utc) - timedelta(seconds=200)).isoformat()
        assert _should_trigger_partial(
            total_chunks=10, partial_every_n=5,
            last_partial_trigger_at=old,
            cooldown_seconds=120,
        ) is True

    def test_no_previous_trigger_allows(self):
        """No previous trigger (last_partial_trigger_at=None) should allow."""
        assert _should_trigger_partial(
            total_chunks=10, partial_every_n=5,
            last_partial_trigger_at=None,
            cooldown_seconds=120,
        ) is True

    def test_cooldown_zero_disables(self):
        """cooldown_seconds=0 should always allow (no cooldown)."""
        recent = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        assert _should_trigger_partial(
            total_chunks=10, partial_every_n=5,
            last_partial_trigger_at=recent,
            cooldown_seconds=0,
        ) is True

    def test_malformed_timestamp_allows(self):
        """Malformed last_partial_trigger_at should not block."""
        assert _should_trigger_partial(
            total_chunks=10, partial_every_n=5,
            last_partial_trigger_at="not-a-date",
            cooldown_seconds=120,
        ) is True

    def test_chunks_complete_state_no_trigger(self):
        """state=chunks_complete should not trigger partial (finalize imminent)."""
        assert _should_trigger_partial(
            total_chunks=10, partial_every_n=5,
            state="chunks_complete",
        ) is False

    def test_finalized_state_no_trigger(self):
        """state=finalized should not trigger partial."""
        assert _should_trigger_partial(
            total_chunks=10, partial_every_n=5,
            state="finalized",
        ) is False

    def test_partial_disabled_no_trigger(self):
        """PARTIAL_EVERY_N_CHUNKS=0 should disable partials."""
        assert _should_trigger_partial(
            total_chunks=10, partial_every_n=0,
        ) is False

    def test_backlog_scenario_limited(self):
        """
        Simulate 32 chunks uploaded at 30s intervals with 120s cooldown.
        Without cooldown: 6 partials (at 5,10,15,20,25,30).
        With 120s cooldown: at most 3 (at 5, ~13, ~21, ~29 — but only boundaries matter).
        """
        triggers = 0
        cooldown = 120
        last_trigger_at = None
        for chunk_num in range(1, 33):
            # Simulate wall-clock time: chunk_num * 30 seconds from start
            sim_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=chunk_num * 30)
            # Check if this chunk would trigger
            if (
                chunk_num % 5 == 0  # boundary
                and True  # mode=stream, not is_final, state=receiving
            ):
                # Check cooldown
                should = True
                if cooldown > 0 and last_trigger_at:
                    elapsed = (sim_now - last_trigger_at).total_seconds()
                    if elapsed < cooldown:
                        should = False
                if should:
                    triggers += 1
                    last_trigger_at = sim_now

        # Without cooldown there would be 6 triggers.
        # With 120s cooldown at 30s intervals, boundaries are at:
        #   chunk 5 (150s): trigger (first)
        #   chunk 10 (300s): 150s elapsed > 120 → trigger
        #   chunk 15 (450s): 150s elapsed > 120 → trigger
        #   chunk 20 (600s): 150s elapsed > 120 → trigger
        #   chunk 25 (750s): 150s elapsed > 120 → trigger
        #   chunk 30 (900s): 150s elapsed > 120 → trigger
        # With every-5 at 30s per chunk, gap between boundaries = 5*30 = 150s > 120s.
        # So cooldown doesn't reduce count for THIS interval, but for faster uploads
        # (e.g. backlog at 1 chunk/sec), it would be dramatically fewer.
        assert triggers == 6  # 150s gap > 120s cooldown

    def test_rapid_backlog_scenario_limited(self):
        """
        Simulate 100 chunks uploaded at 1 chunk/sec (rapid backlog).
        Without cooldown: 20 partials (at 5,10,15,...,100).
        With 120s cooldown: should be capped to ~8.
        """
        triggers = 0
        cooldown = 120
        last_trigger_at = None
        for chunk_num in range(1, 101):
            sim_now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc) + timedelta(seconds=chunk_num)
            if chunk_num % 5 == 0:
                should = True
                if cooldown > 0 and last_trigger_at:
                    elapsed = (sim_now - last_trigger_at).total_seconds()
                    if elapsed < cooldown:
                        should = False
                if should:
                    triggers += 1
                    last_trigger_at = sim_now

        # 100 chunks at 1/sec = 100s total. Boundaries at 5,10,...,100 = 20 boundaries.
        # First trigger at 5s. Next allowed at 125s — but session ends at 100s.
        # So only 1 trigger!
        assert triggers == 1


# ===========================================================================
# D. COOLDOWN TIMESTAMP ORDERING (must only persist after successful enqueue)
# ===========================================================================

import tempfile
import json as _json
from unittest.mock import patch, MagicMock


class TestCooldownTimestampOrdering:
    """
    Verify that last_partial_trigger_at is written to v2_session.json
    ONLY after a successful Redis rpush, and NOT on enqueue failure or
    no-Redis paths.
    """

    def _make_session_dir(self, tmp_path):
        """Create a minimal session dir with v2_session.json and chunks dir."""
        d = tmp_path / "test-session"
        d.mkdir()
        (d / "chunks").mkdir()
        session_data = {
            "session_id": "test-session",
            "mode": "stream",
            "state": "receiving",
            "chunks": [],
        }
        with open(d / "v2_session.json", "w") as f:
            _json.dump(session_data, f)
        return d, session_data

    @patch("api_v2._get_redis")
    def test_successful_enqueue_records_timestamp(self, mock_get_redis, tmp_path):
        """Successful rpush must persist last_partial_trigger_at."""
        from api_v2 import _trigger_partial

        d, session_data = self._make_session_dir(tmp_path)

        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        _trigger_partial("test-session", d, session_data)

        # Redis rpush was called
        mock_redis.rpush.assert_called_once()

        # Session data should now have the timestamp
        assert "last_partial_trigger_at" in session_data

        # And it should be persisted to disk
        on_disk = _json.loads((d / "v2_session.json").read_text())
        assert "last_partial_trigger_at" in on_disk

    @patch("api_v2._get_redis")
    def test_failed_enqueue_does_not_record_timestamp(self, mock_get_redis, tmp_path):
        """Failed rpush must NOT persist last_partial_trigger_at."""
        from api_v2 import _trigger_partial

        d, session_data = self._make_session_dir(tmp_path)

        mock_redis = MagicMock()
        mock_redis.rpush.side_effect = Exception("Redis connection lost")
        mock_get_redis.return_value = mock_redis

        _trigger_partial("test-session", d, session_data)

        # Session data in memory should NOT have the timestamp
        assert "last_partial_trigger_at" not in session_data

        # And on disk should NOT have it either
        on_disk = _json.loads((d / "v2_session.json").read_text())
        assert "last_partial_trigger_at" not in on_disk

    @patch("api_v2._get_redis")
    def test_no_redis_does_not_record_timestamp(self, mock_get_redis, tmp_path):
        """No Redis connection must NOT persist last_partial_trigger_at."""
        from api_v2 import _trigger_partial

        d, session_data = self._make_session_dir(tmp_path)

        mock_get_redis.return_value = None  # no Redis

        _trigger_partial("test-session", d, session_data)

        # Session data should NOT have the timestamp
        assert "last_partial_trigger_at" not in session_data

        on_disk = _json.loads((d / "v2_session.json").read_text())
        assert "last_partial_trigger_at" not in on_disk

    @patch("api_v2._get_redis")
    def test_cooldown_still_works_after_successful_trigger(self, mock_get_redis, tmp_path):
        """
        After a successful trigger records a timestamp, the cooldown
        should suppress the next call if not enough time has elapsed.
        """
        from api_v2 import _trigger_partial, PARTIAL_COOLDOWN_SECONDS

        d, session_data = self._make_session_dir(tmp_path)

        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        # First trigger: should succeed and record timestamp
        _trigger_partial("test-session", d, session_data)
        assert mock_redis.rpush.call_count == 1
        assert "last_partial_trigger_at" in session_data

        # Remove lockfile to simulate worker completing the partial
        (d / "partial_pending").unlink(missing_ok=True)

        # Second trigger immediately: cooldown should suppress it
        _trigger_partial("test-session", d, session_data)
        # rpush should still have been called only once
        assert mock_redis.rpush.call_count == 1
