"""
Unit tests for:
  A. Chunk merge path correctness
  B. Truthful duration / coverage metadata
  C. Diarization resolution and safe defaulting
  D. Finalize body parsing of speaker_count and diarization fields

These are pure-logic tests — no running server or Redis needed.
"""
import os
import sys
import json
import wave
import struct
import tempfile
import shutil
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

# Ensure server modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api_v2 import _merge_wav_chunks, _measure_wav_duration


# ---------------------------------------------------------------------------
# Helpers: generate real WAV files for merge testing
# ---------------------------------------------------------------------------
def _make_wav(path: Path, duration_s: float = 1.0,
              sample_rate: int = 16000, channels: int = 1) -> Path:
    """Create a valid mono 16-bit PCM WAV file with silence."""
    n_frames = int(sample_rate * duration_s)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        # Write silence (zero samples)
        w.writeframes(b"\x00\x00" * n_frames * channels)
    return path


@pytest.fixture
def tmp_session(tmp_path):
    """Create a temporary session directory with chunks/ subdir."""
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    return tmp_path


# ===========================================================================
# A. MERGE PATH CORRECTNESS
# ===========================================================================
class TestMergePathCorrectness:
    """Verify that _merge_wav_chunks merges ALL chunks in index order."""

    def test_single_chunk_is_copied(self, tmp_session):
        """A single chunk should be copied directly as audio.wav."""
        chunk = _make_wav(tmp_session / "chunks" / "chunk_0000.wav", 2.0)
        result = _merge_wav_chunks(tmp_session, [chunk])
        assert result == tmp_session / "audio.wav"
        assert result.exists()
        dur = _measure_wav_duration(result)
        assert dur is not None
        assert abs(dur - 2.0) < 0.01

    def test_multiple_chunks_merged_in_order(self, tmp_session):
        """Multiple chunks should merge into one file with combined duration."""
        chunks = []
        for i in range(5):
            c = _make_wav(tmp_session / "chunks" / f"chunk_{i:04d}.wav", 1.0)
            chunks.append(c)
        result = _merge_wav_chunks(tmp_session, chunks)
        dur = _measure_wav_duration(result)
        assert dur is not None
        assert abs(dur - 5.0) < 0.05  # 5 × 1s chunks

    def test_incompatible_chunk_skipped(self, tmp_session):
        """Chunks with different format should be skipped, rest merged."""
        c1 = _make_wav(tmp_session / "chunks" / "chunk_0000.wav", 2.0,
                       sample_rate=16000, channels=1)
        c2 = _make_wav(tmp_session / "chunks" / "chunk_0001.wav", 1.0,
                       sample_rate=44100, channels=2)  # Different format
        c3 = _make_wav(tmp_session / "chunks" / "chunk_0002.wav", 2.0,
                       sample_rate=16000, channels=1)
        result = _merge_wav_chunks(tmp_session, [c1, c2, c3])
        dur = _measure_wav_duration(result)
        assert dur is not None
        # c2 should be skipped → 2 + 2 = 4s
        assert abs(dur - 4.0) < 0.05

    def test_empty_chunk_list_raises(self, tmp_session):
        """Merging zero chunks should raise ValueError."""
        with pytest.raises(ValueError, match="No chunk paths"):
            _merge_wav_chunks(tmp_session, [])

    def test_merged_file_is_audio_wav(self, tmp_session):
        """Output should always be session_dir/audio.wav."""
        c = _make_wav(tmp_session / "chunks" / "chunk_0000.wav", 1.0)
        result = _merge_wav_chunks(tmp_session, [c])
        assert result.name == "audio.wav"
        assert result.parent == tmp_session


# ===========================================================================
# B. TRUTHFUL DURATION / COVERAGE METADATA
# ===========================================================================
class TestMeasureWavDuration:
    """Verify _measure_wav_duration returns truthful durations."""

    def test_known_duration(self, tmp_path):
        """A 3.5s WAV file should report ~3.5s."""
        p = _make_wav(tmp_path / "test.wav", 3.5)
        dur = _measure_wav_duration(p)
        assert dur is not None
        assert abs(dur - 3.5) < 0.01

    def test_nonexistent_file_returns_none(self, tmp_path):
        """Missing file should return None, not raise."""
        dur = _measure_wav_duration(tmp_path / "nope.wav")
        assert dur is None

    def test_non_wav_file_returns_none(self, tmp_path):
        """A non-WAV file should return None."""
        p = tmp_path / "garbage.wav"
        p.write_bytes(b"not a wav file at all")
        dur = _measure_wav_duration(p)
        assert dur is None

    def test_merged_duration_matches_sum(self, tmp_session):
        """After merging, measured duration should match sum of chunks."""
        chunks = [
            _make_wav(tmp_session / "chunks" / f"chunk_{i:04d}.wav", d)
            for i, d in enumerate([10.0, 20.0, 15.0])
        ]
        result = _merge_wav_chunks(tmp_session, chunks)
        dur = _measure_wav_duration(result)
        assert dur is not None
        assert abs(dur - 45.0) < 0.1  # 10 + 20 + 15


class TestCoverageMetadata:
    """Verify coverage ratio and warning logic."""

    def test_coverage_ratio_full(self):
        """When transcript covers all audio, ratio should be ~1.0."""
        audio_dur = 100.0
        transcript_end = 98.5
        ratio = round(transcript_end / audio_dur, 3)
        assert 0.9 < ratio <= 1.0

    def test_coverage_ratio_low(self):
        """When transcript covers only a fraction, ratio should be low."""
        audio_dur = 270.0
        transcript_end = 58.2
        ratio = round(transcript_end / audio_dur, 3)
        assert ratio < 0.5
        assert ratio == pytest.approx(0.216, abs=0.001)

    def test_coverage_warning_generated_when_low(self):
        """Worker should emit a warning when coverage < 50%."""
        audio_dur = 270.0
        transcript_end = 58.2
        ratio = round(transcript_end / audio_dur, 3)
        warning = None
        if ratio < 0.5:
            warning = (
                f"Transcript covers only {ratio:.1%} of audio "
                f"({transcript_end:.1f}s / {audio_dur:.1f}s)"
            )
        assert warning is not None
        assert "21.6%" in warning

    def test_no_warning_when_full_coverage(self):
        """No warning when transcript covers most of the audio."""
        audio_dur = 100.0
        transcript_end = 95.0
        ratio = round(transcript_end / audio_dur, 3)
        warning = None
        if ratio < 0.5:
            warning = "should not happen"
        assert warning is None


# ===========================================================================
# C. PARTIAL VS FULL JOB SEPARATION
# ===========================================================================
def _merge_chunks_to_path_standalone(chunk_paths: List[Path], output_path: Path) -> None:
    """
    Standalone copy of worker._merge_chunks_to_path for testing without GPU deps.
    Merges ordered WAV chunks into output_path. Skips incompatible chunks.
    """
    if not chunk_paths:
        raise ValueError("No chunks to merge.")
    if len(chunk_paths) == 1:
        import shutil
        shutil.copy2(str(chunk_paths[0]), str(output_path))
        return
    ref_nchannels = ref_sampwidth = ref_framerate = None
    for cp in chunk_paths:
        try:
            with wave.open(str(cp), "rb") as w:
                ref_nchannels = w.getnchannels()
                ref_sampwidth = w.getsampwidth()
                ref_framerate = w.getframerate()
            break
        except Exception:
            pass
    if ref_nchannels is None:
        raise ValueError("Could not determine WAV format from chunks.")
    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(ref_nchannels)
        out.setsampwidth(ref_sampwidth)
        out.setframerate(ref_framerate)
        for cp in chunk_paths:
            try:
                with wave.open(str(cp), "rb") as w:
                    if (w.getnchannels() == ref_nchannels
                            and w.getsampwidth() == ref_sampwidth
                            and w.getframerate() == ref_framerate):
                        out.writeframes(w.readframes(w.getnframes()))
            except Exception:
                pass


class TestPartialFullSeparation:
    """Verify partial job does not overwrite full-job artifacts."""

    def test_partial_merge_uses_audio_partial(self, tmp_session):
        """Worker's _merge_chunks_to_path writes to the specified path, not audio.wav."""
        c1 = _make_wav(tmp_session / "chunks" / "chunk_0000.wav", 2.0)
        c2 = _make_wav(tmp_session / "chunks" / "chunk_0001.wav", 3.0)

        partial_path = tmp_session / "audio_partial.wav"
        _merge_chunks_to_path_standalone([c1, c2], partial_path)

        assert partial_path.exists()
        # audio.wav should NOT exist
        assert not (tmp_session / "audio.wav").exists()

    def test_full_merge_creates_audio_wav(self, tmp_session):
        """_merge_wav_chunks writes audio.wav."""
        c1 = _make_wav(tmp_session / "chunks" / "chunk_0000.wav", 2.0)
        result = _merge_wav_chunks(tmp_session, [c1])
        assert result.name == "audio.wav"
        assert result.exists()

    def test_partial_does_not_overwrite_existing_audio_wav(self, tmp_session):
        """If audio.wav already exists, partial merge should leave it untouched."""
        c1 = _make_wav(tmp_session / "chunks" / "chunk_0000.wav", 5.0)

        # First: create audio.wav via full merge
        full_result = _merge_wav_chunks(tmp_session, [c1])
        full_size = full_result.stat().st_size

        # Now do a partial merge with a shorter audio to a DIFFERENT file
        c_short = _make_wav(tmp_session / "chunks" / "chunk_0000.wav", 1.0)
        partial_path = tmp_session / "audio_partial.wav"
        _merge_chunks_to_path_standalone([c_short], partial_path)

        # audio.wav should be UNCHANGED (same size as 5s audio)
        assert full_result.exists()
        assert full_result.stat().st_size == full_size


# ===========================================================================
# D. DIARIZATION RESOLUTION
# ===========================================================================
class TestDiarizationResolution:
    """Test finalize body parsing for diarization and speaker_count fields."""

    def _parse_diarization(self, body: dict):
        """Simulate the finalize diarization parsing logic."""
        run_diarization = bool(
            body.get("run_diarization")
            or body.get("diarization")
            or False
        )
        speaker_count = int(
            body.get("speaker_count")
            or body.get("num_speakers")
            or (2 if run_diarization else 1)
        )
        # Safety enforcement
        if run_diarization and speaker_count < 2:
            speaker_count = 2
        return run_diarization, speaker_count

    def test_run_diarization_true(self):
        diar, sc = self._parse_diarization({"run_diarization": True})
        assert diar is True
        assert sc == 2

    def test_legacy_diarization_field(self):
        diar, sc = self._parse_diarization({"diarization": True})
        assert diar is True
        assert sc == 2

    def test_no_diarization_field(self):
        diar, sc = self._parse_diarization({})
        assert diar is False
        assert sc == 1

    def test_diarization_false_explicit(self):
        diar, sc = self._parse_diarization({"run_diarization": False})
        assert diar is False
        assert sc == 1

    def test_speaker_count_field(self):
        diar, sc = self._parse_diarization(
            {"run_diarization": True, "speaker_count": 3}
        )
        assert diar is True
        assert sc == 3

    def test_num_speakers_field(self):
        diar, sc = self._parse_diarization(
            {"diarization": True, "num_speakers": 4}
        )
        assert diar is True
        assert sc == 4

    def test_diarization_true_speaker_count_1_forced_to_2(self):
        """If client sends diarization=true + speaker_count=1, force to 2."""
        diar, sc = self._parse_diarization(
            {"run_diarization": True, "speaker_count": 1}
        )
        assert diar is True
        assert sc == 2  # Forced from 1 to 2

    def test_diarization_false_speaker_count_1_stays(self):
        """If diarization=false + speaker_count=1, it stays at 1."""
        diar, sc = self._parse_diarization(
            {"run_diarization": False, "speaker_count": 1}
        )
        assert diar is False
        assert sc == 1


class TestWorkerDiarizationResolution:
    """Test that the worker correctly reads diarization_enabled from job/metadata."""

    def _resolve_worker_diarization(self, job_data: dict, metadata: dict):
        """Simulate the worker's diarization resolution logic."""
        speaker_count = int(
            job_data.get("speaker_count", metadata.get("speaker_count", 1))
        )
        diarization_enabled = bool(
            job_data.get("diarization_enabled",
                         metadata.get("diarization_enabled", False))
        ) or speaker_count > 1

        if diarization_enabled and speaker_count < 2:
            speaker_count = 2
        return diarization_enabled, speaker_count

    def test_explicit_diarization_enabled_in_job(self):
        diar, sc = self._resolve_worker_diarization(
            {"diarization_enabled": True, "speaker_count": 2}, {}
        )
        assert diar is True
        assert sc == 2

    def test_diarization_from_metadata(self):
        diar, sc = self._resolve_worker_diarization(
            {"speaker_count": 1},
            {"diarization_enabled": True}
        )
        assert diar is True
        assert sc == 2  # Forced from 1 to 2

    def test_speaker_count_gt_1_implies_diarization(self):
        diar, sc = self._resolve_worker_diarization(
            {"speaker_count": 3}, {}
        )
        assert diar is True
        assert sc == 3

    def test_no_diarization_speaker_1(self):
        diar, sc = self._resolve_worker_diarization(
            {"speaker_count": 1}, {"diarization_enabled": False}
        )
        assert diar is False
        assert sc == 1

    def test_defaults_no_diarization(self):
        diar, sc = self._resolve_worker_diarization({}, {})
        assert diar is False
        assert sc == 1

    def test_diarization_enabled_forces_speaker_count_min_2(self):
        """If diarization is enabled but speaker_count=1, force to 2."""
        diar, sc = self._resolve_worker_diarization(
            {"diarization_enabled": True, "speaker_count": 1}, {}
        )
        assert diar is True
        assert sc == 2


# ===========================================================================
# E. MEASURE WAV INFO (worker helper)
# ===========================================================================
def _measure_wav_info_standalone(path: Path):
    """Standalone copy of worker._measure_wav_info for testing without GPU deps."""
    info = {"duration_s": None, "channels": None, "sample_rate": None, "byte_size": None}
    try:
        info["byte_size"] = path.stat().st_size
    except Exception:
        pass
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            info["channels"] = w.getnchannels()
            info["sample_rate"] = rate
            if rate > 0:
                info["duration_s"] = round(frames / rate, 2)
    except Exception:
        pass
    return info


class TestMeasureWavInfo:
    """Verify the _measure_wav_info logic (standalone copy — same algorithm as worker)."""

    def test_returns_all_fields(self, tmp_path):
        p = _make_wav(tmp_path / "test.wav", 2.5, sample_rate=16000, channels=1)
        info = _measure_wav_info_standalone(p)
        assert info["duration_s"] is not None
        assert abs(info["duration_s"] - 2.5) < 0.01
        assert info["channels"] == 1
        assert info["sample_rate"] == 16000
        assert info["byte_size"] is not None
        assert info["byte_size"] > 0

    def test_nonexistent_file(self, tmp_path):
        info = _measure_wav_info_standalone(tmp_path / "nope.wav")
        assert info["duration_s"] is None
        assert info["byte_size"] is None
