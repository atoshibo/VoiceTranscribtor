"""
Worker service that processes transcription jobs from Redis queue.
Runs heavy GPU operations (preprocessing, diarization, transcription).

Queues:
  REDIS_QUEUE         — main finalized jobs (job_type: v2 or legacy)
  REDIS_PARTIAL_QUEUE — provisional partial transcription jobs (job_type: v2_partial)

The worker polls the main queue first (priority), then the partial queue.

Transcript artifacts written per session:
  transcript.txt              — raw full text (backward compat)
  raw_transcript.json         — raw text + segments with corruption flags
  clean_transcript.json       — reading-oriented clean text + paragraphs
  clean_transcript.txt        — clean text as plain text (quick access)
  quality_report.json         — structured quality analysis
  transcript_timestamps.json  — segments with start/end in seconds
  transcript_by_speaker.json  — aligned diarization segments
"""
import os
import json
import redis
import time
import shutil
import wave
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, List
import sys
import traceback

# Import transcription modules
from transcription import transcribe_audio, generate_summary, generate_analytics
from diarization import perform_diarization, align_diarization_with_transcript
from audio_preprocess import preprocess_audio
from transcript_quality import analyze_transcript_quality
from clean_transcript import build_clean_transcript

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "transcription_jobs")
REDIS_PARTIAL_QUEUE = os.getenv("REDIS_PARTIAL_QUEUE", REDIS_QUEUE + "_partial")
SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", "/data/sessions"))
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "1"))  # Only 1 GPU job at a time

# Status constants
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"


def _utcnow_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -----------------------------
# Robust JSON IO helpers
# -----------------------------
def _safe_read_json(path: Path, retries: int = 10, sleep_s: float = 0.05) -> Dict[str, Any]:
    """
    Read JSON file safely.
    Handles empty/partially-written files by retrying a few times.
    Returns {} if file doesn't exist or is unreadable after retries.
    """
    if not path.exists():
        return {}

    last_err = None
    for _ in range(retries):
        try:
            try:
                if path.stat().st_size == 0:
                    time.sleep(sleep_s)
                    continue
            except Exception:
                pass

            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            if not raw or not raw.strip():
                time.sleep(sleep_s)
                continue

            return json.loads(raw)
        except json.JSONDecodeError as e:
            last_err = e
            time.sleep(sleep_s)
            continue
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
            continue

    if last_err:
        print(f"[WARN] Failed to read JSON after retries: {path} ({last_err})")
    return {}


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Atomic JSON write:
      write to temp file, fsync, then os.replace(temp, final).
    This prevents partially-written JSON that crashes readers.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    payload = json.dumps(data, indent=2, ensure_ascii=False)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def update_session_status(
    session_id: str,
    status: str,
    error: Optional[str] = None,
    progress: Optional[Dict[str, Any]] = None
) -> None:
    """Update session status.json file safely and atomically.

    Automatically tracks started_at (first time status=running) and
    finished_at (when status=done or error).
    """
    session_dir = SESSIONS_DIR / session_id
    status_path = session_dir / "status.json"

    existing = _safe_read_json(status_path)
    status_data = dict(existing) if isinstance(existing, dict) else {}

    status_data.update({
        "status": status,
        "updated_at": _utcnow_str(),
    })

    # Track started_at on first transition to running
    if status == STATUS_RUNNING and "started_at" not in status_data:
        status_data["started_at"] = _utcnow_str()

    # Track finished_at on terminal states
    if status in (STATUS_DONE, STATUS_ERROR):
        status_data["finished_at"] = _utcnow_str()

    if error:
        status_data["error"] = error

    if progress is not None:
        status_data["progress"] = progress

    _atomic_write_json(status_path, status_data)


def update_progress(session_id: str, stage: str, processing_percent: int) -> None:
    """Update progress in status.json."""
    update_session_status(
        session_id,
        STATUS_RUNNING,
        progress={
            "upload": 100,
            "processing": processing_percent,
            "stage": stage
        }
    )
    print(f"[{session_id}] Progress: Upload 100%, Processing {processing_percent}% - {stage}")


def write_partial_preview(session_id: str, segments: list) -> None:
    """
    Write the last 1-2 transcribed segments as a live partial_preview into
    status.json so the API can surface it while the job is still running.
    Called from a segment_callback during transcription — must be fast and
    non-throwing (errors are silently swallowed).
    """
    try:
        text = " ".join(s.get("text", "") for s in segments).strip()
        if not text:
            return
        session_dir = SESSIONS_DIR / session_id
        status_path = session_dir / "status.json"
        existing = _safe_read_json(status_path)
        status_data = dict(existing) if isinstance(existing, dict) else {}
        status_data["partial_preview"] = text
        _atomic_write_json(status_path, status_data)
    except Exception:
        pass


# -----------------------------
# GPU diagnostics
# -----------------------------
def _get_gpu_diagnostics() -> Dict[str, Any]:
    """
    Run nvidia-smi inside the worker container (which has GPU access) to get
    current utilization and memory usage. Called at worker startup and written
    to worker_health.json for the web service to expose.
    Returns a dict with keys: name, utilization_percent, memory_used_mb, memory_total_mb.
    All values are None if nvidia-smi is unavailable or parsing fails.
    """
    result: Dict[str, Any] = {
        "gpu_name": None,
        "utilization_percent": None,
        "memory_used_mb": None,
        "memory_total_mb": None,
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            timeout=5,
            text=True,
        )
        if proc.returncode == 0:
            line = proc.stdout.strip().split("\n")[0]  # first GPU
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                result["gpu_name"] = parts[0] if parts[0] else None
                result["utilization_percent"] = int(parts[1]) if parts[1].isdigit() else None
                result["memory_used_mb"] = int(parts[2]) if parts[2].isdigit() else None
                result["memory_total_mb"] = int(parts[3]) if parts[3].isdigit() else None
    except Exception as e:
        print(f"[GPU-DIAG] nvidia-smi failed: {e}")
    return result


# -----------------------------
# Subtitle helpers
# -----------------------------
def format_timestamp_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def generate_subtitles(session_dir: Path, timestamps: list) -> None:
    srt_lines = []
    for i, item in enumerate(timestamps, 1):
        start = format_timestamp_srt(item["start"])
        end = format_timestamp_srt(item["end"])
        text = item["text"].strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    (session_dir / "subtitles.srt").write_text("\n".join(srt_lines), encoding="utf-8")

    vtt_lines = ["WEBVTT", ""]
    for item in timestamps:
        start = format_timestamp_vtt(item["start"])
        end = format_timestamp_vtt(item["end"])
        text = item["text"].strip()
        vtt_lines.append(f"{start} --> {end}\n{text}")

    (session_dir / "subtitles.vtt").write_text("\n".join(vtt_lines), encoding="utf-8")


def generate_subtitles_with_speakers(session_dir: Path, aligned_segments: list) -> None:
    srt_lines = []
    for i, item in enumerate(aligned_segments, 1):
        start = format_timestamp_srt(item["start"])
        end = format_timestamp_srt(item["end"])
        speaker = item.get("speaker", "SPEAKER_00")
        text = item["text"].strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n[{speaker}] {text}\n")

    (session_dir / "subtitles.srt").write_text("\n".join(srt_lines), encoding="utf-8")

    vtt_lines = ["WEBVTT", ""]
    for item in aligned_segments:
        start = format_timestamp_vtt(item["start"])
        end = format_timestamp_vtt(item["end"])
        speaker = item.get("speaker", "SPEAKER_00")
        text = item["text"].strip()
        vtt_lines.append(f"{start} --> {end}\n[{speaker}] {text}")

    (session_dir / "subtitles.vtt").write_text("\n".join(vtt_lines), encoding="utf-8")


# -----------------------------
# WAV merge helper (for partial jobs)
# -----------------------------
def _merge_chunks_to_path(chunk_paths: List[Path], output_path: Path) -> None:
    """Merge ordered WAV chunks into output_path. Skips incompatible chunks."""
    if not chunk_paths:
        raise ValueError("No chunks to merge.")

    if len(chunk_paths) == 1:
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
                    if (
                        w.getnchannels() == ref_nchannels
                        and w.getsampwidth() == ref_sampwidth
                        and w.getframerate() == ref_framerate
                    ):
                        out.writeframes(w.readframes(w.getnframes()))
                    else:
                        print(f"[PARTIAL-MERGE] Skipping {cp.name}: incompatible format")
            except Exception as e:
                print(f"[PARTIAL-MERGE] Skipping {cp.name}: {e}")


# -----------------------------
# Raw transcript builder
# -----------------------------
def _build_raw_transcript(
    raw_text: str,
    quality_input: List[Dict],
    quality_report: Optional[Dict],
) -> Dict[str, Any]:
    """
    Build raw_transcript.json: raw ASR text + segments annotated with
    corruption flags from the quality report.

    This is the technical-truth view: every recognized segment is present,
    flagged or not.  Nothing is removed.
    """
    # Build a map: segment_index → list of issue types
    flag_map: Dict[int, List[str]] = {}
    if quality_report:
        for issue in quality_report.get("issues", []):
            idx = issue.get("segment_index")
            if idx is not None:
                flag_map.setdefault(idx, []).append(issue["type"])

    ordered = sorted(quality_input, key=lambda s: s.get("start", s.get("start_ms", 0) / 1000.0))

    raw_segments = []
    for i, seg in enumerate(ordered):
        # Normalise to ms for the raw transcript
        if "start_ms" in seg:
            start_ms = int(seg["start_ms"])
            end_ms = int(seg["end_ms"])
        else:
            start_ms = int(seg.get("start", 0) * 1000)
            end_ms = int(seg.get("end", 0) * 1000)

        raw_segments.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": seg.get("text", "").strip(),
            "speaker": seg.get("speaker", "SPEAKER_00"),
            "corruption_flags": flag_map.get(i, []),
        })

    return {
        "raw_text": raw_text,
        "segments": raw_segments,
        "segment_count": len(raw_segments),
        "flagged_segment_count": (
            quality_report.get("flagged_segment_count", 0) if quality_report else 0
        ),
        "generated_at": _utcnow_str(),
        "analysis_version": (
            quality_report.get("analysis_version") if quality_report else None
        ),
    }


# -----------------------------
# Continuity hints builder
# -----------------------------
def _build_continuity_hints(
    session_meta: Dict,
    job_data: Dict,
) -> Optional[Dict]:
    """
    Assemble continuity_hints dict from:
      - per-chunk metadata stored by api_v2 in v2_session.json
      - session-level integrity info from the finalize request body
        (stored in v2_session.json.session_integrity or job_data.session_integrity)

    Returns None if no meaningful hints are available (all defaults).
    """
    chunks = session_meta.get("chunks", [])
    if not chunks:
        return None

    per_chunk = []
    any_nondefault = False

    for ch in chunks:
        dropped = int(ch.get("dropped_frames", 0))
        decode_fail = bool(ch.get("decode_failure", False))
        gap_before = int(ch.get("gap_before_ms", 0))
        source_degraded = bool(ch.get("source_degraded", False))

        if dropped or decode_fail or gap_before or source_degraded:
            any_nondefault = True

        per_chunk.append({
            "chunk_index": ch.get("chunk_index", 0),
            "chunk_started_ms": ch.get("chunk_started_ms", 0),
            "chunk_duration_ms": ch.get("chunk_duration_ms", 0),
            "dropped_frames": dropped,
            "decode_failure": decode_fail,
            "gap_before_ms": gap_before,
            "source_degraded": source_degraded,
        })

    # Session-level integrity: first try job_data (passed at finalize), then session meta
    session_integrity = (
        job_data.get("session_integrity")
        or session_meta.get("session_integrity")
        or {}
    )

    if session_integrity:
        any_nondefault = True

    if not any_nondefault and not session_integrity:
        return None

    return {
        "per_chunk": per_chunk,
        "session_integrity": session_integrity,
    }


# -----------------------------
# Job processing
# -----------------------------
def _derive_action_hint(error_msg: str) -> str:
    """Return a short, actionable hint based on a CUDA-related error message."""
    msg = (error_msg or "").lower()

    if "driver version is insufficient" in msg:
        return (
            "CUDA driver version is insufficient for the container CUDA runtime. "
            "Update the NVIDIA driver on the host to a version compatible with this container, "
            "or use a container image built with an older CUDA runtime. Verify with `nvidia-smi` "
            "inside the worker container."
        )

    if "libcuda.so" in msg or "no cuda-capable device" in msg or "no cuda capable device" in msg:
        return (
            "No CUDA-capable GPU is visible inside the container. Ensure Docker is started with GPU "
            "access (e.g. `--gpus all` or compose device reservations) and that nvidia-container-toolkit "
            "is installed and configured on the host."
        )

    if "float16" in msg and "do not support efficient float16" in msg:
        return (
            "The configured CUDA compute_type=float16 is not supported efficiently on this GPU. "
            "Set WHISPER_COMPUTE_TYPE_CUDA to a supported value such as int8_float16 or int8, "
            "then restart the worker."
        )

    return (
        "GPU initialization failed. Check that the host NVIDIA driver, CUDA runtime inside the "
        "container, and Docker GPU configuration are compatible. See SYSTEM_NOTES for GPU setup steps."
    )


def process_transcription_job(job_data: Dict[str, Any]) -> None:
    session_id = job_data["session_id"]
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{session_id}] Starting transcription job")

    status_path = session_dir / "status.json"

    try:
        # Idempotency: skip if already done
        existing_status = _safe_read_json(status_path)
        if existing_status.get("status") == STATUS_DONE:
            print(f"[{session_id}] Already processed, skipping")
            return

        if existing_status.get("status") == STATUS_CANCELLED:
            print(f"[{session_id}] Job cancelled, skipping")
            return

        # Set running (records started_at automatically)
        update_session_status(session_id, STATUS_RUNNING)

        metadata_path = session_dir / "metadata.json"
        metadata = _safe_read_json(metadata_path)

        audio_path = session_dir / "audio.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # None → faster-whisper auto-detects language; never fall back to "en" by default.
        language = job_data.get("language") or metadata.get("language") or None
        speaker_count = int(job_data.get("speaker_count", metadata.get("speaker_count", 1)))
        model_size = job_data.get("model_size", metadata.get("model_size", "base"))
        timestamps_enabled = bool(job_data.get("timestamps", metadata.get("timestamps", True)))

        diarization_enabled = speaker_count > 1

        update_progress(session_id, "loading_model", 10)

        # Optional audio preprocessing for diarization
        processed_audio_path = audio_path
        if diarization_enabled:
            processed_audio_path = session_dir / "audio_processed.wav"
            update_progress(session_id, "preprocessing", 5)
            ok = preprocess_audio(str(audio_path), str(processed_audio_path))
            if not ok:
                processed_audio_path = audio_path

        diarization_segments = []
        if diarization_enabled:
            update_progress(session_id, "diarizing", 8)
            try:
                diarization_segments = perform_diarization(
                    str(processed_audio_path),
                    num_speakers=speaker_count,
                    min_speakers=1,
                    max_speakers=speaker_count
                )
                (session_dir / "diarization.json").write_text(
                    json.dumps(diarization_segments, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                print(f"[{session_id}] Diarization found {len(diarization_segments)} segments")
            except Exception as e:
                print(f"[{session_id}] Diarization failed: {e} — continuing without diarization")
                diarization_segments = []

        # Transcribe — segment_callback fires every 5 segments with the last 2
        # so the API can expose a live partial_preview via status.json
        update_progress(session_id, "transcribing", 15)
        transcript, timestamps = transcribe_audio(
            str(audio_path),
            language=language,
            diarization_enabled=diarization_enabled,
            speaker_count=speaker_count,
            model_size=model_size,
            timestamps_enabled=timestamps_enabled,
            progress_callback=lambda p: update_progress(session_id, "transcribing", min(74, 15 + int(p * 0.65))),
            segment_callback=lambda segs: write_partial_preview(session_id, segs),
        )
        update_progress(session_id, "aligning", 75)

        aligned_segments = []
        if diarization_segments and timestamps:
            aligned_segments = align_diarization_with_transcript(diarization_segments, timestamps)
        elif timestamps:
            aligned_segments = [{**seg, "speaker": "SPEAKER_00"} for seg in timestamps]

        update_progress(session_id, "saving_results", 80)

        # Save canonical transcript text (complete ordered join of all segments)
        (session_dir / "transcript.txt").write_text(transcript, encoding="utf-8")

        if timestamps_enabled:
            # Ensure timestamps are in start-time order before saving
            ordered_timestamps = sorted(timestamps, key=lambda s: s.get("start", 0))
            _atomic_write_json(session_dir / "transcript_timestamps.json", ordered_timestamps)

        if aligned_segments:
            # Ensure aligned segments are in start-time order
            ordered_aligned = sorted(aligned_segments, key=lambda s: s.get("start", 0))
            _atomic_write_json(session_dir / "transcript_by_speaker.json", ordered_aligned)

            with open(session_dir / "transcript_by_speaker.txt", "w", encoding="utf-8") as f:
                for seg in ordered_aligned:
                    speaker_label = seg.get("speaker", "SPEAKER_00")
                    start_time = seg.get("start", 0)
                    end_time = seg.get("end", 0)
                    text = seg.get("text", "")
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s] {speaker_label}: {text}\n")

        if timestamps_enabled and timestamps:
            if aligned_segments:
                generate_subtitles_with_speakers(session_dir, ordered_aligned)
            else:
                generate_subtitles(session_dir, ordered_timestamps)

        _atomic_write_json(session_dir / "summary.json", generate_summary(transcript))
        _atomic_write_json(session_dir / "analytics.json", generate_analytics(transcript, timestamps))

        # ----------------------------------------------------------------
        # Continuity hints: build from session metadata + job data
        # so quality analysis can use Android-provided integrity info.
        # ----------------------------------------------------------------
        session_meta = _safe_read_json(session_dir / "v2_session.json")
        continuity_hints = _build_continuity_hints(session_meta, job_data)

        # ----------------------------------------------------------------
        # Quality analysis — runs on the aligned (or plain) segments after
        # all other processing is done.  Non-fatal: a failure here must not
        # prevent the transcript from being marked done.
        # ----------------------------------------------------------------
        quality_report: Optional[Dict] = None
        try:
            chunk_count = len(session_meta.get("chunks", []))
            total_audio_ms = sum(
                c.get("chunk_duration_ms", 0)
                for c in session_meta.get("chunks", [])
            )
            # Use aligned_segments when available (has speaker + start/end in seconds).
            # Falls back to raw timestamps (same format).
            quality_input = aligned_segments if aligned_segments else timestamps
            quality_report = analyze_transcript_quality(
                quality_input,
                chunk_count=chunk_count,
                total_audio_ms=total_audio_ms,
                continuity_hints=continuity_hints,
            )
            _atomic_write_json(session_dir / "quality_report.json", quality_report)
            print(
                f"[{session_id}] Quality analysis: "
                f"{quality_report['flagged_segment_count']} flagged / "
                f"{quality_report['segment_count']} segments, "
                f"{len(quality_report['issues'])} issue(s)"
            )
        except Exception as qe:
            print(f"[{session_id}] Quality analysis failed (non-fatal): {qe}")

        # ----------------------------------------------------------------
        # Raw transcript — technical truth: all segments + corruption flags.
        # ----------------------------------------------------------------
        try:
            quality_input_for_raw = aligned_segments if aligned_segments else timestamps
            raw_result = _build_raw_transcript(
                raw_text=transcript,
                quality_input=quality_input_for_raw,
                quality_report=quality_report,
            )
            _atomic_write_json(session_dir / "raw_transcript.json", raw_result)
            print(f"[{session_id}] Raw transcript written: {len(raw_result['segments'])} segments")
        except Exception as re_:
            print(f"[{session_id}] Raw transcript write failed (non-fatal): {re_}")

        # ----------------------------------------------------------------
        # Clean transcript — reading-oriented, conservative.
        # Uses quality_report to exclude noise and mark uncertain spans.
        # ----------------------------------------------------------------
        try:
            quality_input_for_clean = aligned_segments if aligned_segments else timestamps
            clean_result = build_clean_transcript(
                segments=quality_input_for_clean,
                quality_report=quality_report,
            )
            _atomic_write_json(session_dir / "clean_transcript.json", clean_result)
            # Plain text version for quick access
            clean_text = clean_result.get("clean_text", "")
            (session_dir / "clean_transcript.txt").write_text(clean_text, encoding="utf-8")
            print(
                f"[{session_id}] Clean transcript written: "
                f"excluded={clean_result['excluded_count']} "
                f"uncertain={clean_result['uncertainty_count']} "
                f"merges={clean_result['merge_count']}"
            )
        except Exception as ce:
            print(f"[{session_id}] Clean transcript generation failed (non-fatal): {ce}")

        # Done (records finished_at automatically)
        update_session_status(
            session_id,
            STATUS_DONE,
            progress={"upload": 100, "processing": 100, "stage": "completed"}
        )
        print(f"[{session_id}] Transcription completed successfully")

    except Exception as e:
        error_msg = str(e)
        print(f"[{session_id}] Transcription error: {error_msg}")
        tb_str = traceback.format_exc()
        traceback.print_exc()

        error_payload: Dict[str, Any] = {
            "error_type": type(e).__name__,
            "error_message": error_msg,
            "traceback": tb_str,
            "timestamp": _utcnow_str(),
            "stage": "transcription",
            "action_hint": _derive_action_hint(error_msg),
        }
        _atomic_write_json(session_dir / "error.json", error_payload)

        # Records finished_at automatically
        update_session_status(
            session_id,
            STATUS_ERROR,
            error=error_msg,
            progress={"upload": 100, "processing": 0, "stage": "error"}
        )


def process_partial_job(job_data: Dict[str, Any]) -> None:
    """
    Process a provisional partial transcription job.

    Merges available chunks (without touching the final audio.wav),
    runs a quick transcription pass, and writes partial_transcript.json.
    Always releases the partial_pending lockfile when done.

    Does NOT modify status.json — partial transcription is transparent to the
    main job pipeline and the UI progress bar.

    Does NOT produce raw_transcript.json, clean_transcript.json, or
    quality_report.json — those are final-only artifacts.
    """
    session_id = job_data["session_id"]
    session_dir = SESSIONS_DIR / session_id
    pending_flag = session_dir / "partial_pending"

    print(f"[{session_id}] Starting partial transcription")

    try:
        # Load v2_session.json to get current chunk list
        session_path = session_dir / "v2_session.json"
        session_data = _safe_read_json(session_path)

        chunks = session_data.get("chunks", [])
        if not chunks:
            print(f"[{session_id}] No chunks for partial — skipping")
            return

        # Gather chunk files in order
        chunk_paths: List[Path] = []
        for ci in sorted(chunks, key=lambda c: c["chunk_index"]):
            p = session_dir / "chunks" / f"chunk_{ci['chunk_index']:04d}.wav"
            if p.exists():
                chunk_paths.append(p)

        if not chunk_paths:
            print(f"[{session_id}] No chunk files on disk for partial — skipping")
            return

        # Merge to a temporary partial audio file (never touches audio.wav)
        partial_audio = session_dir / "audio_partial.wav"
        _merge_chunks_to_path(chunk_paths, partial_audio)

        # Transcription params (best-effort from metadata or job_data)
        metadata = _safe_read_json(session_dir / "metadata.json")
        language = job_data.get("language") or metadata.get("language") or None
        model_size = job_data.get("model_size") or metadata.get("model_size") or "base"

        transcript, timestamps = transcribe_audio(
            str(partial_audio),
            language=language,
            model_size=model_size,
            timestamps_enabled=True,
        )

        # Sort segments by start time for consistency
        ordered_ts = sorted(timestamps, key=lambda s: s.get("start", 0))
        segments = [
            {
                "start_ms": int(seg["start"] * 1000),
                "end_ms": int(seg["end"] * 1000),
                "text": seg["text"],
            }
            for seg in ordered_ts
            if seg.get("text", "").strip()
        ]

        now = _utcnow_str()
        partial_result = {
            "text": transcript,
            "segments": segments,
            "chunk_count_at_time": len(chunks),
            "generated_at": now,
            "partial_updated_at": now,      # explicit alias for API consumers
            "provisional": True,
        }
        _atomic_write_json(session_dir / "partial_transcript.json", partial_result)

        # Advance v2 session state to partially_processed (if still pre-finalization)
        if session_data.get("state") in ("receiving", "chunks_complete"):
            session_data["state"] = "partially_processed"
            _atomic_write_json(session_path, session_data)

        print(
            f"[{session_id}] Partial transcription done: "
            f"{len(segments)} segments, {len(chunks)} chunks processed"
        )

    except Exception as e:
        print(f"[{session_id}] Partial transcription failed (non-fatal): {e}")
        traceback.print_exc()
    finally:
        # Always release the lock so future chunk uploads can trigger new partials
        try:
            pending_flag.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    print("=" * 50)
    print("Starting Transcription Worker")
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print(f"Main queue: {REDIS_QUEUE}")
    print(f"Partial queue: {REDIS_PARTIAL_QUEUE}")
    print(f"Concurrency: {WORKER_CONCURRENCY}")
    print("=" * 50)

    # Connect to Redis with retries
    r = None
    max_retries = 10
    for attempt in range(max_retries):
        try:
            r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            r.ping()
            print("Connected to Redis\n")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}")
                print("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                print(f"Failed to connect to Redis after {max_retries} attempts: {e}")
                sys.exit(1)

    # GPU health check on startup — fail fast in strict CUDA mode
    try:
        from transcription import probe_cuda_health  # type: ignore

        cuda_health = probe_cuda_health()
        strict_cuda = bool(cuda_health.get("strict_cuda"))
        gpu_available = bool(cuda_health.get("gpu_available"))
        gpu_reason = cuda_health.get("gpu_reason")
        selected_compute_type = cuda_health.get("selected_compute_type")

        print(
            f"[GPU-HEALTH] strict_cuda={strict_cuda} "
            f"gpu_available={gpu_available} "
            f"selected_compute_type={selected_compute_type} "
            f"reason={gpu_reason}"
        )

        if strict_cuda and not gpu_available:
            print(
                "[GPU-HEALTH] GPU unavailable in strict CUDA mode; "
                f"failing worker startup: {gpu_reason}"
            )
            sys.exit(1)

        # Write worker health to shared /data so the web service can expose it.
        # Includes nvidia-smi utilization/memory data for the /api/v2/system/gpu endpoint.
        try:
            gpu_diag = _get_gpu_diagnostics()
            health_path = SESSIONS_DIR.parent / "worker_health.json"
            _atomic_write_json(health_path, {
                "gpu_available": gpu_available,
                "gpu_reason": gpu_reason,
                "selected_compute_type": selected_compute_type,
                "strict_cuda": strict_cuda,
                "worker_started_at": _utcnow_str(),
                # nvidia-smi data
                "gpu_name": gpu_diag.get("gpu_name"),
                "utilization_percent": gpu_diag.get("utilization_percent"),
                "memory_used_mb": gpu_diag.get("memory_used_mb"),
                "memory_total_mb": gpu_diag.get("memory_total_mb"),
            })
            print(
                f"[GPU-HEALTH] Wrote worker_health.json — "
                f"name={gpu_diag.get('gpu_name')} "
                f"util={gpu_diag.get('utilization_percent')}% "
                f"mem={gpu_diag.get('memory_used_mb')}/{gpu_diag.get('memory_total_mb')} MB"
            )
        except Exception as e:
            print(f"[GPU-HEALTH] Could not write worker_health.json: {e}")

    except Exception as e:
        print(f"[GPU-HEALTH] Failed to run CUDA health check: {e}")

    # Main event loop: drain the main queue first (priority), then process partials
    while True:
        try:
            result = r.blpop([REDIS_QUEUE, REDIS_PARTIAL_QUEUE], timeout=5)
            if result is None:
                continue

            queue_name, job_json = result
            job_data = json.loads(job_json)
            session_id = job_data["session_id"]
            job_type = job_data.get("job_type", "")

            print(f"Received job: {session_id} (type={job_type or 'legacy'}, queue={queue_name})")

            if job_type == "v2_partial":
                process_partial_job(job_data)
            else:
                # Cancel check for main jobs
                status_path = (SESSIONS_DIR / session_id) / "status.json"
                status_data = _safe_read_json(status_path)
                if status_data.get("status") == STATUS_CANCELLED:
                    print(f"[{session_id}] Job was cancelled, skipping")
                    continue

                process_transcription_job(job_data)

        except KeyboardInterrupt:
            print("\nShutting down worker...")
            break
        except Exception as e:
            print(f"Worker loop error: {e}")
            traceback.print_exc()
            time.sleep(1)


if __name__ == "__main__":
    main()
