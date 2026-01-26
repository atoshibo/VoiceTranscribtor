"""
Worker service that processes transcription jobs from Redis queue.
Runs heavy GPU operations (preprocessing, diarization, transcription).
"""
import os
import json
import redis
import time
from pathlib import Path
from typing import Dict, Optional, Any
import sys
import traceback

# Import transcription modules
from transcription import transcribe_audio, generate_summary, generate_analytics
from diarization import perform_diarization, align_diarization_with_transcript
from audio_preprocess import preprocess_audio

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "transcription_jobs")
SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", "/data/sessions"))
WORKER_CONCURRENCY = int(os.getenv("WORKER_CONCURRENCY", "1"))  # Only 1 GPU job at a time

# Status constants
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"


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
            # If file is empty (size 0), retry
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

    # Give up safely
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
    """Update session status.json file safely and atomically."""
    session_dir = SESSIONS_DIR / session_id
    status_path = session_dir / "status.json"

    # Load existing safely (could be empty/partial)
    existing = _safe_read_json(status_path)

    # Start from existing, then override with new values (IMPORTANT)
    status_data = dict(existing) if isinstance(existing, dict) else {}

    status_data.update({
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

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
# Job processing
# -----------------------------
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

        # If cancelled, skip
        if existing_status.get("status") == STATUS_CANCELLED:
            print(f"[{session_id}] Job cancelled, skipping")
            return

        # Set running
        update_session_status(session_id, STATUS_RUNNING)

        # Load metadata safely
        metadata_path = session_dir / "metadata.json"
        metadata = _safe_read_json(metadata_path)

        audio_path = session_dir / "audio.wav"
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        language = job_data.get("language", metadata.get("language", "en"))
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

        # Transcribe
        update_progress(session_id, "transcribing", 15)
        transcript, timestamps = transcribe_audio(
            str(audio_path),
            language=language,
            diarization_enabled=diarization_enabled,
            speaker_count=speaker_count,
            model_size=model_size,
            timestamps_enabled=timestamps_enabled,
            progress_callback=lambda p: update_progress(session_id, "transcribing", min(74, 15 + int(p * 0.65)))
        )
        update_progress(session_id, "aligning", 75)

        aligned_segments = []
        if diarization_segments and timestamps:
            aligned_segments = align_diarization_with_transcript(diarization_segments, timestamps)
        elif timestamps:
            aligned_segments = [{**seg, "speaker": "SPEAKER_00"} for seg in timestamps]

        update_progress(session_id, "saving_results", 80)

        # Save transcript
        (session_dir / "transcript.txt").write_text(transcript, encoding="utf-8")

        # Save timestamps
        if timestamps_enabled:
            _atomic_write_json(session_dir / "transcript_timestamps.json", timestamps)

        # Save speaker transcript
        if aligned_segments:
            _atomic_write_json(session_dir / "transcript_by_speaker.json", aligned_segments)

            with open(session_dir / "transcript_by_speaker.txt", "w", encoding="utf-8") as f:
                for seg in aligned_segments:
                    speaker_label = seg.get("speaker", "SPEAKER_00")
                    start_time = seg.get("start", 0)
                    end_time = seg.get("end", 0)
                    text = seg.get("text", "")
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s] {speaker_label}: {text}\n")

        # Subtitles
        if timestamps_enabled and timestamps:
            if aligned_segments:
                generate_subtitles_with_speakers(session_dir, aligned_segments)
            else:
                generate_subtitles(session_dir, timestamps)

        # Summary + analytics
        _atomic_write_json(session_dir / "summary.json", generate_summary(transcript))
        _atomic_write_json(session_dir / "analytics.json", generate_analytics(transcript, timestamps))

        # Done
        update_session_status(
            session_id,
            STATUS_DONE,
            progress={"upload": 100, "processing": 100, "stage": "completed"}
        )
        print(f"[{session_id}] Transcription completed successfully")

    except Exception as e:
        error_msg = str(e)
        print(f"[{session_id}] Transcription error: {error_msg}")
        traceback.print_exc()
        update_session_status(
            session_id,
            STATUS_ERROR,
            error=error_msg,
            progress={"upload": 100, "processing": 0, "stage": "error"}
        )


def main() -> None:
    print("=" * 50)
    print("Starting Transcription Worker")
    print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    print(f"Queue: {REDIS_QUEUE}")
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

    while True:
        try:
            result = r.blpop(REDIS_QUEUE, timeout=5)
            if result is None:
                continue

            _, job_json = result
            job_data = json.loads(job_json)
            session_id = job_data["session_id"]

            print(f"Received job: {session_id}")

            # Cancel check (safe read)
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
