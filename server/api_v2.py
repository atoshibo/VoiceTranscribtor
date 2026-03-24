"""
API v2 - Android-friendly REST API for VoiceRecordTranscriptor.

Key differences from v1:
- Auth via Authorization: Bearer <TOKEN> or X-Api-Token: <TOKEN> headers (no URL tokens)
- Session + chunk model: create session → upload chunks → finalize → poll → get transcript
- Rate limiting per token
- Structured JSON transcript response with ms timestamps, sorted by start time
- Partial transcription support for stream-mode sessions (every PARTIAL_EVERY_N_CHUNKS chunks)
- GPU monitoring endpoint with utilization/memory data
- Full session state machine:
    created → receiving → partially_processed → finalized → queued → running → done | error

SESSION ROLLOVER (continuous BLE usage):
  To start a new logical conversation without BLE reconnect:
    1. POST /api/v2/sessions/{old_id}/finalize
    2. POST /api/v2/sessions             → get new_session_id
    3. POST /api/v2/sessions/{new_id}/chunks  → all subsequent chunks go here
  Late chunks arriving at old_id after finalize are rejected with 409 — no duplication possible.

CONTINUITY METADATA (Android-provided audio integrity hints):
  chunk upload accepts optional form fields:
    dropped_frames    int  — BLE/audio frames dropped in this chunk (default 0)
    decode_failure    bool — whether this chunk had a decode error (default false)
    gap_before_ms     int  — reported gap before this chunk in ms (default 0)
    source_degraded   bool — whether source signal was degraded (default false)

  finalize body accepts optional field:
    session_integrity  dict  — session-level summary:
      { session_degraded: bool, total_dropped_frames: int, integrity_note: str }

  These hints improve quality analysis confidence on missing_audio_continuity issues.
  All fields are optional and backward-compatible — absence means no hints.

TRANSCRIPT RESPONSE (backward-compatible extensions):
  text          — canonical raw text (unchanged for existing clients)
  raw_text      — same as text; explicit raw-view field
  clean_text    — conservative reading-oriented text (null until processed)
  paragraphs    — clean text broken into paragraphs (null until processed)
  reading_text  — legacy alias: noise-only-excluded text from quality report
  segments      — all segments with timestamps and speaker labels
  quality_report — structured corruption analysis (null until processed)
  source_integrity — upstream audio integrity summary (null if no hints)
"""
import os
import json
import uuid
import shutil
import wave
import hashlib
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import FileResponse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", "/data/sessions"))
MAX_CHUNK_MB = int(os.getenv("MAX_CHUNK_MB", "30"))
MAX_SESSION_CHUNKS = int(os.getenv("MAX_SESSION_CHUNKS", "500"))
# How many chunks between partial transcription triggers (0 = disabled)
PARTIAL_EVERY_N_CHUNKS = int(os.getenv("PARTIAL_EVERY_N_CHUNKS", "5"))
# Minimum wall-clock seconds between partial transcription triggers for the same session.
# Prevents the GPU from being overwhelmed by repeated growing-audio partials during
# backlog catch-up uploads. Default 120s = at most one partial per 2 minutes per session.
# Set to 0 to disable the cooldown (lockfile-only concurrency guard remains).
PARTIAL_COOLDOWN_SECONDS = int(os.getenv("PARTIAL_COOLDOWN_SECONDS", "120"))

# Rate limiting — separate buckets for general API calls vs chunk uploads.
# Upload traffic (many chunks in rapid succession) needs a much higher ceiling
# than ordinary polling/finalize/transcript-fetch calls.
RATE_LIMIT_GENERAL_PER_MINUTE = int(os.getenv("RATE_LIMIT_GENERAL_PER_MINUTE",
                                               os.getenv("RATE_LIMIT_PER_MINUTE", "60")))
RATE_LIMIT_UPLOAD_PER_MINUTE = int(os.getenv("RATE_LIMIT_UPLOAD_PER_MINUTE", "300"))

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "transcription_jobs")
REDIS_PARTIAL_QUEUE = os.getenv("REDIS_PARTIAL_QUEUE", REDIS_QUEUE + "_partial")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

# Allowed model sizes — validated at finalize time
ALLOWED_MODELS = frozenset({"tiny", "base", "small", "medium", "large-v2", "large-v3"})

# Default model for full-job transcription when the client does not specify one.
# Falls back to WHISPER_MODEL env (which the worker also reads), then "small".
# Partial preview jobs always use "base" for speed regardless of this setting.
DEFAULT_FULL_MODEL: str = os.getenv(
    "WHISPER_MODEL_FULL",
    os.getenv("WHISPER_MODEL", "small"),
)
DEFAULT_PARTIAL_MODEL: str = "base"

# In-memory rate limit buckets: bucket_key → list of epoch timestamps
# General and upload traffic use separate bucket namespaces.
_rate_buckets: dict = {}

# Startup diagnostic — confirms env vars reached the container
print(
    f"[api_v2] config: MAX_CHUNK_MB={MAX_CHUNK_MB} "
    f"MAX_SESSION_CHUNKS={MAX_SESSION_CHUNKS} "
    f"PARTIAL_EVERY_N_CHUNKS={PARTIAL_EVERY_N_CHUNKS} "
    f"PARTIAL_COOLDOWN_SECONDS={PARTIAL_COOLDOWN_SECONDS} "
    f"RATE_LIMIT_GENERAL={RATE_LIMIT_GENERAL_PER_MINUTE}/min "
    f"RATE_LIMIT_UPLOAD={RATE_LIMIT_UPLOAD_PER_MINUTE}/min "
    f"DEFAULT_FULL_MODEL={DEFAULT_FULL_MODEL} "
    f"DEFAULT_PARTIAL_MODEL={DEFAULT_PARTIAL_MODEL}"
)

router = APIRouter(prefix="/api/v2", tags=["v2"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_redis():
    try:
        import redis as _redis
        r = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


def _check_rate_limit(token: str, bucket_prefix: str, limit: int) -> bool:
    """
    Sliding-window rate limit: max `limit` requests per 60s per (token, bucket_prefix).

    Different bucket_prefix values create independent counters, so upload
    traffic does not compete with general API traffic.
    """
    now = time.monotonic()
    token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
    key = f"{bucket_prefix}:{token_hash}"
    bucket = _rate_buckets.setdefault(key, [])
    bucket[:] = [t for t in bucket if now - t < 60.0]
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True


def _extract_token(request: Request) -> Optional[str]:
    """Extract token from Bearer header or X-Api-Token header only (never from URL)."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and len(auth) > 7:
        return auth[7:]
    xapi = request.headers.get("X-Api-Token", "")
    if xapi:
        return xapi
    return None


def _validate_token(request: Request) -> str:
    """Validate auth token. Returns the token string or raises 401/403."""
    token = _extract_token(request)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Use 'Authorization: Bearer <token>' or 'X-Api-Token: <token>' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if AUTH_TOKEN and token != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token.")
    return token


async def require_auth(request: Request) -> str:
    """FastAPI dependency: validate token and check GENERAL rate limit."""
    token = _validate_token(request)
    if not _check_rate_limit(token, "general", RATE_LIMIT_GENERAL_PER_MINUTE):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_GENERAL_PER_MINUTE} requests/minute).",
            headers={"Retry-After": "10"},
        )
    return token


async def require_auth_upload(request: Request) -> str:
    """
    FastAPI dependency: validate token and check UPLOAD rate limit.

    Chunk uploads happen in rapid bursts (many chunks per session, multiple
    sessions concurrently). They use a separate, higher-ceiling bucket so
    that normal polling/finalize traffic is not starved and uploads are not
    needlessly throttled.
    """
    token = _validate_token(request)
    if not _check_rate_limit(token, "upload", RATE_LIMIT_UPLOAD_PER_MINUTE):
        token_hash = hashlib.sha256(token.encode()).hexdigest()[:8]
        print(
            f"[RATE-LIMIT] Upload 429 for token_hash={token_hash} "
            f"(limit={RATE_LIMIT_UPLOAD_PER_MINUTE}/min)"
        )
        raise HTTPException(
            status_code=429,
            detail={
                "message": f"Upload rate limit exceeded ({RATE_LIMIT_UPLOAD_PER_MINUTE} requests/minute).",
                "reason": "upload_rate_limited",
                "limit_per_minute": RATE_LIMIT_UPLOAD_PER_MINUTE,
                "retry_after_seconds": 5,
                "endpoint_category": "upload",
            },
            headers={"Retry-After": "5"},
        )
    return token


def _session_dir(session_id: str) -> Path:
    return SESSIONS_DIR / session_id


def _require_v2_session(session_id: str) -> Path:
    """Return session dir or raise 404 if not a valid V2 session."""
    d = _session_dir(session_id)
    if not d.exists() or not (d / "v2_session.json").exists():
        raise HTTPException(status_code=404, detail="Session not found.")
    return d


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _derive_session_state(session_data: dict, status_data: dict) -> str:
    """
    Derive the observable session state from v2_session.json + status.json.

    Priority: worker terminal states (done/running/error) override v2 session state.
    Full state machine:
        created → receiving → partially_processed → finalized →
        queued → running → done | error
    """
    worker_status = status_data.get("status", "")
    v2_state = session_data.get("state", "created")

    if worker_status == "done":
        return "done"
    if worker_status in ("processing", "running"):
        return "running"
    if worker_status == "error":
        return "error"
    if worker_status in ("uploaded", "pending") or v2_state == "finalized":
        return "queued"
    if v2_state == "partially_processed":
        return "partially_processed"
    if v2_state in ("receiving", "chunks_complete"):
        return "receiving"
    return "created"


def _derive_backend_outcome(state: str, session_dir: Path) -> str:
    """
    Derive a stable, unambiguous backend outcome for Android reconciliation.

    This value is the single source of truth for "what happened on the server".
    Android should use this to override any local integrity heuristic when
    determining whether to show a session as ready, failed, or in-progress.

    Returns one of:
        completed       — transcription finished AND transcript file exists
        completed_empty — transcription finished but no transcript on disk (unusual)
        failed          — worker reported an error
        processing      — worker is actively transcribing
        queued          — waiting for worker pickup
        not_started     — session exists but has not been finalized yet
    """
    if state == "done":
        has_transcript = (
            (session_dir / "transcript.txt").exists()
            or (session_dir / "raw_transcript.json").exists()
        )
        return "completed" if has_transcript else "completed_empty"
    if state == "error":
        return "failed"
    if state == "running":
        return "processing"
    if state == "queued":
        return "queued"
    return "not_started"


def _derive_failure_category(error_obj: Optional[dict]) -> Optional[str]:
    """
    Classify a worker failure into a stable category Android can act on.

    Returns None for non-error states.  Categories:
        gpu_error     — CUDA / GPU / driver issue
        audio_missing — audio.wav not found on disk
        model_error   — whisper model loading failure
        internal_error — anything else
    """
    if not error_obj:
        return None

    etype = (error_obj.get("error_type") or "").lower()
    emsg = (error_obj.get("error_message") or "").lower()
    combined = f"{etype} {emsg}"

    if "filenotfounderror" in etype and "audio" in emsg:
        return "audio_missing"
    if any(kw in combined for kw in ("cuda", "gpu", "nvidia", "cublas", "cudnn", "cuinit")):
        return "gpu_error"
    if any(kw in combined for kw in ("model", "loading model", "download")):
        return "model_error"
    return "internal_error"


def _trigger_partial(session_id: str, session_dir: Path, session_data: dict) -> None:
    """
    Enqueue a partial transcription job if not already pending and the
    wall-clock cooldown since the last partial trigger has elapsed.

    Guard 1 — lockfile (partial_pending):
      Prevents queuing more than one partial job per session at a time.
      The worker removes the lockfile when done.

    Guard 2 — cooldown (PARTIAL_COOLDOWN_SECONDS):
      Prevents *sequential* partial re-triggers from consuming excessive
      GPU time during backlog/catch-up uploads.  The timestamp of the
      last trigger is stored in v2_session.json["last_partial_trigger_at"].

    If the worker crashes mid-partial the lockfile may linger and suppress
    further partial jobs. This is acceptable — the final finalize() transcript
    still works correctly.
    """
    pending_flag = session_dir / "partial_pending"
    if pending_flag.exists():
        return  # already one in flight

    # --- Cooldown guard ---
    if PARTIAL_COOLDOWN_SECONDS > 0:
        last_trigger = session_data.get("last_partial_trigger_at")
        if last_trigger:
            try:
                last_ts = datetime.fromisoformat(last_trigger.replace("Z", "+00:00"))
                elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds()
                if elapsed < PARTIAL_COOLDOWN_SECONDS:
                    return  # cooldown not yet elapsed
            except (ValueError, TypeError):
                pass  # malformed timestamp — ignore, allow trigger

    # Best-effort: read existing metadata for language/model params
    metadata: dict = {}
    metadata_path = session_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = _read_json(metadata_path)
        except Exception:
            pass

    # Claim the lock before enqueuing to avoid races
    try:
        pending_flag.touch()
    except Exception:
        return

    r = _get_redis()
    if r:
        try:
            job_data = {
                "session_id": session_id,
                "job_type": "v2_partial",
                "language": metadata.get("language"),
                "speaker_count": metadata.get("speaker_count", 1),
                "model_size": metadata.get("model_size", DEFAULT_PARTIAL_MODEL),
                "timestamps": True,
            }
            r.rpush(REDIS_PARTIAL_QUEUE, json.dumps(job_data))
            print(
                f"[V2] Enqueued partial job for session {session_id} "
                f"(cooldown={PARTIAL_COOLDOWN_SECONDS}s)"
            )
            # Record cooldown timestamp ONLY after successful enqueue.
            # Failed enqueue or no-Redis must NOT consume the cooldown window.
            session_data["last_partial_trigger_at"] = _utcnow()
            try:
                _write_json(session_dir / "v2_session.json", session_data)
            except Exception:
                pass  # best-effort — lock is already claimed, job is enqueued
        except Exception as e:
            print(f"[V2] Warning: failed to enqueue partial job: {e}")
            try:
                pending_flag.unlink(missing_ok=True)
            except Exception:
                pass
    else:
        # No Redis — release lock so next chunk can retry
        try:
            pending_flag.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WAV merging
# ---------------------------------------------------------------------------
def _measure_wav_duration(path: Path) -> Optional[float]:
    """Return duration in seconds of a WAV file, or None on error."""
    try:
        with wave.open(str(path), "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate > 0:
                return round(frames / rate, 2)
    except Exception:
        pass
    return None


def _merge_wav_chunks(session_dir: Path, chunk_paths: List[Path]) -> Path:
    """
    Merge ordered WAV chunks into a single audio.wav.
    Validates that all chunks share the same format (nchannels, sampwidth, framerate).
    Incompatible chunks are skipped with a warning.
    """
    output_path = session_dir / "audio.wav"

    if not chunk_paths:
        raise ValueError("No chunk paths provided for merging.")

    if len(chunk_paths) == 1:
        shutil.copy2(str(chunk_paths[0]), str(output_path))
        return output_path

    ref_nchannels = ref_sampwidth = ref_framerate = None
    for cp in chunk_paths:
        try:
            with wave.open(str(cp), "rb") as w:
                ref_nchannels = w.getnchannels()
                ref_sampwidth = w.getsampwidth()
                ref_framerate = w.getframerate()
            break
        except Exception as e:
            print(f"[V2-MERGE] Warning: could not read chunk {cp.name}: {e}")

    if ref_nchannels is None:
        raise ValueError("Could not read any chunk to determine WAV parameters.")

    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(ref_nchannels)
        out.setsampwidth(ref_sampwidth)
        out.setframerate(ref_framerate)

        for cp in chunk_paths:
            try:
                with wave.open(str(cp), "rb") as w:
                    if (
                        w.getnchannels() != ref_nchannels
                        or w.getsampwidth() != ref_sampwidth
                        or w.getframerate() != ref_framerate
                    ):
                        print(
                            f"[V2-MERGE] Skipping {cp.name}: incompatible format "
                            f"(ch={w.getnchannels()}, sw={w.getsampwidth()}, "
                            f"fr={w.getframerate()} vs ref {ref_nchannels}/{ref_sampwidth}/{ref_framerate})"
                        )
                        continue
                    out.writeframes(w.readframes(w.getnframes()))
            except Exception as e:
                print(f"[V2-MERGE] Warning: skipping {cp.name}: {e}")

    return output_path


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Quick health check. No auth required.

    GPU status is reported from the worker's last startup probe
    (written to /data/worker_health.json on worker startup).
    """
    gpu_available = False
    gpu_reason: Optional[str] = "worker not yet started"
    selected_device = "cuda"
    selected_compute_type = "none"
    strict_cuda = True

    worker_health_path = SESSIONS_DIR.parent / "worker_health.json"
    if worker_health_path.exists():
        try:
            wh = _read_json(worker_health_path)
            gpu_available = bool(wh.get("gpu_available", False))
            gpu_reason = wh.get("gpu_reason")
            selected_compute_type = wh.get("selected_compute_type", "none")
            strict_cuda = bool(wh.get("strict_cuda", True))
        except Exception as e:
            gpu_reason = f"could not read worker health: {e}"

    return {
        "ok": True,
        "version": "2.4.0",
        "gpu_available": gpu_available,
        "gpu_reason": gpu_reason,
        "selected_device": selected_device,
        "selected_compute_type": selected_compute_type,
        "strict_cuda": strict_cuda,
        "timestamp": _utcnow(),
    }


@router.get("/system/gpu")
async def get_gpu_status(_token: str = Depends(require_auth)):
    """
    Structured GPU/runtime diagnostics. Requires auth.

    Returns:
        GPU name, availability, compute type, utilization %, memory used/total,
        active jobs, Redis queue depths, rolling processing duration statistics.
    """
    # Worker health file (written by worker on startup, includes nvidia-smi data)
    worker_health_path = SESSIONS_DIR.parent / "worker_health.json"
    gpu_info: dict = {}
    if worker_health_path.exists():
        try:
            gpu_info = _read_json(worker_health_path)
        except Exception:
            pass

    # Scan recent sessions for active jobs + collect timing data
    active_jobs: List[dict] = []
    completed_timings: List[float] = []
    try:
        if SESSIONS_DIR.exists():
            # Sort by mtime desc, cap at 500 to avoid scanning huge session dirs
            all_dirs = sorted(
                (p for p in SESSIONS_DIR.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:500]
            for sd in all_dirs:
                sp = sd / "status.json"
                if not sp.exists():
                    continue
                try:
                    s = _read_json(sp)
                    status = s.get("status", "")
                    if status in ("processing", "running"):
                        started_at = s.get("started_at")
                        elapsed_s: Optional[float] = None
                        if started_at:
                            try:
                                t0 = datetime.fromisoformat(started_at.rstrip("Z"))
                                elapsed_s = round((datetime.utcnow() - t0).total_seconds(), 1)
                            except Exception:
                                pass
                        active_jobs.append({
                            "session_id": sd.name,
                            "started_at": started_at,
                            "updated_at": s.get("updated_at"),
                            "elapsed_s": elapsed_s,
                        })
                    elif status == "done" and s.get("started_at") and s.get("finished_at"):
                        try:
                            t0 = datetime.fromisoformat(s["started_at"].rstrip("Z"))
                            t1 = datetime.fromisoformat(s["finished_at"].rstrip("Z"))
                            dur = (t1 - t0).total_seconds()
                            if 0 < dur < 3600:
                                completed_timings.append(dur)
                        except Exception:
                            pass
                except Exception:
                    pass
    except Exception:
        pass

    # Redis queue depths
    queue_depth = 0
    partial_queue_depth = 0
    try:
        r = _get_redis()
        if r:
            queue_depth = int(r.llen(REDIS_QUEUE) or 0)
            partial_queue_depth = int(r.llen(REDIS_PARTIAL_QUEUE) or 0)
    except Exception:
        pass

    # Rolling stats from up to 50 most recent completed jobs
    recent = completed_timings[:50]
    processing_stats: dict = {}
    if recent:
        srt = sorted(recent)
        n = len(srt)
        processing_stats = {
            "count": n,
            "avg_s": round(sum(srt) / n, 1),
            "min_s": round(srt[0], 1),
            "p50_s": round(srt[n // 2], 1),
            "p95_s": round(srt[min(int(n * 0.95), n - 1)], 1),
            "max_s": round(srt[-1], 1),
        }

    return {
        "gpu_available": gpu_info.get("gpu_available", False),
        "gpu_name": gpu_info.get("gpu_name"),
        "gpu_reason": gpu_info.get("gpu_reason"),
        "selected_compute_type": gpu_info.get("selected_compute_type"),
        "strict_cuda": gpu_info.get("strict_cuda"),
        # Live GPU utilization + memory (from nvidia-smi at worker startup)
        "utilization_percent": gpu_info.get("utilization_percent"),
        "memory_used_mb": gpu_info.get("memory_used_mb"),
        "memory_total_mb": gpu_info.get("memory_total_mb"),
        "worker_started_at": gpu_info.get("worker_started_at"),
        "active_jobs": active_jobs,
        "active_job_count": len(active_jobs),
        "queue_depth": queue_depth,
        "partial_queue_depth": partial_queue_depth,
        "processing_stats": processing_stats,
        "timestamp": _utcnow(),
    }


@router.post("/sessions")
async def create_session(request: Request, _token: str = Depends(require_auth)):
    """
    Create a new recording session.

    Body (JSON, all fields optional):
        device_id           str
        client_session_id   str
        started_at_utc      ISO datetime str
        sample_rate_hz      int (default 16000) — also accepted as "sample_rate"
        channels            int (default 1)
        format              str (default "wav")
        mode                "stream" | "file" (default "stream")
        source_type         str — Android client sends this; "device_import" maps to
                            mode="file" (suppresses partial transcription triggers).
                            Other values (e.g. "ble_stream") map to mode="stream".
        chunk_duration_sec  float — stored for diagnostics (Android sends this)
        diarization         bool — stored as session-level hint for finalize default

    Session rollover for continuous BLE:
        Finalize the current session, then immediately call this to get a new
        session_id. All subsequent chunks go to the new session. The old session
        continues to be transcribed independently.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    session_id = str(uuid.uuid4())
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "chunks").mkdir(exist_ok=True)

    # --- Field alias resolution ---
    # sample_rate: accept "sample_rate" as alias for "sample_rate_hz" (Android client)
    sample_rate_hz = body.get("sample_rate_hz") or body.get("sample_rate") or 16000

    # source_type → mode mapping:
    #   "device_import" → "file"  (no partial transcription triggers)
    #   anything else   → keep explicit "mode" if set, else "stream"
    source_type = body.get("source_type")  # e.g. "ble_stream", "device_import"
    if source_type == "device_import":
        mode = "file"
    else:
        mode = body.get("mode", "stream")

    session_data = {
        "session_id": session_id,
        "client_session_id": body.get("client_session_id"),
        "device_id": body.get("device_id"),
        "started_at_utc": body.get("started_at_utc"),
        "sample_rate_hz": int(sample_rate_hz),
        "channels": body.get("channels", 1),
        "format": body.get("format", "wav"),
        "mode": mode,
        "source_type": source_type,  # preserve original for diagnostics
        "chunk_duration_sec": body.get("chunk_duration_sec"),
        "diarization_hint": bool(body.get("diarization", False)),
        "created_at": _utcnow(),
        "state": "created",
        "chunks": [],
        # Timing fields (filled in as events occur)
        "first_chunk_at": None,
        "last_chunk_at": None,
        "finalize_requested_at": None,
        # Session-level integrity hint (filled in at finalize if provided)
        "session_integrity": None,
    }

    _write_json(d / "v2_session.json", session_data)
    _write_json(d / "status.json", {"status": "created", "session_id": session_id})

    return {
        "session_id": session_id,
        "upload_url": f"/api/v2/sessions/{session_id}/chunks",
    }


@router.post("/sessions/{session_id}/chunks")
async def upload_chunk(
    session_id: str,
    file: UploadFile = File(...),
    chunk_index: int = Form(...),
    chunk_started_ms: int = Form(0),
    chunk_duration_ms: int = Form(0),
    is_final: bool = Form(False),
    # --- Continuity metadata (legacy field names) ---
    dropped_frames: int = Form(0),
    decode_failure: bool = Form(False),
    gap_before_ms: int = Form(0),
    source_degraded: bool = Form(False),
    # --- Continuity metadata (current Android field names) ---
    # These are aliases for the legacy names above.  If both the legacy and
    # the current name are sent, the current-name value wins (non-zero trumps).
    decode_errors: int = Form(0),
    ble_gaps: int = Form(0),
    plc_frames_applied: int = Form(0),
    has_continuity_warning: bool = Form(False),
    _token: str = Depends(require_auth_upload),
):
    """
    Upload one WAV chunk for a session.

    Fields (multipart/form-data):
        file              WAV audio data
        chunk_index       0-based sequential index
        chunk_started_ms  Offset from session start in ms (default 0)
        chunk_duration_ms Duration of this chunk in ms (default 0)
        is_final          true if this is the last chunk (default false)

        Continuity metadata (all optional, all default 0/false):
          Legacy names (original server contract):
            dropped_frames    BLE/audio frames dropped
            decode_failure    Whether this chunk had a decode error
            gap_before_ms     Reported gap before this chunk (ms)
            source_degraded   Whether source signal was degraded

          Current Android names (accepted as aliases):
            decode_errors          int  — maps to decode_failure (>0 = True)
            ble_gaps               int  — maps to gap_before_ms
            plc_frames_applied     int  — added to dropped_frames
            has_continuity_warning bool — maps to source_degraded

    Returns 409 if session is already finalized — ensures no chunk lands in two sessions.
    """
    # Normalize current-Android continuity fields into legacy stored schema.
    # Non-zero current-name values take precedence over zero legacy values.
    if decode_errors > 0 and not decode_failure:
        decode_failure = True
    if ble_gaps > 0 and gap_before_ms == 0:
        gap_before_ms = ble_gaps
    dropped_frames = dropped_frames + plc_frames_applied
    if has_continuity_warning and not source_degraded:
        source_degraded = True
    d = _require_v2_session(session_id)

    session_path = d / "v2_session.json"
    session_data = _read_json(session_path)

    if session_data.get("state") in ("finalized", "cancelled"):
        raise HTTPException(
            409,
            f"Session {session_id} is already finalized. "
            "Create a new session for subsequent audio.",
        )

    existing_chunks = session_data.get("chunks", [])
    if len(existing_chunks) >= MAX_SESSION_CHUNKS:
        print(f"[api_v2] 413 too_many_chunks: session={session_id} "
              f"current={len(existing_chunks)} limit={MAX_SESSION_CHUNKS} "
              f"chunk_index={chunk_index}")
        raise HTTPException(413, detail={
            "reason": "too_many_chunks",
            "message": f"Too many chunks (max {MAX_SESSION_CHUNKS}).",
            "limit": MAX_SESSION_CHUNKS,
            "current": len(existing_chunks),
        })

    # Stream-write chunk with size limit
    max_bytes = MAX_CHUNK_MB * 1024 * 1024
    chunk_path = d / "chunks" / f"chunk_{chunk_index:04d}.wav"
    bytes_written = 0

    with open(chunk_path, "wb") as f:
        while data := await file.read(65536):
            bytes_written += len(data)
            if bytes_written > max_bytes:
                f.close()
                chunk_path.unlink(missing_ok=True)
                print(f"[api_v2] 413 chunk_too_large: session={session_id} "
                      f"chunk_index={chunk_index} bytes={bytes_written} "
                      f"limit_mb={MAX_CHUNK_MB}")
                raise HTTPException(413, detail={
                    "reason": "chunk_too_large",
                    "message": f"Chunk too large (max {MAX_CHUNK_MB} MB).",
                    "limit_mb": MAX_CHUNK_MB,
                    "chunk_bytes": bytes_written,
                })
            f.write(data)

    now = _utcnow()

    # Update chunk registry (replace if same index, keeping latest upload)
    chunk_info = {
        "chunk_index": chunk_index,
        "chunk_started_ms": chunk_started_ms,
        "chunk_duration_ms": chunk_duration_ms,
        "bytes": bytes_written,
        "uploaded_at": now,
        # Continuity metadata — normalized into the canonical schema.
        # Worker _build_continuity_hints reads these exact field names.
        "dropped_frames": dropped_frames,
        "decode_failure": decode_failure,
        "gap_before_ms": gap_before_ms,
        "source_degraded": source_degraded,
    }
    # Preserve raw Android-origin fields for diagnostics (only if non-zero)
    if decode_errors:
        chunk_info["_raw_decode_errors"] = decode_errors
    if ble_gaps:
        chunk_info["_raw_ble_gaps"] = ble_gaps
    if plc_frames_applied:
        chunk_info["_raw_plc_frames_applied"] = plc_frames_applied
    chunks = [c for c in existing_chunks if c["chunk_index"] != chunk_index]
    chunks.append(chunk_info)
    chunks.sort(key=lambda c: c["chunk_index"])
    session_data["chunks"] = chunks

    # Timing fields
    if not session_data.get("first_chunk_at"):
        session_data["first_chunk_at"] = now
    session_data["last_chunk_at"] = now

    # State machine: created → receiving
    if session_data.get("state") == "created":
        session_data["state"] = "receiving"

    if is_final and session_data.get("state") == "receiving":
        session_data["state"] = "chunks_complete"

    _write_json(session_path, session_data)

    # Trigger partial transcription every N chunks for stream sessions.
    # Suppressed when:
    #   - mode != "stream" (e.g. device_import file uploads)
    #   - is_final is set (finalize imminent — partial would be wasted GPU work)
    #   - cooldown not yet elapsed (see _trigger_partial)
    total_chunks = len(chunks)
    if (
        not is_final
        and session_data.get("mode", "stream") == "stream"
        and PARTIAL_EVERY_N_CHUNKS > 0
        and total_chunks % PARTIAL_EVERY_N_CHUNKS == 0
        and session_data.get("state") in ("receiving", "partially_processed")
    ):
        try:
            _trigger_partial(session_id, d, session_data)
        except Exception as e:
            print(f"[V2] Warning: partial trigger failed: {e}")

    return {
        "accepted": True,
        "session_id": session_id,
        "chunk_index": chunk_index,
        "chunk_count": total_chunks,
        "status": "accepted",
    }


@router.post("/sessions/{session_id}/finalize")
async def finalize_session(
    session_id: str,
    request: Request,
    _token: str = Depends(require_auth),
):
    """
    Finalize session: merge chunks → audio.wav, write metadata, enqueue transcription job.

    Body (JSON, all fields optional):
        run_diarization    bool (default false)
        language           ISO-639-1 code or "auto" (default "auto" → faster-whisper autodetects)
        model_size         one of: tiny, base, small, medium, large-v2, large-v3 (default "base")
        speaker_count      int (also accepted as num_speakers for Android client compat)
        merge_strategy     "timeline" (reserved, ignored)
        session_integrity  dict — session-level audio integrity summary:
                             { session_degraded: bool, total_dropped_frames: int,
                               integrity_note: str }
                           Stored in v2_session.json and forwarded to worker for quality analysis.

    Idempotent: calling finalize twice returns the existing job_id without re-queuing.

    For session rollover (new conversation on continuous BLE stream):
        After this call succeeds, immediately POST /api/v2/sessions to start a new session.
        This session is locked against further chunk uploads (409 on any late arrival).
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    d = _require_v2_session(session_id)
    session_path = d / "v2_session.json"
    session_data = _read_json(session_path)

    state = session_data.get("state")
    if state == "finalized":
        # Idempotent: return existing job info.
        # But first check if the job is stuck (never picked up by worker).
        # If status is still "uploaded" and Redis is available, silently re-enqueue.
        status_path = d / "status.json"
        stuck_requeued = False
        if status_path.exists():
            try:
                cur_status = _read_json(status_path)
                cur_internal = cur_status.get("status", "")
                if cur_internal in ("uploaded", "pending", "created"):
                    # Job was never picked up — try to re-enqueue
                    r = _get_redis()
                    if r:
                        meta_path = d / "metadata.json"
                        meta = _read_json(meta_path) if meta_path.exists() else {}
                        retry_job = {
                            "session_id": session_id,
                            "language": meta.get("language"),
                            "speaker_count": meta.get("speaker_count", 1),
                            "diarization_enabled": meta.get("diarization_enabled", False),
                            "model_size": meta.get("model_size", DEFAULT_FULL_MODEL),
                            "timestamps": True,
                            "job_type": "v2",
                            "queued_at": _utcnow(),
                            "retry": True,
                        }
                        si = session_data.get("session_integrity")
                        if si:
                            retry_job["session_integrity"] = si
                        r.rpush(REDIS_QUEUE, json.dumps(retry_job))
                        stuck_requeued = True
                        print(
                            f"[V2-FINALIZE] Re-enqueued stuck job for session {session_id} "
                            f"(status was '{cur_internal}')"
                        )
            except Exception as e:
                print(f"[V2-FINALIZE] Stuck-job re-enqueue check failed: {e}")

        msg = "Already finalized — re-enqueued (was stuck)." if stuck_requeued else "Already finalized."
        return {
            "session_id": session_id,
            "job_id": session_id,
            "status_url": f"/api/v2/jobs/{session_id}",
            "message": msg,
            "requeued": stuck_requeued,
        }
    if state == "cancelled":
        raise HTTPException(409, "Session was cancelled.")

    chunks = session_data.get("chunks", [])
    if not chunks:
        raise HTTPException(400, "No chunks uploaded. Upload at least one chunk before finalizing.")

    # Gather chunk files in order — with merge diagnostics
    registered_indices = sorted(c["chunk_index"] for c in chunks)
    chunk_paths: List[Path] = []
    missing_indices: List[int] = []
    total_chunk_bytes = 0
    for ci in sorted(chunks, key=lambda c: c["chunk_index"]):
        p = d / "chunks" / f"chunk_{ci['chunk_index']:04d}.wav"
        if p.exists():
            chunk_paths.append(p)
            try:
                total_chunk_bytes += p.stat().st_size
            except Exception:
                pass
        else:
            missing_indices.append(ci["chunk_index"])

    if missing_indices:
        print(
            f"[V2-MERGE] WARNING session={session_id}: "
            f"{len(missing_indices)} chunk(s) missing on disk — "
            f"indices={missing_indices} "
            f"registered={len(chunks)} found={len(chunk_paths)}"
        )

    if not chunk_paths:
        raise HTTPException(400, "Chunk files missing on disk. Re-upload chunks.")

    print(
        f"[V2-MERGE] session={session_id}: "
        f"registered={len(chunks)} found={len(chunk_paths)} "
        f"indices=[{registered_indices[0]}..{registered_indices[-1]}] "
        f"missing={len(missing_indices)} "
        f"total_chunk_bytes={total_chunk_bytes}"
    )

    # Merge into audio.wav
    try:
        _merge_wav_chunks(d, chunk_paths)
    except Exception as e:
        raise HTTPException(500, f"Failed to merge audio chunks: {e}")

    # Measure merged audio duration for truthful metadata
    merged_path = d / "audio.wav"
    audio_duration_s = _measure_wav_duration(merged_path)
    merged_bytes = 0
    try:
        merged_bytes = merged_path.stat().st_size
    except Exception:
        pass
    print(
        f"[V2-MERGE] session={session_id}: merged audio.wav "
        f"bytes={merged_bytes} duration_s={audio_duration_s}"
    )

    # Build transcription options
    # Accept num_speakers as alias for speaker_count (Android client field drift compat)
    # Accept "diarization" as legacy alias for "run_diarization" (older Android clients)
    language_raw: str = body.get("language", "auto")
    run_diarization: bool = bool(
        body.get("run_diarization")
        or body.get("diarization")
        or False
    )
    _diarization_field_used = (
        "run_diarization" if "run_diarization" in body
        else ("diarization" if "diarization" in body else "default(false)")
    )

    model_size_raw = body.get("model_size") or None
    if model_size_raw is not None and model_size_raw not in ALLOWED_MODELS:
        raise HTTPException(
            400,
            f"Invalid model_size '{model_size_raw}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_MODELS))}",
        )
    model_size: str = model_size_raw or DEFAULT_FULL_MODEL

    _speaker_count_field_used = (
        "speaker_count" if "speaker_count" in body
        else ("num_speakers" if "num_speakers" in body else "default")
    )
    speaker_count: int = int(
        body.get("speaker_count")
        or body.get("num_speakers")
        or (2 if run_diarization else 1)
    )

    # Safety: if diarization is requested but speaker_count ended up < 2
    # (e.g., client sent speaker_count=1 with diarization=true), force to 2.
    if run_diarization and speaker_count < 2:
        print(
            f"[V2-FINALIZE] session={session_id}: diarization enabled but "
            f"speaker_count={speaker_count} (from {_speaker_count_field_used}) "
            f"— forcing speaker_count=2"
        )
        speaker_count = 2

    print(
        f"[V2-FINALIZE] session={session_id}: diarization={run_diarization} "
        f"(field={_diarization_field_used}) speaker_count={speaker_count} "
        f"(field={_speaker_count_field_used})"
    )

    # "auto" → pass None to faster-whisper for language auto-detection (never coerce to "en")
    language_for_worker = None if language_raw == "auto" else language_raw

    now = _utcnow()

    # Optional session-level integrity hint from Android app
    session_integrity = body.get("session_integrity") or None

    # Write metadata.json (consumed by worker)
    metadata = {
        "session_id": session_id,
        "language": language_for_worker,
        "speaker_count": speaker_count,
        "model_size": model_size,
        "timestamps": True,
        "diarization_enabled": run_diarization,
        "audio_duration_s": audio_duration_s,
        "merge_info": {
            "registered_chunks": len(chunks),
            "merged_chunks": len(chunk_paths),
            "missing_chunks": len(missing_indices),
            "merged_bytes": merged_bytes,
        },
        "created_at": session_data.get("created_at"),
        "v2": True,
    }
    _write_json(d / "metadata.json", metadata)

    # Update status to "uploaded" so the worker picks it up
    _write_json(
        d / "status.json",
        {
            "status": "uploaded",
            "session_id": session_id,
            "queued_at": now,
            "progress": {"upload": 100, "processing": 0, "stage": "queued"},
        },
    )

    # Lock the session against further chunk uploads
    session_data["state"] = "finalized"
    session_data["finalize_requested_at"] = now
    if not session_data.get("finalized_at"):
        session_data["finalized_at"] = now
    # Persist session_integrity in v2_session.json for later transcript read
    if session_integrity:
        session_data["session_integrity"] = session_integrity
    _write_json(session_path, session_data)

    # Enqueue to Redis — include session_integrity so worker can use it
    job_data = {
        "session_id": session_id,
        "language": language_for_worker,
        "speaker_count": speaker_count,
        "diarization_enabled": run_diarization,
        "model_size": model_size,
        "timestamps": True,
        "job_type": "v2",
        "queued_at": now,
    }
    if session_integrity:
        job_data["session_integrity"] = session_integrity

    enqueued = False
    r = _get_redis()
    if r:
        try:
            r.rpush(REDIS_QUEUE, json.dumps(job_data))
            enqueued = True
            print(
                f"[V2-FINALIZE] Accepted session={session_id} "
                f"chunks={len(chunks)} model={model_size} "
                f"speakers={speaker_count} lang={language_for_worker or 'auto'} "
                f"diarization={run_diarization} (field={_diarization_field_used}) "
                f"integrity={'yes' if session_integrity else 'no'}"
            )
        except Exception as e:
            print(f"[V2-FINALIZE] ENQUEUE FAILED session={session_id}: {e}")
    else:
        print(
            f"[V2-FINALIZE] REDIS UNAVAILABLE session={session_id} — "
            "job NOT enqueued. Client should retry finalize."
        )

    return {
        "session_id": session_id,
        "job_id": session_id,
        "status_url": f"/api/v2/jobs/{session_id}",
        "enqueued": enqueued,
    }


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str, _token: str = Depends(require_auth)):
    """
    Poll transcription job status.

    States: queued | running | done | error

    Reconciliation fields (new, backward-compatible):
        backend_outcome     — stable truth: "completed" | "completed_empty" | "failed" |
                              "processing" | "queued". Android SHOULD prefer this over
                              local integrity heuristics when deciding what to show the user.
        transcript_present  — true iff a transcript file exists on disk
        failure_category    — null unless failed: "gpu_error" | "audio_missing" |
                              "model_error" | "internal_error"
        partial_transcript_available — true iff a partial transcript was generated

    progress: {upload: 0-100, processing: 0-100, stage: str}
    partial_preview: last 1-2 transcribed sentences while running, null otherwise
    """
    session_dir = SESSIONS_DIR / job_id
    if not session_dir.exists():
        raise HTTPException(404, "Job not found.")

    status_path = session_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(404, "Job not found.")

    status_data = _read_json(status_path)
    internal = status_data.get("status", "unknown")

    state_map = {
        "created":    "queued",
        "uploaded":   "queued",
        "pending":    "queued",
        "processing": "running",
        "running":    "running",
        "done":       "done",
        "error":      "error",
        "cancelled":  "error",
    }
    state = state_map.get(internal, "queued")

    progress_data = status_data.get("progress", {})
    if not isinstance(progress_data, dict):
        progress_data = {}

    error_obj: Optional[dict] = None
    message: str = progress_data.get("stage", "") or ""

    if state == "error":
        error_path = session_dir / "error.json"
        if error_path.exists():
            try:
                error_obj = _read_json(error_path)
            except Exception:
                error_obj = None

        if error_obj and error_obj.get("error_message"):
            message = str(error_obj["error_message"])
        elif status_data.get("error"):
            message = str(status_data["error"])
        elif not message:
            message = "Job failed"

    # Extra metadata
    meta: dict = {}
    metadata_path = session_dir / "metadata.json"
    if metadata_path.exists():
        try:
            md = _read_json(metadata_path)
            meta["language_requested"] = md.get("language")  # None = auto
            meta["model_size"] = md.get("model_size")
            meta["speaker_count"] = md.get("speaker_count")
            meta["diarization_enabled"] = md.get("diarization_enabled", False)
            # Audio duration measured at merge time (truthful, not transcript-derived)
            if md.get("audio_duration_s") is not None:
                meta["audio_duration_s"] = md["audio_duration_s"]
        except Exception:
            pass

    if state == "done":
        tsp = session_dir / "transcript_timestamps.json"
        if tsp.exists():
            try:
                segs = json.loads(tsp.read_text(encoding="utf-8"))
                meta["segment_count"] = len(segs)
                if segs:
                    # Backward compat: duration_s = last segment end (transcript coverage)
                    meta["duration_s"] = round(segs[-1].get("end", 0), 1)
            except Exception:
                pass
        ana = session_dir / "analytics.json"
        if ana.exists():
            try:
                a = _read_json(ana)
                meta.setdefault("duration_s", a.get("duration"))
                meta["word_count"] = a.get("word_count")
            except Exception:
                pass
        # Coverage metadata from worker (truthful audio vs transcript comparison)
        cov_path = session_dir / "coverage_meta.json"
        if cov_path.exists():
            try:
                cov = _read_json(cov_path)
                # audio_duration_s: actual WAV file duration (ground truth)
                meta.setdefault("audio_duration_s", cov.get("audio_duration_s"))
                # transcript_last_end_s: where the transcript actually ends
                meta["transcript_last_end_s"] = cov.get("transcript_last_end_s")
                # coverage_ratio: transcript_end / audio_duration (1.0 = full coverage)
                meta["transcript_coverage_ratio"] = cov.get("transcript_coverage_ratio")
                # Warning flag if coverage is suspiciously low
                if cov.get("coverage_warning"):
                    meta["coverage_warning"] = cov["coverage_warning"]
            except Exception:
                pass
        # Fallback: measure audio.wav directly if no coverage_meta.json yet
        if "audio_duration_s" not in meta or meta["audio_duration_s"] is None:
            audio_path = session_dir / "audio.wav"
            if audio_path.exists():
                dur = _measure_wav_duration(audio_path)
                if dur is not None:
                    meta["audio_duration_s"] = dur
        # Classification summary (if available)
        clf_path = session_dir / "classification.json"
        if clf_path.exists():
            try:
                clf = _read_json(clf_path)
                meta["classification"] = clf.get("category")
                meta["classification_confidence"] = clf.get("confidence")
            except Exception:
                pass

    # Live partial preview: populated by worker's segment_callback every 5 segments.
    # Only returned while running; null in all other states.
    partial_preview: Optional[str] = None
    if state == "running":
        partial_preview = status_data.get("partial_preview") or None

    # ----- Reconciliation fields (Android truth source) -----
    backend_outcome = _derive_backend_outcome(state, session_dir)
    transcript_present = (
        (session_dir / "transcript.txt").exists()
        or (session_dir / "raw_transcript.json").exists()
    )
    failure_category = _derive_failure_category(error_obj) if state == "error" else None
    partial_transcript_available = (session_dir / "partial_transcript.json").exists()

    response = {
        "job_id": job_id,
        "state": state,
        # Reconciliation: Android should use these to override local integrity guesses
        "backend_outcome": backend_outcome,
        "transcript_present": transcript_present,
        "failure_category": failure_category,
        "partial_transcript_available": partial_transcript_available,
        "progress": progress_data,
        "message": message,
        "partial_preview": partial_preview,
        "queued_at": status_data.get("queued_at"),
        "started_at": status_data.get("started_at") or status_data.get("created_at"),
        "finished_at": (
            status_data.get("finished_at")
            or status_data.get("completed_at")
            or (status_data.get("updated_at") if state in ("done", "error") else None)
        ),
        "meta": meta,
    }

    if state == "error":
        if error_obj is not None:
            response["error"] = error_obj
        elif status_data.get("error"):
            response["error"] = {
                "error_type": "RuntimeError",
                "error_message": status_data.get("error"),
            }

    return response


@router.get("/jobs/{job_id}/error")
async def get_job_error(job_id: str, _token: str = Depends(require_auth)):
    """
    Retrieve detailed error information for a failed job.
    Returns 404 if the job or error.json is not found.
    """
    session_dir = SESSIONS_DIR / job_id
    if not session_dir.exists():
        raise HTTPException(404, "Job not found.")

    error_path = session_dir / "error.json"
    if not error_path.exists():
        raise HTTPException(404, "Error details not available for this job.")

    return _read_json(error_path)


@router.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str, _token: str = Depends(require_auth)):
    """
    Rich session status — covers the full state machine including pre-finalization states
    (created / receiving / partially_processed) not visible in /jobs/{id}.

    Reconciliation fields (new, backward-compatible):
        backend_outcome     — stable truth: "completed" | "completed_empty" | "failed" |
                              "processing" | "queued" | "not_started".
                              Android SHOULD prefer this over local integrity guesses.
        transcript_present  — true iff a transcript file exists on disk
        error_message       — human-readable error description (null unless failed)

    Designed to drive native Android UX:
      - Poll every 5-10s during a live recording to show chunk count + partial preview
      - Transition to polling /jobs/{id} after finalize for worker progress
      - Check finished_at to decide when to fetch final /transcript
      - Use backend_outcome to decide card color / status label
    """
    d = _require_v2_session(session_id)

    session_data = _read_json(d / "v2_session.json")
    status_data: dict = {}
    if (d / "status.json").exists():
        try:
            status_data = _read_json(d / "status.json")
        except Exception:
            pass

    state = _derive_session_state(session_data, status_data)
    chunks = session_data.get("chunks", [])
    total_duration_ms = sum(c.get("chunk_duration_ms", 0) for c in chunks)

    # Last uploaded chunk index
    last_chunk_index: Optional[int] = None
    if chunks:
        last_chunk_index = max(c["chunk_index"] for c in chunks)

    # Partial transcript data
    partial_available = False
    partial_preview: Optional[str] = None
    partial_updated_at: Optional[str] = None
    last_processed_chunk_index: Optional[int] = None

    partial_path = d / "partial_transcript.json"
    if partial_path.exists():
        partial_available = True
        try:
            pt = _read_json(partial_path)
            partial_updated_at = pt.get("partial_updated_at") or pt.get("generated_at")
            cnt = pt.get("chunk_count_at_time", 0)
            if cnt > 0:
                last_processed_chunk_index = cnt - 1
            # Show partial preview for pre-final states
            if state in ("receiving", "partially_processed"):
                segs = pt.get("segments", [])
                if segs:
                    partial_preview = " ".join(
                        s.get("text", "") for s in segs[-2:]
                    ).strip() or None
        except Exception:
            pass

    # During final transcription, prefer the live segment_callback preview
    if state == "running":
        live = status_data.get("partial_preview")
        if live:
            partial_preview = live

    # ----- Reconciliation fields -----
    backend_outcome = _derive_backend_outcome(state, d)
    transcript_present = (
        (d / "transcript.txt").exists()
        or (d / "raw_transcript.json").exists()
    )

    # Error message — extract from status.json or error.json
    error_message: Optional[str] = None
    if state == "error":
        error_message = status_data.get("error")
        if not error_message:
            error_path = d / "error.json"
            if error_path.exists():
                try:
                    ej = _read_json(error_path)
                    error_message = ej.get("error_message")
                except Exception:
                    pass
        if not error_message:
            error_message = "Job failed"

    return {
        "session_id": session_id,
        "state": state,
        # Reconciliation: Android should use these to override local integrity guesses
        "backend_outcome": backend_outcome,
        "transcript_present": transcript_present,
        "error_message": error_message,
        # Upload progress
        "chunk_count": len(chunks),
        "chunks_received": len(chunks),       # alias for Android compat
        "last_chunk_index": last_chunk_index,
        "total_audio_ms": total_duration_ms,
        # Partial transcript
        "partial_transcript_available": partial_available,
        "partial_preview": partial_preview,
        "partial_updated_at": partial_updated_at,
        "last_processed_chunk_index": last_processed_chunk_index,
        # Timing
        "created_at": session_data.get("created_at"),
        "first_chunk_at": session_data.get("first_chunk_at"),
        "last_chunk_at": session_data.get("last_chunk_at"),
        "finalize_requested_at": session_data.get("finalize_requested_at"),
        "queued_at": status_data.get("queued_at"),
        "started_at": status_data.get("started_at"),
        "finished_at": status_data.get("finished_at"),
        # Worker progress (populated during / after transcription)
        "progress": status_data.get("progress", {}),
        "mode": session_data.get("mode", "stream"),
        "device_id": session_data.get("device_id"),
    }


@router.get("/sessions/{session_id}/transcript/partial")
async def get_partial_transcript(session_id: str, _token: str = Depends(require_auth)):
    """
    Get the latest provisional partial transcript for a live recording session.

    Available once at least one partial transcription has been processed
    (state: partially_processed or later). Always marked provisional=true.

    Returns 404 (not an error) if no partial is ready yet — the app should
    poll /sessions/{id}/status.partial_transcript_available before calling this.

    The final authoritative transcript is at /sessions/{id}/transcript.
    Partial transcripts are never processed through quality analysis or clean
    transcript generation — those are final-only artifacts.
    """
    d = _require_v2_session(session_id)

    partial_path = d / "partial_transcript.json"
    if not partial_path.exists():
        raise HTTPException(
            404,
            detail={
                "message": "No partial transcript available yet.",
                "hint": (
                    f"Partial transcripts are generated every {PARTIAL_EVERY_N_CHUNKS} chunks "
                    "for stream-mode sessions. Upload more chunks or wait for the worker."
                ),
            },
        )

    try:
        partial = _read_json(partial_path)
    except Exception as e:
        raise HTTPException(500, f"Failed to read partial transcript: {e}")

    return {
        "session_id": session_id,
        "provisional": True,
        "text": partial.get("text", ""),
        "segments": partial.get("segments", []),
        "chunk_count_at_time": partial.get("chunk_count_at_time", 0),
        "generated_at": partial.get("generated_at"),
        "partial_updated_at": partial.get("partial_updated_at") or partial.get("generated_at"),
    }


@router.get("/sessions/{session_id}/transcript")
async def get_transcript(session_id: str, _token: str = Depends(require_auth)):
    """
    Get the final authoritative transcript for a completed session.

    Returns 202 if processing is not yet complete.

    Response fields (backward-compatible — existing clients using text/segments unchanged):

    text          — canonical raw text (complete ordered join of all recognized segments)
    raw_text      — same as text; explicit raw-view field for clients that want clarity
    clean_text    — reading-oriented conservative text (null until worker writes it)
    paragraphs    — clean text as paragraph list (null until worker writes it)
    reading_text  — legacy: noise-excluded text from quality report (null until quality done)
    segments      — all segments sorted by start_ms with speaker labels and corruption_flags
    words         — word-level timestamps (if available)
    quality_report — structured corruption analysis (null until worker writes it)
    source_integrity — upstream audio damage summary from Android hints (null if no hints)
    classification — LLM-based transcript category (null if not configured or not run)

    Clients that only use text/segments are completely unaffected.
    """
    d = _require_v2_session(session_id)

    status_path = d / "status.json"
    status_data = _read_json(status_path) if status_path.exists() else {}
    internal_status = status_data.get("status", "unknown")

    if internal_status != "done":
        raise HTTPException(
            status_code=202,
            detail={
                "message": "Transcript not ready yet.",
                "state": internal_status,
                "progress": status_data.get("progress", {}),
            },
        )

    # -----------------------------------------------------------------------
    # Segments — primary source: raw_transcript.json (has corruption_flags).
    # Fallback chain: transcript_by_speaker.json → transcript_timestamps.json.
    # Always sorted by start_ms; empty-text entries filtered out.
    # -----------------------------------------------------------------------
    segments: List[dict] = []

    raw_trans_path = d / "raw_transcript.json"
    if raw_trans_path.exists():
        try:
            raw_data = _read_json(raw_trans_path)
            raw_segs = raw_data.get("segments", [])
            segments = sorted(
                [s for s in raw_segs if s.get("text", "").strip()],
                key=lambda s: s.get("start_ms", 0),
            )
        except Exception:
            pass

    if not segments:
        sp = d / "transcript_by_speaker.json"
        if sp.exists():
            try:
                raw = json.loads(sp.read_text(encoding="utf-8"))
                segments = sorted(
                    [
                        {
                            "start_ms": int(seg.get("start", 0) * 1000),
                            "end_ms": int(seg.get("end", 0) * 1000),
                            "speaker": seg.get("speaker", "SPEAKER_00"),
                            "text": seg.get("text", "").strip(),
                            "corruption_flags": [],
                        }
                        for seg in raw
                        if seg.get("text", "").strip()
                    ],
                    key=lambda s: s["start_ms"],
                )
            except Exception:
                pass

    if not segments:
        tsp_path = d / "transcript_timestamps.json"
        if tsp_path.exists():
            try:
                ts_data = json.loads(tsp_path.read_text(encoding="utf-8"))
                segments = sorted(
                    [
                        {
                            "start_ms": int(seg.get("start", 0) * 1000),
                            "end_ms": int(seg.get("end", 0) * 1000),
                            "speaker": "SPEAKER_00",
                            "text": seg.get("text", "").strip(),
                            "corruption_flags": [],
                        }
                        for seg in ts_data
                        if seg.get("text", "").strip()
                    ],
                    key=lambda s: s["start_ms"],
                )
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Canonical raw text
    # Primary: transcript.txt (written by worker from all segments in order)
    # Fallback: reconstruct from sorted segments (completeness guarantee)
    # -----------------------------------------------------------------------
    raw_text = ""
    tp = d / "transcript.txt"
    if tp.exists():
        raw_text = tp.read_text(encoding="utf-8").strip()

    if not raw_text and segments:
        raw_text = " ".join(s["text"] for s in segments)

    # -----------------------------------------------------------------------
    # Clean transcript — reading-oriented; produced by worker after quality analysis.
    # -----------------------------------------------------------------------
    clean_text: Optional[str] = None
    paragraphs: Optional[List[dict]] = None

    clean_trans_path = d / "clean_transcript.json"
    if clean_trans_path.exists():
        try:
            clean_data = _read_json(clean_trans_path)
            clean_text = clean_data.get("clean_text") or None
            paragraphs = clean_data.get("paragraphs") or None
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Word-level timestamps (optional, from transcript_timestamps.json)
    # -----------------------------------------------------------------------
    words: List[dict] = []
    tsp = d / "transcript_timestamps.json"
    if tsp.exists():
        try:
            ts_data = json.loads(tsp.read_text(encoding="utf-8"))
            for seg in ts_data:
                for w in seg.get("words", []):
                    word_text = w.get("word", "").strip()
                    if word_text:
                        words.append(
                            {
                                "t_ms": int(w.get("start", seg.get("start", 0)) * 1000),
                                "speaker": "SPEAKER_00",
                                "w": word_text,
                            }
                        )
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Quality report (written by worker after transcription — optional).
    # reading_text comes from the report when available; falls back to raw_text.
    # -----------------------------------------------------------------------
    quality_report: Optional[dict] = None
    qrp = d / "quality_report.json"
    if qrp.exists():
        try:
            quality_report = _read_json(qrp)
        except Exception:
            pass

    reading_text: Optional[str] = None
    if quality_report is not None:
        reading_text = quality_report.get("reading_text") or raw_text

    # -----------------------------------------------------------------------
    # Source integrity — from quality report or session metadata
    # -----------------------------------------------------------------------
    source_integrity: Optional[dict] = None
    if quality_report is not None:
        source_integrity = quality_report.get("source_integrity")
    if source_integrity is None:
        # Fallback: read from v2_session.json (stored at finalize time)
        try:
            v2_data = _read_json(d / "v2_session.json")
            si = v2_data.get("session_integrity")
            if si:
                source_integrity = {
                    "hints_available": True,
                    "session_degraded": bool(si.get("session_degraded", False)),
                    "total_dropped_frames": int(si.get("total_dropped_frames", 0)),
                    "chunks_with_decode_failure": None,  # session-level only, no per-chunk count
                    "chunks_with_gaps": None,
                    "integrity_note": si.get("integrity_note"),
                }
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Classification — LLM-based transcript category (optional, null if not run)
    # -----------------------------------------------------------------------
    classification: Optional[dict] = None
    clf_path = d / "classification.json"
    if clf_path.exists():
        try:
            classification = _read_json(clf_path)
        except Exception:
            pass

    return {
        "session_id": session_id,
        # Backward-compatible canonical field — never loses recognized content
        "text": raw_text,
        # Explicit raw view (same as text; for clients that want clarity)
        "raw_text": raw_text,
        # Reading-oriented conservative text (null until worker writes clean_transcript.json)
        "clean_text": clean_text,
        # Clean text as paragraphs (null until worker writes clean_transcript.json)
        "paragraphs": paragraphs,
        # Legacy field: noise-excluded text from quality report
        "reading_text": reading_text,
        "segments": segments,
        "words": words,
        "formats": {
            "srt_url": f"/api/v2/sessions/{session_id}/subtitle.srt",
            "vtt_url": f"/api/v2/sessions/{session_id}/subtitle.vtt",
        },
        # Structured quality analysis (null until worker produces quality_report.json)
        "quality_report": quality_report,
        # Upstream audio damage summary (null if no Android continuity hints were provided)
        "source_integrity": source_integrity,
        # LLM-based transcript classification (null if not configured or not yet run)
        "classification": classification,
    }


@router.get("/sessions/{session_id}/subtitle.srt")
async def get_srt(session_id: str, _token: str = Depends(require_auth)):
    d = _require_v2_session(session_id)
    p = d / "subtitles.srt"
    if not p.exists():
        raise HTTPException(404, "SRT subtitle not available (session not done yet?).")
    return FileResponse(str(p), media_type="text/plain", filename=f"{session_id}.srt")


@router.get("/sessions/{session_id}/subtitle.vtt")
async def get_vtt(session_id: str, _token: str = Depends(require_auth)):
    d = _require_v2_session(session_id)
    p = d / "subtitles.vtt"
    if not p.exists():
        raise HTTPException(404, "VTT subtitle not available (session not done yet?).")
    return FileResponse(str(p), media_type="text/vtt", filename=f"{session_id}.vtt")


@router.post("/sessions/{session_id}/retry")
async def retry_session(session_id: str, _token: str = Depends(require_auth)):
    """
    Re-enqueue a finalized session whose job failed or was never picked up.

    Allowed only when the session is finalized AND the job status is one of:
      error, uploaded, pending, created (i.e. failed or stuck).

    Running or already-done jobs cannot be retried (returns 409).
    Resets status.json to "uploaded" and pushes a new job to Redis.

    Idempotent with respect to the audio: no re-merge needed (audio.wav already exists).
    """
    d = _require_v2_session(session_id)
    session_data = _read_json(d / "v2_session.json")

    if session_data.get("state") != "finalized":
        raise HTTPException(
            409,
            "Session is not finalized. Finalize first, then retry if it fails.",
        )

    status_path = d / "status.json"
    if not status_path.exists():
        raise HTTPException(500, "Missing status.json for finalized session.")

    cur_status = _read_json(status_path)
    cur_internal = cur_status.get("status", "")

    # Only allow retry for non-terminal-success, non-running states
    if cur_internal == "done":
        raise HTTPException(409, "Job already completed successfully. Nothing to retry.")
    if cur_internal in ("processing", "running"):
        raise HTTPException(409, "Job is currently running. Wait for it to finish.")

    # Verify audio.wav exists (should exist from original finalize merge)
    if not (d / "audio.wav").exists():
        raise HTTPException(
            500,
            "audio.wav missing — cannot retry. Re-upload chunks and finalize again.",
        )

    now = _utcnow()

    # Reset status to uploaded/queued
    _write_json(status_path, {
        "status": "uploaded",
        "session_id": session_id,
        "queued_at": now,
        "progress": {"upload": 100, "processing": 0, "stage": "queued"},
        "retry_at": now,
        "previous_status": cur_internal,
    })

    # Read metadata for job params
    meta_path = d / "metadata.json"
    meta = _read_json(meta_path) if meta_path.exists() else {}

    job_data = {
        "session_id": session_id,
        "language": meta.get("language"),
        "speaker_count": meta.get("speaker_count", 1),
        "diarization_enabled": meta.get("diarization_enabled", False),
        "model_size": meta.get("model_size", DEFAULT_FULL_MODEL),
        "timestamps": True,
        "job_type": "v2",
        "queued_at": now,
        "retry": True,
    }
    si = session_data.get("session_integrity")
    if si:
        job_data["session_integrity"] = si

    r = _get_redis()
    if not r:
        raise HTTPException(503, "Redis unavailable. Retry later.")

    try:
        r.rpush(REDIS_QUEUE, json.dumps(job_data))
    except Exception as e:
        raise HTTPException(503, f"Failed to enqueue retry job: {e}")

    print(
        f"[V2-RETRY] Re-enqueued session {session_id} "
        f"(previous_status={cur_internal})"
    )

    return {
        "session_id": session_id,
        "job_id": session_id,
        "status_url": f"/api/v2/jobs/{session_id}",
        "message": "Job re-enqueued for retry.",
        "previous_status": cur_internal,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, _token: str = Depends(require_auth)):
    """Delete all session data (audio, chunks, transcripts). GDPR-ish."""
    d = _require_v2_session(session_id)
    shutil.rmtree(str(d))
    return {"deleted": True, "session_id": session_id}
