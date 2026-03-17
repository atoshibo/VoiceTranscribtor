"""
API v2 - Android-friendly REST API for VoiceRecordTranscriptor.

Key differences from v1:
- Auth via Authorization: Bearer <TOKEN> or X-Api-Token: <TOKEN> headers (no URL tokens)
- Session + chunk model: create session → upload chunks → finalize → poll → get transcript
- Rate limiting per token
- Structured JSON transcript response with ms timestamps
- Partial transcription support for stream-mode sessions (every PARTIAL_EVERY_N_CHUNKS chunks)
- GPU monitoring endpoint
- Full session state machine: created → receiving → partially_processed → finalized →
  queued → running → done | error
"""
import os
import json
import uuid
import shutil
import wave
import hashlib
import time
import subprocess
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
MAX_SESSION_CHUNKS = int(os.getenv("MAX_SESSION_CHUNKS", "200"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
# How many chunks between partial transcription triggers (0 = disabled)
PARTIAL_EVERY_N_CHUNKS = int(os.getenv("PARTIAL_EVERY_N_CHUNKS", "5"))

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "transcription_jobs")
REDIS_PARTIAL_QUEUE = os.getenv("REDIS_PARTIAL_QUEUE", REDIS_QUEUE + "_partial")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")

# In-memory rate limit buckets: token_hash → list of epoch timestamps
_rate_buckets: dict = {}

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


def _check_rate_limit(token: str) -> bool:
    """Sliding-window rate limit: max RATE_LIMIT_PER_MINUTE requests/60s per token."""
    now = time.monotonic()
    key = hashlib.sha256(token.encode()).hexdigest()[:16]
    bucket = _rate_buckets.setdefault(key, [])
    bucket[:] = [t for t in bucket if now - t < 60.0]
    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
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


async def require_auth(request: Request) -> str:
    """FastAPI dependency: validate token and check rate limit."""
    token = _extract_token(request)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Use 'Authorization: Bearer <token>' or 'X-Api-Token: <token>' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if AUTH_TOKEN and token != AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token.")
    if not _check_rate_limit(token):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({RATE_LIMIT_PER_MINUTE} requests/minute).",
            headers={"Retry-After": "60"},
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


def _trigger_partial(session_id: str, session_dir: Path, session_data: dict) -> None:
    """
    Enqueue a partial transcription job if not already pending.

    Uses a lockfile (partial_pending) to prevent queuing more than one
    partial job per session at a time. The worker removes the lockfile when done.

    Note: if the worker crashes mid-partial, the lockfile may linger and suppress
    further partial jobs. This is acceptable — the final finalize() transcript will
    still work correctly.
    """
    pending_flag = session_dir / "partial_pending"
    if pending_flag.exists():
        return  # already one in flight

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
                "model_size": metadata.get("model_size", "base"),
                "timestamps": True,
            }
            r.rpush(REDIS_PARTIAL_QUEUE, json.dumps(job_data))
            print(f"[V2] Enqueued partial job for session {session_id}")
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

    # Read reference parameters from first valid chunk
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
        "version": "2.1.0",
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
        GPU availability, compute type, active jobs, Redis queue depths,
        rolling processing duration statistics (from recently finished jobs).
    """
    # Worker health file (written by worker on startup)
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
                        active_jobs.append({
                            "session_id": sd.name,
                            "started_at": s.get("started_at"),
                            "updated_at": s.get("updated_at"),
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
        "gpu_reason": gpu_info.get("gpu_reason"),
        "selected_compute_type": gpu_info.get("selected_compute_type"),
        "strict_cuda": gpu_info.get("strict_cuda"),
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

    Body (JSON):
        device_id, client_session_id, started_at_utc, sample_rate_hz,
        channels, format, mode ("stream" | "file")
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    session_id = str(uuid.uuid4())
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "chunks").mkdir(exist_ok=True)

    session_data = {
        "session_id": session_id,
        "client_session_id": body.get("client_session_id"),
        "device_id": body.get("device_id"),
        "started_at_utc": body.get("started_at_utc"),
        "sample_rate_hz": body.get("sample_rate_hz", 16000),
        "channels": body.get("channels", 1),
        "format": body.get("format", "wav"),
        "mode": body.get("mode", "stream"),
        "created_at": _utcnow(),
        "state": "created",
        "chunks": [],
        # Timing fields (filled in as events occur)
        "first_chunk_at": None,
        "last_chunk_at": None,
        "finalize_requested_at": None,
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
    _token: str = Depends(require_auth),
):
    """
    Upload one WAV chunk for a session.

    Fields (multipart/form-data):
        file             WAV audio data
        chunk_index      0-based sequential index
        chunk_started_ms Offset from session start in ms
        chunk_duration_ms Duration of this chunk in ms
        is_final         true if this is the last chunk
    """
    d = _require_v2_session(session_id)

    session_path = d / "v2_session.json"
    session_data = _read_json(session_path)

    if session_data.get("state") in ("finalized", "cancelled"):
        raise HTTPException(409, "Session already finalized or cancelled.")

    existing_chunks = session_data.get("chunks", [])
    if len(existing_chunks) >= MAX_SESSION_CHUNKS:
        raise HTTPException(413, f"Too many chunks (max {MAX_SESSION_CHUNKS}).")

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
                raise HTTPException(413, f"Chunk too large (max {MAX_CHUNK_MB} MB).")
            f.write(data)

    now = _utcnow()

    # Update chunk registry (replace if same index)
    chunk_info = {
        "chunk_index": chunk_index,
        "chunk_started_ms": chunk_started_ms,
        "chunk_duration_ms": chunk_duration_ms,
        "bytes": bytes_written,
        "uploaded_at": now,
    }
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

    # Trigger partial transcription every N chunks for stream sessions
    total_chunks = len(chunks)
    if (
        session_data.get("mode", "stream") == "stream"
        and PARTIAL_EVERY_N_CHUNKS > 0
        and total_chunks % PARTIAL_EVERY_N_CHUNKS == 0
        and session_data.get("state") in ("receiving", "chunks_complete", "partially_processed")
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
    Finalize session: merge chunks, enqueue transcription job.

    Body (JSON):
        run_diarization  bool (default false)
        language         ISO code or "auto" (default "auto")
        model_size       "tiny"|"base"|"small"|"medium"|"large" (default "base")
        speaker_count    int (also accepted as num_speakers for Android compat)
        merge_strategy   "timeline" (reserved)
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
        # Idempotent: return existing job info
        return {
            "session_id": session_id,
            "job_id": session_id,
            "status_url": f"/api/v2/jobs/{session_id}",
            "message": "Already finalized.",
        }
    if state == "cancelled":
        raise HTTPException(409, "Session was cancelled.")

    chunks = session_data.get("chunks", [])
    if not chunks:
        raise HTTPException(400, "No chunks uploaded. Upload at least one chunk before finalizing.")

    # Gather chunk files in order
    chunk_paths: List[Path] = []
    for ci in sorted(chunks, key=lambda c: c["chunk_index"]):
        p = d / "chunks" / f"chunk_{ci['chunk_index']:04d}.wav"
        if p.exists():
            chunk_paths.append(p)

    if not chunk_paths:
        raise HTTPException(400, "Chunk files missing on disk. Re-upload chunks.")

    # Merge into audio.wav
    try:
        _merge_wav_chunks(d, chunk_paths)
    except Exception as e:
        raise HTTPException(500, f"Failed to merge audio chunks: {e}")

    # Build transcription options
    # Accept num_speakers as alias for speaker_count (Android client field drift compat)
    language_raw: str = body.get("language", "auto")
    run_diarization: bool = bool(body.get("run_diarization", False))
    model_size: str = body.get("model_size", "base")
    speaker_count: int = int(
        body.get("speaker_count")
        or body.get("num_speakers")
        or (2 if run_diarization else 1)
    )

    # "auto" → pass None to faster-whisper for language auto-detection
    language_for_worker = None if language_raw == "auto" else language_raw

    now = _utcnow()

    # Write metadata.json (consumed by worker)
    metadata = {
        "session_id": session_id,
        "language": language_for_worker,
        "speaker_count": speaker_count,
        "model_size": model_size,
        "timestamps": True,
        "diarization_enabled": run_diarization,
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

    # Update v2 session state
    session_data["state"] = "finalized"
    session_data["finalize_requested_at"] = now
    if not session_data.get("finalized_at"):
        session_data["finalized_at"] = now
    _write_json(session_path, session_data)

    # Enqueue to Redis — language_for_worker is None for auto-detect (never coerce to "en")
    job_data = {
        "session_id": session_id,
        "language": language_for_worker,
        "speaker_count": speaker_count,
        "model_size": model_size,
        "timestamps": True,
        "job_type": "v2",
        "queued_at": now,
    }
    r = _get_redis()
    if r:
        try:
            r.rpush(REDIS_QUEUE, json.dumps(job_data))
            print(f"[V2] Enqueued job for session {session_id}")
        except Exception as e:
            print(f"[V2] Warning: failed to enqueue job: {e}")
    else:
        print(f"[V2] Warning: Redis unavailable, job {session_id} not enqueued")

    return {
        "session_id": session_id,
        "job_id": session_id,
        "status_url": f"/api/v2/jobs/{session_id}",
    }


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str, _token: str = Depends(require_auth)):
    """
    Poll transcription job status.

    States: queued | running | done | error
    progress: dict with keys {upload: 0-100, processing: 0-100, stage: str}
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

    # Extra metadata from metadata.json and result files
    meta: dict = {}
    metadata_path = session_dir / "metadata.json"
    if metadata_path.exists():
        try:
            md = _read_json(metadata_path)
            meta["language_requested"] = md.get("language")  # None = auto
            meta["model_size"] = md.get("model_size")
            meta["speaker_count"] = md.get("speaker_count")
        except Exception:
            pass

    if state == "done":
        tsp = session_dir / "transcript_timestamps.json"
        if tsp.exists():
            try:
                segs = json.loads(tsp.read_text(encoding="utf-8"))
                meta["segment_count"] = len(segs)
                if segs:
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

    response = {
        "job_id": job_id,
        "state": state,
        "progress": progress_data,
        "message": message,
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
    Rich session status combining v2_session.json + status.json.

    Covers the full state machine including pre-finalization states
    (created / receiving / partially_processed) that /jobs/{id} doesn't expose.

    Returns:
        session_id, state, chunk_count, total_audio_ms,
        partial_transcript_available, timing fields, worker progress.
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
    partial_available = (d / "partial_transcript.json").exists()

    chunks = session_data.get("chunks", [])
    total_duration_ms = sum(c.get("chunk_duration_ms", 0) for c in chunks)

    return {
        "session_id": session_id,
        "state": state,
        "chunk_count": len(chunks),
        "total_audio_ms": total_duration_ms,
        "partial_transcript_available": partial_available,
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
    (state: partially_processed or later). Marked provisional=true.

    Returns 404 if no partial transcript is available yet.
    The final authoritative transcript is at /sessions/{id}/transcript.
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
    }


@router.get("/sessions/{session_id}/transcript")
async def get_transcript(session_id: str, _token: str = Depends(require_auth)):
    """
    Get the structured transcript result for a finished session.

    Returns 202 with state info if processing is not yet complete.
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

    # Plain text
    transcript_text = ""
    tp = d / "transcript.txt"
    if tp.exists():
        transcript_text = tp.read_text(encoding="utf-8")

    # Segments with speaker + ms timestamps
    segments = []
    sp = d / "transcript_by_speaker.json"
    if sp.exists():
        raw = json.loads(sp.read_text(encoding="utf-8"))
        segments = [
            {
                "start_ms": int(seg.get("start", 0) * 1000),
                "end_ms": int(seg.get("end", 0) * 1000),
                "speaker": seg.get("speaker", "SPEAKER_00"),
                "text": seg.get("text", "").strip(),
            }
            for seg in raw
        ]

    # Word-level (if transcript_timestamps.json has word-level data)
    words: List[dict] = []
    tsp = d / "transcript_timestamps.json"
    if tsp.exists():
        ts_data = json.loads(tsp.read_text(encoding="utf-8"))
        for seg in ts_data:
            for w in seg.get("words", []):
                words.append(
                    {
                        "t_ms": int(w.get("start", seg.get("start", 0)) * 1000),
                        "speaker": "SPEAKER_00",
                        "w": w.get("word", "").strip(),
                    }
                )

    return {
        "session_id": session_id,
        "text": transcript_text,
        "segments": segments,
        "words": words,
        "formats": {
            "srt_url": f"/api/v2/sessions/{session_id}/subtitle.srt",
            "vtt_url": f"/api/v2/sessions/{session_id}/subtitle.vtt",
        },
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


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, _token: str = Depends(require_auth)):
    """Delete session data (audio, chunks, transcripts). GDPR-ish."""
    d = _require_v2_session(session_id)
    shutil.rmtree(str(d))
    return {"deleted": True, "session_id": session_id}
