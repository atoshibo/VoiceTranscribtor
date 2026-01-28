from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, PlainTextResponse
from starlette.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
import os
import json
import uuid
import shutil
from pathlib import Path
from typing import Optional, List, Dict
import asyncio
from datetime import datetime

# Optional import - diarization only needed for worker, not web service
try:
    from diarization import group_by_speaker
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False

    def group_by_speaker(segments):
        """Fallback when diarization module not available."""
        return {}


app = FastAPI(title="Voice Transcript Server")

# Authentication configuration
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
AUTH_QUERY_PARAM = os.getenv("AUTH_QUERY_PARAM", "token")


def is_authorized(request: StarletteRequest) -> bool:
    """Check if request is authorized via query param or Authorization header."""
    if not AUTH_TOKEN:
        return False
    
    # Check query parameter
    query_token = request.query_params.get(AUTH_QUERY_PARAM)
    if query_token == AUTH_TOKEN:
        return True
    
    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]  # Remove "Bearer " prefix
        if token == AUTH_TOKEN:
            return True
    
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce static token authentication."""

    async def dispatch(self, request: StarletteRequest, call_next):
        # Allow health endpoints without authentication
        if request.url.path in ["/health", "/api/health"]:
            return await call_next(request)

        # If AUTH_TOKEN is not set, refuse all requests
        if not AUTH_TOKEN:
            return PlainTextResponse(
                "500 Internal Server Error: AUTH_TOKEN not set. Server cannot run without authentication.",
                status_code=500
            )

        # Check authorization
        if not is_authorized(request):
            return PlainTextResponse(
                f"403 Forbidden. Authentication required. Add ?{AUTH_QUERY_PARAM}=YOUR_TOKEN to the URL or send Authorization: Bearer YOUR_TOKEN header.",
                status_code=403
            )

        # Authorized, proceed
        return await call_next(request)


app.add_middleware(AuthMiddleware)

# Redis connection for job queue
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_QUEUE = os.getenv("REDIS_QUEUE", "transcription_jobs")

_redis_client = None

def get_redis():
    """Get or create Redis client."""
    global _redis_client
    if _redis_client is None:
        try:
            import redis
            _redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            _redis_client.ping()  # Test connection
            print(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            print(f"Warning: Could not connect to Redis: {e}")
            print("Worker queue disabled - jobs will not be processed")
    return _redis_client

@app.on_event("startup")
async def startup_event():
    print("=" * 50)
    print("Starting Voice Transcript Web Server")
    print("=" * 50)
    # Test Redis connection
    get_redis()
    print("=" * 50)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

DATA_DIR = Path("/data")
SESSIONS_DIR = DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/api/selftest")
async def selftest():
    """Quick selftest - verify GPU/device configuration without full transcription."""
    import subprocess
    from transcription import get_device_diagnostics, select_device

    device, device_index, compute_type = select_device()

    result = {
        "device_selected": device,
        "device_index": device_index,
        "compute_type": compute_type,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "force_device": os.getenv("FORCE_DEVICE"),
        "nvidia_smi_present": False,
        "nvidia_smi_output": ""
    }

    # Quick nvidia-smi check
    try:
        smi_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            timeout=2,
            text=True
        )
        if smi_result.returncode == 0:
            result["nvidia_smi_present"] = True
            result["nvidia_smi_output"] = smi_result.stdout.strip()
    except Exception as e:
        result["nvidia_smi_error"] = str(e)

    return result


@app.get("/api/diagnostics")
async def diagnostics():
    """Full diagnostics: device info, GPU info, env, queue stats, errors."""
    from transcription import get_device_diagnostics, run_smoke_test, select_device

    device, device_index, compute_type = select_device()

    # Get Redis stats
    redis_stats = {"connected": False, "queue_length": 0}
    r = get_redis()
    if r:
        try:
            redis_stats["connected"] = True
            redis_stats["queue_length"] = r.llen(REDIS_QUEUE)
        except Exception as e:
            redis_stats["error"] = str(e)

    # Get session stats
    session_stats = {
        "total": 0,
        "uploaded": 0,
        "processing": 0,
        "running": 0,
        "done": 0,
        "error": 0,
        "cancelled": 0
    }
    if SESSIONS_DIR.exists():
        for session_dir in SESSIONS_DIR.iterdir():
            if session_dir.is_dir():
                status_path = session_dir / "status.json"
                if status_path.exists():
                    session_stats["total"] += 1
                    try:
                        with open(status_path, "r") as f:
                            status_data = json.load(f)
                            status = status_data.get("status", "unknown")
                            session_stats[status] = session_stats.get(status, 0) + 1
                    except:
                        pass

    # Run quick smoke test (tiny model)
    smoke_result = run_smoke_test("tiny")

    return {
        "device_info": {
            "selected_device": device,
            "device_index": device_index,
            "compute_type": compute_type
        },
        "env_vars": {
            "FORCE_DEVICE": os.getenv("FORCE_DEVICE"),
            "CUDA_DEVICE_INDEX": os.getenv("CUDA_DEVICE_INDEX"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            "COMPUTE_TYPE": os.getenv("COMPUTE_TYPE"),
            "MAX_UPLOAD_MB": os.getenv("MAX_UPLOAD_MB", "200"),
            "MAX_QUEUE": os.getenv("MAX_QUEUE", "20"),
            "MAX_WORKERS": os.getenv("MAX_WORKERS", "1")
        },
        "gpu_diagnostics": get_device_diagnostics(),
        "smoke_test": smoke_result,
        "redis_stats": redis_stats,
        "session_stats": session_stats
    }


@app.get("/api/jobs")
async def list_jobs():
    """List all jobs (sessions) with their current status."""
    jobs = []
    if SESSIONS_DIR.exists():
        for session_dir in SESSIONS_DIR.iterdir():
            if session_dir.is_dir():
                status_path = session_dir / "status.json"
                metadata_path = session_dir / "metadata.json"
                if status_path.exists():
                    with open(status_path, "r") as f:
                        status_data = json.load(f)

                    metadata = {}
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)

                    jobs.append({
                        "job_id": session_dir.name,
                        "session_id": session_dir.name,
                        "status": status_data.get("status", "unknown"),
                        "progress": status_data.get("progress", {}),
                        "created_at": metadata.get("created_at", status_data.get("created_at", "")),
                        "error": status_data.get("error")
                    })

    # Sort by created_at descending
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"jobs": jobs}


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job."""
    session_dir = SESSIONS_DIR / job_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    status_path = session_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Job status not found")

    with open(status_path, "r") as f:
        status_data = json.load(f)

    metadata = {}
    metadata_path = session_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    return {
        "job_id": job_id,
        "session_id": job_id,
        "status": status_data.get("status", "unknown"),
        "progress": status_data.get("progress", {}),
        "error": status_data.get("error"),
        "metadata": metadata
    }


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a job (best effort - only works if not yet started)."""
    return await cancel_transcription(job_id)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    context = {
        "request": request,
        "auth_token": AUTH_TOKEN,
        "auth_query_param": AUTH_QUERY_PARAM,
    }
    return templates.TemplateResponse("index.html", context)


@app.get("/sessions", response_class=HTMLResponse)
async def sessions_list(request: Request):
    sessions = []
    if SESSIONS_DIR.exists():
        for session_dir in SESSIONS_DIR.iterdir():
            if session_dir.is_dir():
                status_path = session_dir / "status.json"
                metadata_path = session_dir / "metadata.json"
                if status_path.exists():
                    with open(status_path, "r") as f:
                        status_data = json.load(f)
                    metadata = {}
                    if metadata_path.exists():
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                    
                    analytics_path = session_dir / "analytics.json"
                    duration = 0
                    if analytics_path.exists():
                        with open(analytics_path, "r") as f:
                            analytics = json.load(f)
                            duration = analytics.get("duration", 0)
                    
                    sessions.append({
                        "session_id": session_dir.name,
                        "status": status_data.get("status", "unknown"),
                        "created_at": metadata.get("created_at", status_data.get("created_at", "")),
                        "duration": duration
                    })
    
    sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    context = {
        "request": request,
        "sessions": sessions,
        "auth_token": AUTH_TOKEN,
        "auth_query_param": AUTH_QUERY_PARAM,
    }
    return templates.TemplateResponse("sessions.html", context)


@app.get("/session/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, session_id: str):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Load status
    status_path = session_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Status not found")
    
    with open(status_path, "r") as f:
        status_data = json.load(f)
    
    # Load metadata
    metadata = {}
    metadata_path = session_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    # Load results if done
    transcript = ""
    timestamps = []
    summary = {}
    analytics = {}
    transcript_by_speaker = []
    speaker_groups = {}
    
    if status_data.get("status") == "done":
        transcript_path = session_dir / "transcript.txt"
        if transcript_path.exists():
            transcript = transcript_path.read_text(encoding="utf-8")
        
        timestamps_path = session_dir / "transcript_timestamps.json"
        if timestamps_path.exists():
            with open(timestamps_path, "r", encoding="utf-8") as f:
                timestamps = json.load(f)
        
        # Load speaker-attributed transcript
        transcript_by_speaker_path = session_dir / "transcript_by_speaker.json"
        if transcript_by_speaker_path.exists():
            with open(transcript_by_speaker_path, "r", encoding="utf-8") as f:
                transcript_by_speaker = json.load(f)
                speaker_groups = group_by_speaker(transcript_by_speaker)
        
        summary_path = session_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        
        analytics_path = session_dir / "analytics.json"
        if analytics_path.exists():
            with open(analytics_path, "r", encoding="utf-8") as f:
                analytics = json.load(f)
    
    context = {
        "request": request,
        "session_id": session_id,
        "status": status_data.get("status"),
        "metadata": metadata,
        "transcript": transcript,
        "timestamps": timestamps,
        "transcript_by_speaker": transcript_by_speaker,
        "speaker_groups": speaker_groups,
        "summary": summary,
        "analytics": analytics,
        "auth_token": AUTH_TOKEN,
        "auth_query_param": AUTH_QUERY_PARAM,
    }
    return templates.TemplateResponse("session_detail.html", context)


MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "200"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "20"))

@app.post("/api/upload")
async def upload(
    audio: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
    language: Optional[str] = Form("en"),
    speaker_count: Optional[int] = Form(1),
    model_size: Optional[str] = Form("base"),
    timestamps: Optional[bool] = Form(True)
):
    try:
        # Check queue size limit
        r = get_redis()
        if r:
            try:
                queue_len = r.llen(REDIS_QUEUE)
                if queue_len >= MAX_QUEUE:
                    raise HTTPException(status_code=429, detail=f"Queue full ({queue_len}/{MAX_QUEUE}). Please wait.")
            except Exception as e:
                print(f"Warning: Could not check queue length: {e}")

        # Handle both API (JSON metadata) and web UI (form fields)
        if metadata:
            metadata_dict = json.loads(metadata)
            session_id = metadata_dict.get("session_id") or str(uuid.uuid4())
            language = metadata_dict.get("language", language)
            speaker_count = metadata_dict.get("speaker_count", speaker_count)
            model_size = metadata_dict.get("model_size", model_size)
            timestamps = metadata_dict.get("timestamps", timestamps)
        else:
            # Web UI form submission
            session_id = str(uuid.uuid4())
            metadata_dict = {
                "session_id": session_id,
                "language": language,
                "speaker_count": speaker_count,
                "model_size": model_size,
                "timestamps": timestamps,
                "created_at": datetime.now().isoformat()
            }

        session_dir = SESSIONS_DIR / session_id

        # IDEMPOTENCY CHECK: If session already exists and is processed, return existing
        if session_dir.exists():
            status_path = session_dir / "status.json"
            if status_path.exists():
                with open(status_path, "r") as f:
                    existing = json.load(f)
                existing_status = existing.get("status")
                if existing_status in ["uploaded", "processing", "running", "done"]:
                    return {
                        "session_id": session_id,
                        "status": "already_exists",
                        "existing_status": existing_status
                    }

        session_dir.mkdir(exist_ok=True)

        # Store original filename
        original_filename = audio.filename or "audio.wav"
        metadata_dict["original_filename"] = original_filename

        # Save audio file with streaming (avoid buffering entire file in RAM)
        audio_path = session_dir / "audio.wav"
        bytes_written = 0
        max_bytes = MAX_UPLOAD_MB * 1024 * 1024

        with open(audio_path, "wb") as f:
            while chunk := await audio.read(8192):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    # Clean up partial file
                    f.close()
                    audio_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_MB}MB)")
                f.write(chunk)

        print(f"[{session_id}] Upload complete: {bytes_written} bytes")

        # Save metadata
        metadata_path = session_dir / "metadata.json"
        if "uploaded_at" not in metadata_dict:
            metadata_dict["uploaded_at"] = datetime.now().isoformat()
        if "created_at" not in metadata_dict:
            metadata_dict["created_at"] = datetime.now().isoformat()
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        # Initialize session status with progress
        status_path = session_dir / "status.json"
        with open(status_path, "w") as f:
            json.dump({
                "status": "uploaded",
                "session_id": session_id,
                "progress": {
                    "upload": 100,
                    "processing": 0,
                    "stage": "uploaded"
                }
            }, f)

        # Automatically enqueue transcription job to Redis
        if r:
            job_data = {
                "session_id": session_id,
                "language": language,
                "speaker_count": speaker_count,
                "model_size": model_size,
                "timestamps": timestamps
            }
            try:
                r.rpush(REDIS_QUEUE, json.dumps(job_data))
                print(f"[{session_id}] Job enqueued to Redis")
            except Exception as e:
                print(f"[{session_id}] Failed to enqueue job: {e}")
        else:
            print(f"[{session_id}] Warning: Redis not available, job not enqueued")

        return {"session_id": session_id, "status": "uploaded", "progress": {"upload": 100, "processing": 0}}

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Store active transcription tasks for cancellation
_active_tasks = {}

@app.post("/api/transcribe/{session_id}")
async def transcribe(session_id: str):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    audio_path = session_dir / "audio.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    metadata_path = session_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    language = metadata.get("language", "en")
    diarization_enabled = metadata.get("diarization_enabled", False)
    speaker_count = metadata.get("speaker_count", 1)
    model_size = metadata.get("model_size", "base")
    timestamps_enabled = metadata.get("timestamps", True)
    
    # Update status to processing with progress
    status_path = session_dir / "status.json"
    with open(status_path, "w") as f:
        json.dump({
            "status": "processing",
            "session_id": session_id,
            "progress": {
                "upload": 100,
                "processing": 0,
                "stage": "starting"
            }
        }, f)
    
    print(f"[{session_id}] Starting transcription: Upload 100%, Processing 0%")
    
    # Run transcription in background and store task for cancellation
    task = asyncio.create_task(process_transcription(
        session_id, audio_path, language, diarization_enabled, speaker_count, model_size, timestamps_enabled
    ))
    _active_tasks[session_id] = task
    
    # Clean up task when done
    task.add_done_callback(lambda t: _active_tasks.pop(session_id, None))
    
    return {"session_id": session_id, "status": "processing", "progress": {"upload": 100, "processing": 0}}


def update_progress(session_id: str, stage: str, processing_percent: int):
    """Update progress in status.json and log it"""
    session_dir = SESSIONS_DIR / session_id
    status_path = session_dir / "status.json"
    
    try:
        with open(status_path, "r") as f:
            status_data = json.load(f)
        
        status_data["progress"] = {
            "upload": 100,
            "processing": processing_percent,
            "stage": stage
        }
        
        with open(status_path, "w") as f:
            json.dump(status_data, f)
        
        print(f"[{session_id}] Progress: Upload 100%, Processing {processing_percent}% - {stage}")
    except Exception as e:
        print(f"[{session_id}] Error updating progress: {e}")


# NOTE: process_transcription moved to worker.py
# This function is kept for backward compatibility but should not be called
async def process_transcription(
    session_id: str,
    audio_path: Path,
    language: str,
    diarization_enabled: bool,
    speaker_count: int,
    model_size: str = "base",
    timestamps_enabled: bool = True
):
    """DEPRECATED: Transcription is now handled by worker service."""
    raise HTTPException(status_code=501, detail="Transcription moved to worker service")
    try:
        session_dir = SESSIONS_DIR / session_id
        
        # Load metadata for original filename
        metadata_path = session_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
        
        update_progress(session_id, "loading_model", 10)
        
        # Optional audio preprocessing for diarization
        processed_audio_path = audio_path
        if diarization_enabled and speaker_count > 1:
            processed_audio_path = session_dir / "audio_processed.wav"
            update_progress(session_id, "preprocessing", 5)
            if not preprocess_audio(str(audio_path), str(processed_audio_path)):
                # If preprocessing fails, use original
                processed_audio_path = audio_path
        
        # Perform diarization if enabled
        diarization_segments = []
        if diarization_enabled and speaker_count > 1:
            update_progress(session_id, "diarizing", 8)
            try:
                diarization_segments = perform_diarization(
                    str(processed_audio_path),
                    num_speakers=speaker_count,
                    min_speakers=1,
                    max_speakers=speaker_count
                )
                # Save diarization results
                diarization_path = session_dir / "diarization.json"
                with open(diarization_path, "w", encoding="utf-8") as f:
                    json.dump(diarization_segments, f, indent=2)
                print(f"[{session_id}] Diarization found {len(diarization_segments)} segments")
            except Exception as e:
                print(f"[{session_id}] Diarization failed: {e}, continuing without diarization")
                diarization_segments = []
        
        # Check cancellation before transcription
        if session_id in _active_tasks:
            task = _active_tasks.get(session_id)
            if task and task.cancelled():
                print(f"[{session_id}] Task cancelled before transcription")
                return
        
        # Transcribe
        transcript, timestamps = transcribe_audio(
            str(audio_path),
            language=language,
            diarization_enabled=diarization_enabled,
            speaker_count=speaker_count,
            model_size=model_size,
            timestamps_enabled=timestamps_enabled,
            progress_callback=lambda p: update_progress(session_id, "transcribing", 10 + int(p * 0.6))
        )
        
        update_progress(session_id, "aligning", 75)
        
        # Align diarization with transcript
        aligned_segments = []
        if diarization_segments and timestamps:
            aligned_segments = align_diarization_with_transcript(diarization_segments, timestamps)
        elif timestamps:
            # No diarization, use default speaker
            aligned_segments = [
                {**seg, "speaker": "SPEAKER_00"}
                for seg in timestamps
            ]
        
        # Group by speaker
        speaker_groups = group_by_speaker(aligned_segments) if aligned_segments else {}
        
        update_progress(session_id, "saving_results", 80)
        
        # Save transcript (original format)
        transcript_path = session_dir / "transcript.txt"
        transcript_path.write_text(transcript, encoding="utf-8")
        
        # Save timestamps
        if timestamps_enabled:
            timestamps_path = session_dir / "transcript_timestamps.json"
            with open(timestamps_path, "w", encoding="utf-8") as f:
                json.dump(timestamps, f, indent=2, ensure_ascii=False)
        
        # Save speaker-attributed transcript (canonical JSON)
        if aligned_segments:
            transcript_by_speaker_json = session_dir / "transcript_by_speaker.json"
            with open(transcript_by_speaker_json, "w", encoding="utf-8") as f:
                json.dump(aligned_segments, f, indent=2, ensure_ascii=False)
            
            # Save human-readable speaker transcript
            transcript_by_speaker_txt = session_dir / "transcript_by_speaker.txt"
            with open(transcript_by_speaker_txt, "w", encoding="utf-8") as f:
                for seg in aligned_segments:
                    speaker_label = seg.get("speaker", "SPEAKER_00")
                    start_time = seg.get("start", 0)
                    end_time = seg.get("end", 0)
                    text = seg.get("text", "")
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s] {speaker_label}: {text}\n")
        
        # Generate subtitle files (with speaker labels if available)
        if timestamps_enabled and timestamps:
            if aligned_segments:
                generate_subtitles_with_speakers(session_dir, aligned_segments)
            else:
                generate_subtitles(session_dir, timestamps)
        
        # Generate summary
        summary = generate_summary(transcript)
        summary_path = session_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Generate analytics
        analytics = generate_analytics(transcript, timestamps)
        analytics_path = session_dir / "analytics.json"
        with open(analytics_path, "w", encoding="utf-8") as f:
            json.dump(analytics, f, indent=2, ensure_ascii=False)
        
        update_progress(session_id, "completed", 100)
        
        # Update status to done
        status_path = session_dir / "status.json"
        with open(status_path, "r") as f:
            status_data = json.load(f)
        
        status_data["status"] = "done"
        status_data["completed_at"] = datetime.now().isoformat()
        status_data["progress"] = {
            "upload": 100,
            "processing": 100,
            "stage": "completed"
        }
        
        with open(status_path, "w") as f:
            json.dump(status_data, f)
        
        print(f"[{session_id}] Transcription complete: Upload 100%, Processing 100%")
    
    except asyncio.CancelledError:
        # Task was cancelled
        session_dir = SESSIONS_DIR / session_id
        status_path = session_dir / "status.json"
        with open(status_path, "w") as f:
            json.dump({
                "status": "cancelled",
                "session_id": session_id,
                "cancelled_at": datetime.now().isoformat(),
                "progress": {
                    "upload": 100,
                    "processing": 0,
                    "stage": "cancelled"
                }
            }, f)
        print(f"[{session_id}] Transcription cancelled")
        _active_tasks.pop(session_id, None)
    except Exception as e:
        # Update status to error
        session_dir = SESSIONS_DIR / session_id
        status_path = session_dir / "status.json"
        with open(status_path, "w") as f:
            json.dump({
                "status": "error",
                "session_id": session_id,
                "error": str(e),
                "progress": {
                    "upload": 100,
                    "processing": 0,
                    "stage": "error"
                }
            }, f)
        print(f"[{session_id}] Transcription error: {e}")
        _active_tasks.pop(session_id, None)


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    status_path = session_dir / "status.json"
    if not status_path.exists():
        raise HTTPException(status_code=404, detail="Status not found")
    
    with open(status_path, "r") as f:
        status_data = json.load(f)
    
    status = status_data.get("status", "unknown")
    progress = status_data.get("progress", {"upload": 0, "processing": 0, "stage": "unknown"})
    
    if status != "done":
        return {
            "session_id": session_id,
            "status": status,
            "progress": progress
        }
    
    # Load all results
    transcript_path = session_dir / "transcript.txt"
    timestamps_path = session_dir / "transcript_timestamps.json"
    summary_path = session_dir / "summary.json"
    analytics_path = session_dir / "analytics.json"
    
    result = {
        "session_id": session_id,
        "status": "done",
        "progress": progress
    }
    
    if transcript_path.exists():
        result["transcript"] = transcript_path.read_text(encoding="utf-8")
    
    if timestamps_path.exists():
        with open(timestamps_path, "r", encoding="utf-8") as f:
            result["timestamps"] = json.load(f)
    
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            result["summary"] = json.load(f)
    
    if analytics_path.exists():
        with open(analytics_path, "r", encoding="utf-8") as f:
            result["analytics"] = json.load(f)
    
    return result


@app.get("/api/sessions")
async def list_sessions():
    sessions = []
    if SESSIONS_DIR.exists():
        for session_dir in SESSIONS_DIR.iterdir():
            if session_dir.is_dir():
                status_path = session_dir / "status.json"
                if status_path.exists():
                    with open(status_path, "r") as f:
                        status_data = json.load(f)
                    sessions.append({
                        "session_id": session_dir.name,
                        "status": status_data.get("status", "unknown"),
                        "created_at": status_data.get("created_at", "")
                    })
    return {"sessions": sessions}


@app.post("/api/cancel/{session_id}")
async def cancel_transcription(session_id: str):
    """Cancel a queued transcription job (if not yet started)."""
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    
    status_path = session_dir / "status.json"
    if status_path.exists():
        with open(status_path, "r") as f:
            status_data = json.load(f)
        
        current_status = status_data.get("status")
        if current_status == "running":
            return {"session_id": session_id, "status": "cannot_cancel", "message": "Job already running, cannot cancel"}
        elif current_status == "done":
            return {"session_id": session_id, "status": "already_complete", "message": "Task already completed"}
        elif current_status in ["pending", "uploaded"]:
            # Update status to cancelled
            status_data["status"] = "cancelled"
            status_data["cancelled_at"] = datetime.now().isoformat()
            status_data["progress"] = {
                "upload": 100,
                "processing": 0,
                "stage": "cancelled"
            }
            with open(status_path, "w") as f:
                json.dump(status_data, f)
            
            # Note: Job may still be in Redis queue, but worker will skip if status is cancelled
            return {"session_id": session_id, "status": "cancelled", "message": "Transcription cancelled"}
    
    return {"session_id": session_id, "status": "not_found", "message": "Session not found"}


@app.get("/api/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    # Security: prevent path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    # Load metadata for original filename
    metadata_path = session_dir / "metadata.json"
    original_filename = "audio.wav"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            original_filename = metadata.get("original_filename", "audio.wav")

    # Extract base name without extension and sanitize for ASCII
    base_name = Path(original_filename).stem

    # Sanitize base_name to ASCII-safe characters (replace non-ASCII with underscore)
    try:
        base_name.encode('ascii')
        safe_base_name = base_name
    except UnicodeEncodeError:
        # Use session_id as safe fallback for non-ASCII filenames
        safe_base_name = session_id[:8]

    # Map internal filenames to user-friendly names with sanitized base name
    file_mapping = {
        "audio.wav": f"{safe_base_name}.wav",
        "transcript.txt": f"TRANSC_{safe_base_name}.txt",
        "transcript_by_speaker.txt": f"TRANSC_{safe_base_name}.txt",
        "transcript_by_speaker.json": f"TRANSC_{safe_base_name}.json",
        "transcript_timestamps.json": f"TRANSC_{safe_base_name}_timestamps.json",
        "subtitles.srt": f"SUB_{safe_base_name}.srt",
        "subtitles.vtt": f"SUB_{safe_base_name}.vtt",
        "summary.json": f"{safe_base_name}_summary.json",
        "analytics.json": f"{safe_base_name}_analytics.json"
    }

    # Check if requesting a mapped file
    download_filename = file_mapping.get(filename, filename)
    file_path = session_dir / filename

    # Allow all files in session directory (with security check)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    ext = Path(download_filename).suffix.lower()
    media_types = {
        ".wav": "audio/wav",
        ".txt": "text/plain",
        ".json": "application/json",
        ".srt": "text/srt",
        ".vtt": "text/vtt"
    }
    media_type = media_types.get(ext, "application/octet-stream")

    # Use safe ASCII filename for Content-Disposition header
    return FileResponse(
        path=str(file_path),
        filename=download_filename,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{download_filename}"'}
    )


def generate_subtitles(session_dir: Path, timestamps: List[Dict]):
    """Generate SRT and VTT subtitle files from timestamps."""
    # SRT format
    srt_lines = []
    for i, item in enumerate(timestamps, 1):
        start = format_timestamp_srt(item["start"])
        end = format_timestamp_srt(item["end"])
        text = item["text"].strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    
    srt_path = session_dir / "subtitles.srt"
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    
    # VTT format
    vtt_lines = ["WEBVTT", ""]
    for item in timestamps:
        start = format_timestamp_vtt(item["start"])
        end = format_timestamp_vtt(item["end"])
        text = item["text"].strip()
        vtt_lines.append(f"{start} --> {end}\n{text}")
    
    vtt_path = session_dir / "subtitles.vtt"
    vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")


def generate_subtitles_with_speakers(session_dir: Path, aligned_segments: List[Dict]):
    """Generate SRT and VTT subtitle files with speaker labels."""
    # SRT format
    srt_lines = []
    for i, item in enumerate(aligned_segments, 1):
        start = format_timestamp_srt(item["start"])
        end = format_timestamp_srt(item["end"])
        speaker = item.get("speaker", "SPEAKER_00")
        text = item["text"].strip()
        # Include speaker in subtitle text
        srt_lines.append(f"{i}\n{start} --> {end}\n[{speaker}] {text}\n")
    
    srt_path = session_dir / "subtitles.srt"
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    
    # VTT format
    vtt_lines = ["WEBVTT", ""]
    for item in aligned_segments:
        start = format_timestamp_vtt(item["start"])
        end = format_timestamp_vtt(item["end"])
        speaker = item.get("speaker", "SPEAKER_00")
        text = item["text"].strip()
        vtt_lines.append(f"{start} --> {end}\n[{speaker}] {text}")
    
    vtt_path = session_dir / "subtitles.vtt"
    vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")


def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format timestamp for VTT: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

