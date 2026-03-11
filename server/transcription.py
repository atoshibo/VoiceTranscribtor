"""
Whisper transcription wrapper.

Device selection is env-driven and deterministic:
  WHISPER_DEVICE         cuda | cpu  (default: cuda)
  WHISPER_STRICT_CUDA    1 | 0       (default: 1 — fail if GPU unavailable)
  CUDA_DEVICE_INDEX      int         (default: 0)
  WHISPER_COMPUTE_TYPE_CUDA          (default: float16)
  WHISPER_MODEL          model size  (default: small)
"""
import os
import subprocess
import math
import re
from typing import List, Dict, Tuple, Optional, Callable, Any

from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Configuration (read once at import time)
# ---------------------------------------------------------------------------
DEVICE: str = os.getenv("WHISPER_DEVICE", "cuda").lower().strip()
DEVICE_INDEX: int = int(os.getenv("CUDA_DEVICE_INDEX", "0"))
COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE_CUDA", "float16").lower().strip()
STRICT_CUDA: bool = os.getenv("WHISPER_STRICT_CUDA", "1").strip() == "1"
DEFAULT_MODEL: str = os.getenv("WHISPER_MODEL", "small")

# Enforce strict: cuda is the only allowed device
if STRICT_CUDA and DEVICE != "cuda":
    print(f"[WHISPER-CONFIG] WHISPER_STRICT_CUDA=1 — forcing DEVICE=cuda (was {DEVICE!r})")
    DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Omi/chunked-audio transcription tuning
# These defaults reduce hallucination loops on sparse/BLE recordings.
# Override via environment variables if needed.
# ---------------------------------------------------------------------------
# condition_on_previous_text=True is the main culprit for repetition loops.
# Default off for Omi; set OMI_WHISPER_CONDITION_ON_PREVIOUS_TEXT=1 to re-enable.
_CONDITION_ON_PREV: bool = os.getenv("OMI_WHISPER_CONDITION_ON_PREVIOUS_TEXT", "0") == "1"
# beam_size=1 (greedy) eliminates beam-search induced repetition on sparse audio.
_BEAM_SIZE: int = int(os.getenv("OMI_WHISPER_BEAM_SIZE", "1"))
# best_of=1 disables sampling fallbacks that can loop.
_BEST_OF: int = int(os.getenv("OMI_WHISPER_BEST_OF", "1"))
# Max consecutive identical segments before suppressing — catches remaining loops.
_MAX_REPEAT_SEGS: int = int(os.getenv("OMI_WHISPER_MAX_REPEAT_SEGMENTS", "2"))

# ---------------------------------------------------------------------------
# Model cache: (model_size, device, compute_type) → WhisperModel
# ---------------------------------------------------------------------------
_model_cache: Dict[Tuple[str, str, str], WhisperModel] = {}

# Cached result of probe_cuda_health()
_cuda_health: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def get_model(
    model_size: str = None,
    device: str = None,
    compute_type: str = None,
) -> WhisperModel:
    """
    Return a cached WhisperModel. Loads on first call.

    Uses module-level DEVICE / DEVICE_INDEX / COMPUTE_TYPE unless overridden.
    Raises RuntimeError with a clear message on failure.
    """
    model_size = model_size or DEFAULT_MODEL
    device = (device or DEVICE).lower()
    compute_type = compute_type or (COMPUTE_TYPE if device == "cuda" else "int8")
    key = (model_size, device, compute_type)

    if key in _model_cache:
        return _model_cache[key]

    print(f"[WHISPER] Loading model={model_size} device={device}:{DEVICE_INDEX} compute_type={compute_type}")
    kwargs: Dict[str, Any] = {"device": device, "compute_type": compute_type}
    if device == "cuda":
        kwargs["device_index"] = DEVICE_INDEX

    try:
        m = WhisperModel(model_size, **kwargs)
        _model_cache[key] = m
        print(f"[WHISPER] Model loaded OK")
        return m
    except Exception as e:
        print(f"[WHISPER] Error loading model: {e}")
        raise RuntimeError(
            f"Failed to load Whisper model '{model_size}' on {device} (compute_type={compute_type}): {e}"
        ) from e


# ---------------------------------------------------------------------------
# CUDA health probe (used by worker startup and /api/v2/health)
# ---------------------------------------------------------------------------
def probe_cuda_health(model_size: str = "tiny") -> Dict[str, Any]:
    """
    Check CUDA availability by loading a tiny Whisper model on GPU.
    Result is cached after the first call.

    Returns:
        {gpu_available, gpu_reason, selected_compute_type, strict_cuda}
    """
    global _cuda_health
    if _cuda_health is not None:
        return _cuda_health

    result: Dict[str, Any] = {
        "gpu_available": False,
        "gpu_reason": None,
        "selected_device": "cuda",
        "selected_compute_type": COMPUTE_TYPE,
        "strict_cuda": STRICT_CUDA,
    }

    print(f"[WHISPER] Probing CUDA with model={model_size} compute_type={COMPUTE_TYPE}")
    try:
        _ = WhisperModel(
            model_size,
            device="cuda",
            compute_type=COMPUTE_TYPE,
            device_index=DEVICE_INDEX,
        )
        result["gpu_available"] = True
        print("[WHISPER] CUDA OK")
    except Exception as e:
        result["gpu_reason"] = str(e)
        print(f"[WHISPER] CUDA probe failed: {e}")

    _cuda_health = result
    return result


# ---------------------------------------------------------------------------
# Diagnostics (used by /api/diagnostics)
# ---------------------------------------------------------------------------
def get_device_diagnostics() -> Dict[str, Any]:
    """Return GPU diagnostic info: nvidia-smi output, ctranslate2, device config."""
    diag: Dict[str, Any] = {
        "device": DEVICE,
        "device_index": DEVICE_INDEX,
        "compute_type": COMPUTE_TYPE,
        "strict_cuda": STRICT_CUDA,
        "model_default": DEFAULT_MODEL,
        "nvidia_smi_available": False,
        "nvidia_smi_output": "",
        "detected_gpus": [],
        "ctranslate2_available": False,
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, timeout=3, text=True,
        )
        if result.returncode == 0:
            diag["nvidia_smi_available"] = True
            diag["nvidia_smi_output"] = result.stdout.strip()
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    diag["detected_gpus"].append({
                        "index": parts[0],
                        "name": parts[1],
                        "utilization": parts[2] if len(parts) > 2 else "",
                        "memory_used": parts[3] if len(parts) > 3 else "",
                        "memory_total": parts[4] if len(parts) > 4 else "",
                    })
    except Exception as e:
        diag["nvidia_smi_error"] = str(e)

    try:
        import ctranslate2  # noqa: F401
        diag["ctranslate2_available"] = True
        diag["ctranslate2_version"] = ctranslate2.__version__
    except Exception:
        pass

    # Include cached health result if available
    if _cuda_health is not None:
        diag["cuda_health"] = _cuda_health

    return diag


# ---------------------------------------------------------------------------
# Smoke test (used by /api/selftest)
# ---------------------------------------------------------------------------
def run_smoke_test(model_size: str = "tiny") -> Dict[str, Any]:
    """Load a small model to verify GPU setup end-to-end."""
    result: Dict[str, Any] = {
        "success": False,
        "device": DEVICE,
        "device_index": DEVICE_INDEX,
        "compute_type": COMPUTE_TYPE,
        "model_size": model_size,
        "error": None,
    }
    try:
        print(f"[SMOKE TEST] Loading model={model_size} device={DEVICE}:{DEVICE_INDEX} compute_type={COMPUTE_TYPE}")
        _ = WhisperModel(
            model_size,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            device_index=DEVICE_INDEX if DEVICE == "cuda" else 0,
        )
        result["success"] = True
        print(f"[SMOKE TEST] OK")
    except Exception as e:
        result["error"] = str(e)
        print(f"[SMOKE TEST] FAILED: {e}")
    return result


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------
def _smooth_progress(segment_count: int) -> int:
    """Exponential saturation: 0 segs → 10%, ~250 segs → ~78%."""
    return min(90, max(10, int(10 + 80 * (1.0 - math.exp(-segment_count / 140.0)))))


# ---------------------------------------------------------------------------
# Repetition guard
# ---------------------------------------------------------------------------
def _suppress_repetition_loops(segments: List[Dict], max_repeats: int = 2) -> List[Dict]:
    """
    Drop excess consecutive segments with identical text.
    Catches hallucination loops common on sparse/silent Omi audio.

    Example: ["hello", "hello", "hello", "hello"] → ["hello", "hello"]  (max_repeats=2)
    """
    if not segments or max_repeats < 1:
        return segments
    result: List[Dict] = []
    run_text: Optional[str] = None
    run_count = 0
    dropped = 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if text == run_text:
            run_count += 1
        else:
            run_text = text
            run_count = 1
        if run_count <= max_repeats:
            result.append(seg)
        else:
            dropped += 1
    if dropped:
        print(f"[WHISPER] Suppressed {dropped} repeated segment(s) (hallucination loop guard)")
    return result


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------
def transcribe_audio(
    audio_path: str,
    language: str = None,
    model_size: str = None,
    timestamps_enabled: bool = True,
    progress_callback: Optional[Callable[[int], None]] = None,
    # Legacy params (kept for call-site compatibility with old worker code)
    diarization_enabled: bool = False,
    speaker_count: int = 1,
) -> Tuple[str, List[Dict]]:
    """
    Transcribe audio. Returns (full_text, segments).

    Each segment: {"start": float, "end": float, "text": str}
    """
    model = get_model(model_size)

    kwargs: Dict[str, Any] = {
        "language": language or None,
        "word_timestamps": timestamps_enabled,
        "beam_size": _BEAM_SIZE,
        "best_of": _BEST_OF,
        "patience": 1.0,
        "temperature": 0.0,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": _CONDITION_ON_PREV,
        "vad_filter": True,
        "batch_size": 16,
    }
    print(
        f"[WHISPER] Settings: beam={_BEAM_SIZE} best_of={_BEST_OF} "
        f"condition_on_prev={_CONDITION_ON_PREV} max_repeat={_MAX_REPEAT_SEGS}"
    )

    print("[WHISPER] Starting transcription...")
    if progress_callback:
        progress_callback(5)

    try:
        segments_gen, info = model.transcribe(audio_path, **kwargs)
    except TypeError:
        # older faster-whisper doesn't support batch_size
        kwargs.pop("batch_size", None)
        segments_gen, info = model.transcribe(audio_path, **kwargs)

    parts: List[str] = []
    timestamps: List[Dict] = []
    n = 0

    for seg in segments_gen:
        n += 1
        text = (seg.text or "").strip()
        if text:
            parts.append(text)
            timestamps.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
        if progress_callback and n % 5 == 0:
            progress_callback(_smooth_progress(n))

    lang = getattr(info, "language", None)
    prob = getattr(info, "language_probability", 0.0)
    print(f"[WHISPER] Done: language={lang} prob={prob:.2f} segments={n}")

    if progress_callback:
        progress_callback(90)

    timestamps = _suppress_repetition_loops(timestamps, _MAX_REPEAT_SEGS)
    full_text = " ".join(s["text"] for s in timestamps)
    return full_text, timestamps


# ---------------------------------------------------------------------------
# Summary + analytics (post-processing, no GPU needed)
# ---------------------------------------------------------------------------
def generate_summary(transcript: str) -> Dict:
    sentences = [s.strip() for s in re.split(r"[.!?]+", transcript) if len(s.strip()) > 20]
    if not sentences:
        return {"bullets": ["No content to summarize."]}
    if len(sentences) >= 5:
        idx = [0, len(sentences) // 4, len(sentences) // 2, 3 * len(sentences) // 4, -1]
        bullets = [sentences[i] for i in idx if 0 <= i < len(sentences)]
    else:
        bullets = sentences[:5]
    return {"bullets": bullets[:5]}


def generate_analytics(transcript: str, timestamps: List[Dict]) -> Dict:
    if not timestamps:
        return {"duration": 0, "word_count": 0, "words_per_minute": 0, "long_pauses_count": 0}
    duration = timestamps[-1]["end"]
    word_count = len(transcript.split())
    wpm = round(word_count / duration * 60, 1) if duration > 0 else 0
    pauses = sum(
        1 for i in range(1, len(timestamps))
        if timestamps[i]["start"] - timestamps[i - 1]["end"] > 2.0
    )
    return {
        "duration": round(duration, 2),
        "word_count": word_count,
        "words_per_minute": wpm,
        "long_pauses_count": pauses,
    }
