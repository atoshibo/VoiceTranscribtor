import os
import subprocess
from faster_whisper import WhisperModel
from typing import List, Dict, Tuple, Optional, Callable
import re
import math

# Initialize model (load on first use, per model size)
_model_cache: Dict[str, WhisperModel] = {}
_device_diagnostics: Dict = {}


def get_device_diagnostics() -> Dict:
    """Get diagnostic info about available compute devices."""
    global _device_diagnostics
    if _device_diagnostics:
        return _device_diagnostics

    diag = {
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "force_device": os.getenv("FORCE_DEVICE"),
        "cuda_device_index": os.getenv("CUDA_DEVICE_INDEX"),
        "compute_type": os.getenv("COMPUTE_TYPE"),
        "nvidia_smi_available": False,
        "nvidia_smi_output": "",
        "detected_gpus": [],
        "selected_device": "unknown",
        "ctranslate2_available": False
    }

    # Check nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            timeout=3,
            text=True
        )
        if result.returncode == 0:
            diag["nvidia_smi_available"] = True
            diag["nvidia_smi_output"] = result.stdout.strip()
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    diag["detected_gpus"].append({
                        "index": parts[0],
                        "name": parts[1] if len(parts) > 1 else "",
                        "utilization": parts[2] if len(parts) > 2 else "",
                        "memory_used": parts[3] if len(parts) > 3 else "",
                        "memory_total": parts[4] if len(parts) > 4 else ""
                    })
    except Exception as e:
        diag["nvidia_smi_error"] = str(e)

    # Check CTranslate2
    try:
        import ctranslate2  # noqa: F401
        diag["ctranslate2_available"] = True
        diag["ctranslate2_version"] = ctranslate2.__version__
    except Exception:
        pass

    _device_diagnostics = diag
    return diag


def select_device() -> Tuple[str, int, str]:
    """
    Select device based on environment variables.
    Returns: (device, device_index, compute_type)
    """
    force_device = os.getenv("FORCE_DEVICE", "").lower().strip()
    cuda_device_index = int(os.getenv("CUDA_DEVICE_INDEX", "0"))
    compute_type = os.getenv("COMPUTE_TYPE", "").lower().strip()

    # Force device override
    if force_device == "cpu":
        return "cpu", 0, (compute_type or "int8")

    if force_device == "cuda":
        ct = compute_type or "float16"
        return "cuda", cuda_device_index, ct

    # Auto-detect
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None:
        ct = compute_type or "float16"
        return "cuda", cuda_device_index, ct

    # Check nvidia-smi
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=2, text=True)
        if result.returncode == 0 and "GPU" in result.stdout:
            ct = compute_type or "float16"
            return "cuda", cuda_device_index, ct
    except Exception:
        pass

    # Fallback to CPU
    return "cpu", 0, (compute_type or "int8")


def run_smoke_test(model_size: str = "tiny") -> Dict:
    """
    Run a quick smoke test to verify GPU setup.
    Returns diagnostic info.
    """
    result = {
        "success": False,
        "device": "unknown",
        "device_index": 0,
        "compute_type": "unknown",
        "error": None,
        "model_size": model_size
    }

    try:
        device, device_index, compute_type = select_device()
        result["device"] = device
        result["device_index"] = device_index
        result["compute_type"] = compute_type

        print(f"[SMOKE TEST] Attempting to load model: {model_size}, device={device}, device_index={device_index}, compute_type={compute_type}")

        _ = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            device_index=device_index if device == "cuda" else 0
        )

        result["success"] = True
        print(f"[SMOKE TEST] SUCCESS - Model loaded on {device}:{device_index}")
        return result

    except Exception as e:
        result["error"] = str(e)
        print(f"[SMOKE TEST] FAILED - {e}")
        return result


def get_model(model_size: str = "base", device: str = None, compute_type: str = None) -> WhisperModel:
    """
    Get or create WhisperModel instance.
    Models are cached per size to avoid reloading.
    Supports explicit GPU selection via env vars.
    """
    global _model_cache

    if model_size not in _model_cache:
        # Use environment-based device selection if not provided
        if device is None or compute_type is None:
            auto_device, auto_index, auto_compute = select_device()
            device = device or auto_device
            compute_type = compute_type or auto_compute
            device_index = auto_index
        else:
            device_index = int(os.getenv("CUDA_DEVICE_INDEX", "0"))

        print(f"[WHISPER] Loading model: {model_size}, device={device}, device_index={device_index}, compute_type={compute_type}")
        diag = get_device_diagnostics()
        try:
            print(f"[WHISPER] nvidia-smi detected_gpus={len(diag.get('detected_gpus', []))} CUDA_VISIBLE_DEVICES={diag.get('cuda_visible_devices')}")
        except Exception:
            pass

        model_kwargs = {
            "device": device,
            "compute_type": compute_type,
        }
        if device == "cuda":
            model_kwargs["device_index"] = device_index

        _model_cache[model_size] = WhisperModel(model_size, **model_kwargs)
        print(f"[WHISPER] Model loaded OK on {device}:{device_index}")

    return _model_cache[model_size]


def _smooth_progress_from_segments(segment_count: int) -> int:
    """
    Produce a smooth progress number in [10..90] based on how many segments we've consumed,
    without knowing total segments in advance.
    This avoids "stuck at 69%" and looks continuous.
    """
    # exponential saturation curve
    # seg=0 => 10, seg~120 => ~60, seg~250 => ~78, seg~400 => ~86
    value = 10 + int(80 * (1.0 - math.exp(-segment_count / 140.0)))
    return min(90, max(10, value))


def transcribe_audio(
    audio_path: str,
    language: str = "en",
    diarization_enabled: bool = False,
    speaker_count: int = 1,
    model_size: str = "base",
    timestamps_enabled: bool = True,
    progress_callback: Optional[Callable[[int], None]] = None
) -> Tuple[str, List[Dict]]:
    """
    Transcribe audio file and return transcript text and timestamps.

    Returns:
        (transcript_text, timestamps_list)
    """
    model = get_model(model_size)

    # NOTE: faster-whisper uses CTranslate2 under the hood.
    # batch_size is supported in newer versions; if not, it will error.
    # We'll include it but fall back if needed.
    transcribe_kwargs = {
        "language": language if language in ["en", "ru", "fr"] else None,
        "word_timestamps": timestamps_enabled,

        # Quality vs speed knobs
        "beam_size": 5,
        "best_of": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "temperature": 0.0,

        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
        "initial_prompt": None,

        # VAD: True makes less garbage, sometimes slightly slower but often better.
        "vad_filter": True,
        "vad_parameters": None,

        # Try to increase GPU throughput
        "batch_size": 16,
    }

    print("[TRANSCRIBE] Starting transcription...")
    if progress_callback:
        progress_callback(5)

    # Run transcription (generator of segments)
    try:
        segments, info = model.transcribe(audio_path, **transcribe_kwargs)
    except TypeError as e:
        # batch_size not supported in older faster-whisper -> retry without it
        if "batch_size" in str(e):
            transcribe_kwargs.pop("batch_size", None)
            segments, info = model.transcribe(audio_path, **transcribe_kwargs)
        else:
            raise

    transcript_parts: List[str] = []
    timestamps: List[Dict] = []
    segment_count = 0

    # STREAMING: process segments as they come, no huge RAM usage
    for segment in segments:
        segment_count += 1

        text = (segment.text or "").strip()
        if text:
            transcript_parts.append(text)
            timestamps.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text
            })

        if progress_callback and (segment_count % 5 == 0):
            progress_callback(_smooth_progress_from_segments(segment_count))

    print(f"[TRANSCRIBE] Done. language={getattr(info, 'language', None)} prob={getattr(info, 'language_probability', 0.0):.2f} segments={segment_count}")

    if progress_callback:
        progress_callback(90)

    full_transcript = " ".join(transcript_parts)
    return full_transcript, timestamps


def generate_summary(transcript: str) -> Dict:
    """
    Generate a simple extractive summary (5 bullet points).
    For a more sophisticated approach, use an LLM API.
    """
    sentences = re.split(r"[.!?]+", transcript)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) == 0:
        return {"bullets": ["No content to summarize."]}

    bullets = []
    if len(sentences) >= 5:
        indices = [0, len(sentences) // 4, len(sentences) // 2, 3 * len(sentences) // 4, -1]
        bullets = [sentences[i] for i in indices if 0 <= i < len(sentences)]
    else:
        bullets = sentences[:5]

    bullets = bullets[:5]
    return {
        "bullets": bullets,
        "note": "Extractive summary - for better results, use LLM-based summarization"
    }


def generate_analytics(transcript: str, timestamps: List[Dict]) -> Dict:
    """
    Generate basic analytics: duration, word count, WPM, pauses, signals.
    """
    if not timestamps:
        return {
            "duration": 0,
            "word_count": 0,
            "words_per_minute": 0,
            "long_pauses_count": 0,
            "signals": {}
        }

    duration = timestamps[-1]["end"] if timestamps else 0

    words = transcript.split()
    word_count = len(words)

    words_per_minute = (word_count / duration * 60) if duration > 0 else 0

    long_pauses = 0
    pause_threshold = 2.0
    for i in range(1, len(timestamps)):
        gap = timestamps[i]["start"] - timestamps[i - 1]["end"]
        if gap > pause_threshold:
            long_pauses += 1

    signals: Dict[str, str] = {}
    if words_per_minute < 100:
        signals["speaking_rate"] = "slow"
    elif words_per_minute > 200:
        signals["speaking_rate"] = "fast"
    else:
        signals["speaking_rate"] = "normal"

    pause_frequency = long_pauses / duration if duration > 0 else 0
    if pause_frequency > 0.5:
        signals["pause_frequency"] = "high"
    elif pause_frequency < 0.1:
        signals["pause_frequency"] = "low"
    else:
        signals["pause_frequency"] = "normal"

    signals["disclaimer"] = "These are heuristic signals only, not psychological analysis"

    return {
        "duration": round(duration, 2),
        "word_count": word_count,
        "words_per_minute": round(words_per_minute, 1),
        "long_pauses_count": long_pauses,
        "signals": signals
    }
