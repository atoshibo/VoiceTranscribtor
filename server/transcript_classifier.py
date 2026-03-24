"""
Local LLM transcript classifier.

Classifies completed transcript text into exactly one category:
  meeting | technical | self_reflection | self_therapy

Uses llama-cpp-python with a GGUF model file for inference.
Runs on CPU by default to avoid GPU memory contention with Whisper.

This is a NON-CRITICAL post-processing stage: if classification fails
for any reason, the transcription job still completes successfully.
The classification result is written to classification.json and exposed
via the API, but never blocks the main transcription pipeline.

Configuration (all optional — classification is skipped if model path is unset):
  CLASSIFIER_ENABLED        "1" (default) or "0" to disable entirely
  CLASSIFIER_MODEL_PATH     absolute path to .gguf model file
  CLASSIFIER_N_GPU_LAYERS   layers to offload to GPU (default 0 = pure CPU)
  CLASSIFIER_MAX_TOKENS     max response tokens (default 200)
  CLASSIFIER_CONTEXT_SIZE   context window in tokens (default 2048)

Recommended models (Q4_K_M quantized GGUF):
  ~0.5B params: qwen2-0.5b       — ~400 MB, fastest, acceptable quality
  ~1.1B params: tinyllama-1.1b   — ~700 MB, good balance
  ~2.7B params: phi-2             — ~1.6 GB, best quality on CPU
  ~3.8B params: phi-3-mini        — ~2.2 GB, strongest, needs more RAM

Place the .gguf file in ./models/ and set CLASSIFIER_MODEL_PATH=/models/<filename>.gguf
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLASSIFIER_ENABLED = os.getenv("CLASSIFIER_ENABLED", "1") == "1"
CLASSIFIER_MODEL_PATH = os.getenv("CLASSIFIER_MODEL_PATH", "")
CLASSIFIER_N_GPU_LAYERS = int(os.getenv("CLASSIFIER_N_GPU_LAYERS", "0"))
CLASSIFIER_MAX_TOKENS = int(os.getenv("CLASSIFIER_MAX_TOKENS", "200"))
CLASSIFIER_CONTEXT_SIZE = int(os.getenv("CLASSIFIER_CONTEXT_SIZE", "2048"))

VALID_CATEGORIES = frozenset({
    "meeting", "technical", "self_reflection", "self_therapy",
})

# ---- Prompt sizing ----
# The prompt template (instructions, categories, JSON example) uses ~250 tokens
# for English.  Cyrillic and CJK text can use 2-4x more tokens per character
# than Latin text under BPE tokenization.
#
# Safety budget:
#   available_tokens = CONTEXT_SIZE - MAX_TOKENS - PROMPT_TEMPLATE_TOKENS
#   max_transcript_chars = available_tokens * CHARS_PER_TOKEN
#
# We use pessimistic estimates to avoid overflow on non-Latin text:
#   PROMPT_TEMPLATE_TOKENS = 350  (measured ~220 for English, padded for safety)
#   CHARS_PER_TOKEN = 1.5         (safe for Cyrillic; English is ~3-4)
#
# With CONTEXT_SIZE=2048 and MAX_TOKENS=200:
#   available = 2048 - 200 - 350 = 1498 tokens
#   max_chars = 1498 * 1.5 = 2247 chars
#
# This is conservative but safe.  Raising CLASSIFIER_CONTEXT_SIZE or using
# a model with a larger context window lets you classify longer transcripts.
_PROMPT_TEMPLATE_TOKENS = 350
_CHARS_PER_TOKEN_ESTIMATE = 1.5  # pessimistic for non-Latin (Cyrillic, CJK)

# ---------------------------------------------------------------------------
# Module-level model cache (loaded once, reused across jobs)
# ---------------------------------------------------------------------------
_llm_instance = None
_llm_model_path: Optional[str] = None


def _get_llm():
    """Load or return cached Llama model instance."""
    global _llm_instance, _llm_model_path

    if not CLASSIFIER_MODEL_PATH:
        return None

    model_path = Path(CLASSIFIER_MODEL_PATH)
    if not model_path.exists():
        print(f"[CLASSIFIER] Model file not found: {CLASSIFIER_MODEL_PATH}")
        return None

    # Return cached instance if same model
    if _llm_instance is not None and _llm_model_path == str(model_path):
        return _llm_instance

    try:
        from llama_cpp import Llama
    except ImportError:
        print(
            "[CLASSIFIER] llama-cpp-python not installed — "
            "classification disabled. Install with: pip install llama-cpp-python"
        )
        return None

    try:
        print(
            f"[CLASSIFIER] Loading model: {model_path.name} "
            f"(n_gpu_layers={CLASSIFIER_N_GPU_LAYERS}, "
            f"n_ctx={CLASSIFIER_CONTEXT_SIZE})"
        )
        t0 = time.monotonic()
        instance = Llama(
            model_path=str(model_path),
            n_ctx=CLASSIFIER_CONTEXT_SIZE,
            n_gpu_layers=CLASSIFIER_N_GPU_LAYERS,
            verbose=False,
        )
        elapsed = time.monotonic() - t0
        _llm_instance = instance
        _llm_model_path = str(model_path)
        print(f"[CLASSIFIER] Model loaded in {elapsed:.1f}s")
        return _llm_instance
    except Exception as e:
        print(f"[CLASSIFIER] Failed to load model: {e}")
        return None


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def _build_prompt(transcript_text: str) -> str:
    """Build the classification prompt, truncating transcript to fit context window."""
    available_tokens = CLASSIFIER_CONTEXT_SIZE - CLASSIFIER_MAX_TOKENS - _PROMPT_TEMPLATE_TOKENS
    max_transcript_chars = max(200, int(available_tokens * _CHARS_PER_TOKEN_ESTIMATE))

    truncated = transcript_text[:max_transcript_chars]
    if len(transcript_text) > max_transcript_chars:
        truncated += "\n[...truncated]"

    return (
        "Classify the following transcript into exactly one category.\n"
        "\n"
        "Categories:\n"
        "- meeting: conversation between multiple people, business discussion, "
        "planning, coordination, standup, review\n"
        "- technical: technical discussion, coding, engineering, debugging, "
        "system design, architecture talk\n"
        "- self_reflection: personal thoughts, journaling, thinking out loud, "
        "introspection, planning alone\n"
        "- self_therapy: emotional processing, self-help talk, mental health "
        "reflection, coping, feelings\n"
        "\n"
        "Transcript:\n"
        f"{truncated}\n"
        "\n"
        "Respond with ONLY a JSON object, no other text:\n"
        '{"category": "<one of: meeting, technical, self_reflection, self_therapy>", '
        '"confidence": <0.0 to 1.0>, "rationale": "<one sentence>"}\n'
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def _parse_response(raw: str) -> Optional[Dict[str, Any]]:
    """Extract and validate the JSON classification from LLM output."""
    text = raw.strip()

    # Find the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None

    category = str(obj.get("category", "")).strip().lower()
    # Normalize common variations
    category = category.replace("-", "_").replace(" ", "_")
    if category not in VALID_CATEGORIES:
        return None

    confidence = obj.get("confidence")
    try:
        confidence = float(confidence)
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    rationale = str(obj.get("rationale", ""))[:500]

    return {
        "category": category,
        "confidence": round(confidence, 2),
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_transcript(transcript_text: str) -> Optional[Dict[str, Any]]:
    """
    Classify transcript text into a category using a local LLM.

    Returns a dict with:
        status            — "success" | "skipped" | "failed"
        skip_reason       — (only if skipped) why: disabled / no_model / empty_transcript / model_load_failed
        failure_reason    — (only if failed)  why: inference_error / parse_failure
        category          — (only if success) one of the VALID_CATEGORIES
        confidence        — (only if success) 0.0 to 1.0
        rationale         — (only if success) short explanation from the LLM
        model_file        — GGUF filename used (if known)
        n_gpu_layers      — layers offloaded to GPU (0 = CPU only)
        inference_time_s  — wall-clock inference time (if inference ran)
        generated_at      — ISO timestamp
        transcript_chars  — length of transcript input
        prompt_chars      — length of constructed prompt (if built)

    Returns None ONLY if classification is disabled or no model is configured
    (i.e., the feature is intentionally off). All other outcomes return a dict
    so callers can distinguish success from failure.

    This function NEVER raises — all errors are caught and logged.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not CLASSIFIER_ENABLED:
        print("[CLASSIFIER] Disabled via CLASSIFIER_ENABLED=0")
        return None

    if not CLASSIFIER_MODEL_PATH:
        print("[CLASSIFIER] No model path configured (CLASSIFIER_MODEL_PATH unset)")
        return None

    base_meta = {
        "model_file": Path(CLASSIFIER_MODEL_PATH).name,
        "n_gpu_layers": CLASSIFIER_N_GPU_LAYERS,
        "generated_at": now_str,
        "transcript_chars": len(transcript_text) if transcript_text else 0,
    }

    if not transcript_text or not transcript_text.strip():
        print("[CLASSIFIER] Empty transcript — skipping classification")
        return {**base_meta, "status": "skipped", "skip_reason": "empty_transcript"}

    try:
        llm = _get_llm()
        if llm is None:
            return {**base_meta, "status": "skipped", "skip_reason": "model_load_failed"}

        prompt = _build_prompt(transcript_text)
        base_meta["prompt_chars"] = len(prompt)

        print(
            f"[CLASSIFIER] Running inference "
            f"(transcript_chars={len(transcript_text)}, "
            f"prompt_chars={len(prompt)}, "
            f"max_tokens={CLASSIFIER_MAX_TOKENS})"
        )

        t0 = time.monotonic()
        response = llm(
            prompt,
            max_tokens=CLASSIFIER_MAX_TOKENS,
            temperature=0.1,
            stop=["\n\n"],
        )
        elapsed = time.monotonic() - t0
        base_meta["inference_time_s"] = round(elapsed, 2)

        raw_text = response["choices"][0]["text"]
        print(f"[CLASSIFIER] Inference completed in {elapsed:.1f}s")

        result = _parse_response(raw_text)
        if result is None:
            print(
                f"[CLASSIFIER] Failed to parse LLM response: "
                f"{raw_text[:300]}"
            )
            return {**base_meta, "status": "failed", "failure_reason": "parse_failure"}

        # Attach metadata to success result
        result.update(base_meta)
        result["status"] = "success"

        print(
            f"[CLASSIFIER] Result: category={result['category']} "
            f"confidence={result['confidence']} "
            f"time={result['inference_time_s']}s"
        )

        return result

    except Exception as e:
        print(f"[CLASSIFIER] Classification failed (non-fatal): {e}")
        return {**base_meta, "status": "failed", "failure_reason": str(type(e).__name__)}
