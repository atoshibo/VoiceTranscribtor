# Migration Notes — Omi/Android Refactor

## What Changed

### Removed
- **Browser UI** — login page, sessions browser, session detail page, HTML templates
  (`login.html`, `index.html`, `sessions.html`, `session_detail.html` deleted)
- **Cookie-based session auth** — `SESSION_COOKIE`, `/login`, `/logout` endpoints
- **v1 API endpoints** — `/api/upload`, `/api/transcribe`, `/api/session`, `/api/sessions`,
  `/api/cancel`, `/api/download`, `/api/jobs` (v1 list/get), `/api/jobs/{id}/cancel`
- **v1 inline transcription pipeline** — large `process_transcription` async function that
  duplicated the worker's job (was ~200 lines in main.py)
- **jinja2** dependency removed from `requirements.txt` and `Dockerfile`
- **`detect_gpu_backend()`** from transcription.py — was a duplicate of `probe_cuda_health()`
- **`GpuUnavailableError` / `CudaComputeTypeError`** custom exception classes
- **`_get_device_mode()` / `_get_compute_types()` / `_get_cuda_compute_type_candidates()`** —
  replaced by module-level constants (DEVICE, COMPUTE_TYPE, etc.)
- **Compute-type fallback chain** — was float16 → int8_float16 → int8;
  now uses the configured type directly (default float16, works with nvidia runtime)

### Kept / Preserved
- All **API v2 endpoints** and their contracts (unchanged)
- **Bearer / X-Api-Token** auth
- **WAV chunk upload and merge** logic
- **Redis queue** job flow
- **GPU worker** (worker.py unchanged)
- **Diarization** pipeline
- **error.json** on worker failure, exposed via `GET /api/v2/jobs/{id}`
- **Subtitle generation** (SRT/VTT)
- **Transcript JSON** response format
- **`/api/diagnostics`** and **`/api/selftest`** endpoints
- **`/api/v2/health`** GPU status endpoint

### Simplified
- `transcription.py`: device selection is now 4 module-level constants, no state machine
- `main.py`: reduced from 1142 lines to ~100 lines
- Web service `requirements.txt`: 5 → 4 packages (removed jinja2)

---

## Running for Android / Omi

### Prerequisites
1. Copy `.env.example` to `.env` and set `AUTH_TOKEN`
2. Ensure Docker Desktop is running with NVIDIA GPU support
3. GPU requires `runtime: nvidia` in docker-compose.yml (already configured)

### Start
```bash
docker compose up -d
```

### Verify GPU
```bash
# Worker logs should show:
docker logs voicerecordtranscriptor-worker-1 | grep GPU-HEALTH
# Expected: [GPU-HEALTH] strict_cuda=True gpu_available=True selected_compute_type=float16

# Or via API:
curl -k https://localhost:8443/api/v2/health
# Expected: {"ok":true,"gpu_available":true,"selected_compute_type":"float16",...}
```

### Verify API v2 (curl sanity checklist)

```bash
TOKEN="your_auth_token_here"
BASE="https://localhost:8443"

# 1. Health (no auth)
curl -sk $BASE/api/v2/health | python3 -m json.tool

# 2. Create session
SESSION=$(curl -sk -X POST $BASE/api/v2/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mode":"stream"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo "Session: $SESSION"

# 3. Upload a WAV chunk (replace test.wav with a real WAV file)
curl -sk -X POST $BASE/api/v2/sessions/$SESSION/chunks \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test.wav" \
  -F "chunk_index=0" \
  -F "is_final=true"

# 4. Finalize
JOB=$(curl -sk -X POST $BASE/api/v2/sessions/$SESSION/finalize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"language":"en","model_size":"small"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job: $JOB"

# 5. Poll job (repeat until state=done)
curl -sk $BASE/api/v2/jobs/$JOB \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool

# 6. Fetch transcript
curl -sk $BASE/api/v2/sessions/$SESSION/transcript \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool

# 7. If job failed — get full error details
curl -sk $BASE/api/v2/jobs/$JOB/error \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

### Error visibility
When a worker job fails, `error.json` is written to the session directory and exposed via:
- `GET /api/v2/jobs/{job_id}` — includes `error.error_message` in the response
- `GET /api/v2/jobs/{job_id}/error` — full error.json with traceback

### Auth
Only these are supported (URL tokens removed):
```
Authorization: Bearer <AUTH_TOKEN>
X-Api-Token: <AUTH_TOKEN>
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `AUTH_TOKEN` | — | Required. Shared secret for all API calls |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `WHISPER_STRICT_CUDA` | `1` | `1` = fail if GPU unavailable; `0` = allow CPU fallback |
| `CUDA_DEVICE_INDEX` | `0` | GPU index |
| `WHISPER_COMPUTE_TYPE_CUDA` | `float16` | `float16`, `int8_float16`, `int8` |
| `WHISPER_MODEL` | `small` | Whisper model size |
| `MAX_CHUNK_MB` | `30` | Max WAV chunk size in MB |
| `RATE_LIMIT_PER_MINUTE` | `60` | API rate limit per token |
