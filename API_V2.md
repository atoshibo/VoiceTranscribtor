# API v2 — Android Client Guide

Mobile-friendly REST API for VoiceRecordTranscriptor.
Base URL: `https://api.surv.life` (or your server address)

---

## Authentication

All `/api/v2/*` endpoints (except `/api/v2/health`) require a token.

**Preferred (mobile):**
```
Authorization: Bearer <TOKEN>
```

**Alternative:**
```
X-Api-Token: <TOKEN>
```

> The old `?token=` query string is deprecated for API use (still works for the web UI).
> Tokens are **never** logged or reflected in error messages.

Rate limit: **60 requests / minute** per token (configurable via `RATE_LIMIT_PER_MINUTE` env var).
Exceeding the limit returns **HTTP 429** with `Retry-After: 60`.

---

## Session State Machine

```
[POST /sessions]
      │
      ▼
  created
      │
      │ [POST /sessions/{id}/chunks] × N
      ▼
chunks_uploaded
      │
      │ [POST /sessions/{id}/finalize]
      ▼
  finalized ──► queued (Redis)
                   │
              [worker picks up]
                   │
                   ▼
                running
                   │
              ┌────┴────┐
              ▼         ▼
            done      error
```

Job polling: `GET /api/v2/jobs/{job_id}` — use exponential back-off (start at 2s).

---

## Endpoints

### `GET /api/v2/health`

No auth required. Returns server liveness + GPU availability.

```bash
curl https://api.surv.life/api/v2/health
```

Response:
```json
{
  "ok": true,
  "version": "2.0.0",
  "gpu_available": false,
  "gpu_reason": "CUDA driver version is insufficient for CUDA runtime version",
  "selected_device": "cuda",
  "selected_compute_type": "none",
  "strict_cuda": true,
  "timestamp": "2026-02-27T12:00:00Z"
}
```

---

### `POST /api/v2/sessions`

Create a new recording session.

```bash
curl -X POST https://api.surv.life/api/v2/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "DF:2B:9C:30:78:0E",
    "client_session_id": "550e8400-e29b-41d4-a716-446655440000",
    "started_at_utc": "2026-02-27T12:10:00Z",
    "sample_rate_hz": 16000,
    "channels": 1,
    "format": "wav",
    "mode": "stream"
  }'
```

**Request body fields** (all optional):

| Field              | Type   | Default    | Description                        |
|--------------------|--------|------------|------------------------------------|
| device_id          | string | null       | Device MAC / UUID for tracking     |
| client_session_id  | string | null       | Client-side session UUID           |
| started_at_utc     | string | null       | ISO-8601 recording start time      |
| sample_rate_hz     | int    | 16000      | Audio sample rate                  |
| channels           | int    | 1          | Audio channels (mono = 1)          |
| format             | string | "wav"      | Audio format (only "wav" supported)|
| mode               | string | "stream"   | "stream" (chunks) or "file" (single WAV) |

Response `200`:
```json
{
  "session_id": "b3d7f8a1-...",
  "upload_url": "/api/v2/sessions/b3d7f8a1-.../chunks"
}
```

---

### `POST /api/v2/sessions/{session_id}/chunks`

Upload one WAV chunk. Use `multipart/form-data`.

```bash
curl -X POST "https://api.surv.life/api/v2/sessions/$SESSION_ID/chunks" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@chunk_0000.wav" \
  -F "chunk_index=0" \
  -F "chunk_started_ms=0" \
  -F "chunk_duration_ms=10000" \
  -F "is_final=false"
```

**Form fields:**

| Field            | Type | Required | Description                                  |
|------------------|------|----------|----------------------------------------------|
| file             | file | yes      | WAV audio data (PCM16)                       |
| chunk_index      | int  | yes      | 0-based sequential index                     |
| chunk_started_ms | int  | no       | Offset from session start in milliseconds    |
| chunk_duration_ms| int  | no       | Duration of this chunk in milliseconds       |
| is_final         | bool | no       | Set true on last chunk                       |

Limits: max **30 MB** per chunk (configurable via `MAX_CHUNK_MB`), max **200 chunks** per session.

Response `200`:
```json
{
  "accepted": true,
  "session_id": "b3d7f8a1-...",
  "chunk_index": 0,
  "status": "accepted"
}
```

---

### `POST /api/v2/sessions/{session_id}/finalize`

Trigger transcription after all chunks are uploaded.
Server merges chunks → validates WAV → enqueues worker job.

```bash
curl -X POST "https://api.surv.life/api/v2/sessions/$SESSION_ID/finalize" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "run_diarization": true,
    "language": "auto",
    "model_size": "base"
  }'
```

**Request body fields:**

| Field           | Type   | Default | Description                                          |
|-----------------|--------|---------|------------------------------------------------------|
| run_diarization | bool   | false   | Enable speaker diarization                           |
| language        | string | "auto"  | ISO 639-1 code ("ru", "en", …) or "auto" to detect  |
| model_size      | string | "base"  | Whisper model: tiny/base/small/medium/large          |
| speaker_count   | int    | 2 (if diarization) / 1 | Expected number of speakers          |

Response `200`:
```json
{
  "session_id": "b3d7f8a1-...",
  "job_id": "b3d7f8a1-...",
  "status_url": "/api/v2/jobs/b3d7f8a1-..."
}
```

> `job_id` equals `session_id` in this implementation.

---

### `GET /api/v2/jobs/{job_id}`

Poll transcription progress.

```bash
curl "https://api.surv.life/api/v2/jobs/$JOB_ID" \
  -H "Authorization: Bearer $TOKEN"
```

Response `200`:
```json
{
  "job_id": "b3d7f8a1-...",
  "state": "running",
  "progress": 0.45,
  "message": "transcribing",
  "started_at": "2026-02-27T12:10:05Z",
  "finished_at": null
}
```

When a job fails, the response includes structured error details:
```json
{
  "job_id": "b3d7f8a1-...",
  "state": "error",
  "progress": 0.0,
  "message": "CUDA driver version is insufficient for CUDA runtime version",
  "started_at": "2026-02-27T12:10:05Z",
  "finished_at": "2026-02-27T12:10:06Z",
  "error": {
    "error_type": "RuntimeError",
    "error_message": "CUDA driver version is insufficient for CUDA runtime version",
    "traceback": "...",
    "timestamp": "2026-02-27T12:10:06Z",
    "stage": "transcription"
  }
}
```

**state values:**

| State   | Meaning                            |
|---------|------------------------------------|
| queued  | Waiting in Redis queue             |
| running | Worker is processing               |
| done    | Complete, transcript available     |
| error   | Processing failed                  |

**Recommended polling strategy:**
```
2s → 4s → 8s → 16s → 30s → 30s → …  (max 30s interval)
```

---

### `GET /api/v2/sessions/{session_id}/transcript`

Retrieve the structured transcript.
Returns **HTTP 202** if not ready yet.

```bash
curl "https://api.surv.life/api/v2/sessions/$SESSION_ID/transcript" \
  -H "Authorization: Bearer $TOKEN"
```

Response `200` (done):
```json
{
  "session_id": "b3d7f8a1-...",
  "text": "Hello world. This is a test recording.",
  "segments": [
    {
      "start_ms": 0,
      "end_ms": 1240,
      "speaker": "SPEAKER_00",
      "text": "Hello world."
    },
    {
      "start_ms": 1500,
      "end_ms": 3800,
      "speaker": "SPEAKER_01",
      "text": "This is a test recording."
    }
  ],
  "words": [],
  "formats": {
    "srt_url": "/api/v2/sessions/b3d7f8a1-.../subtitle.srt",
    "vtt_url": "/api/v2/sessions/b3d7f8a1-.../subtitle.vtt"
  }
}
```

Response `202` (not ready):
```json
{
  "detail": {
    "message": "Transcript not ready yet.",
    "state": "running",
    "progress": {"upload": 100, "processing": 45, "stage": "transcribing"}
  }
}
```

---

### `GET /api/v2/sessions/{session_id}/subtitle.srt`
### `GET /api/v2/sessions/{session_id}/subtitle.vtt`

Download subtitle files.

```bash
curl "https://api.surv.life/api/v2/sessions/$SESSION_ID/subtitle.srt" \
  -H "Authorization: Bearer $TOKEN" \
  -o transcript.srt
```

Returns `404` if job is not done yet.

---

### `DELETE /api/v2/sessions/{session_id}`

Delete all session data (audio, chunks, transcripts).

```bash
curl -X DELETE "https://api.surv.life/api/v2/sessions/$SESSION_ID" \
  -H "Authorization: Bearer $TOKEN"
```

Response `200`:
```json
{"deleted": true, "session_id": "b3d7f8a1-..."}
```

---

## Error Responses

All errors return JSON:

```json
{"detail": "Human-readable error message"}
```

| HTTP | Meaning                                      |
|------|----------------------------------------------|
| 400  | Bad request (missing field, no chunks, etc.) |
| 401  | Missing authentication header                |
| 403  | Invalid token                                |
| 404  | Session / job / file not found               |
| 409  | Conflict (session already finalized/deleted) |
| 413  | Chunk too large or too many chunks           |
| 429  | Rate limit exceeded                          |
| 500  | Server error (merge failed, etc.)            |

---

## Android Client Integration Guide

### Minimal flow (single file upload)

```kotlin
// 1. Create session
val session = api.createSession(
    deviceId = Build.ID,
    clientSessionId = UUID.randomUUID().toString(),
    startedAtUtc = Instant.now().toString(),
    sampleRateHz = 16000,
    channels = 1,
    mode = "file"
)

// 2. Upload the WAV file as a single chunk
api.uploadChunk(
    sessionId = session.sessionId,
    file = wavFile,
    chunkIndex = 0,
    chunkStartedMs = 0,
    isFinal = true
)

// 3. Finalize
val job = api.finalize(
    sessionId = session.sessionId,
    runDiarization = false,
    language = "auto",
    modelSize = "base"
)

// 4. Poll until done
var state = "queued"
var delay = 2_000L
while (state !in listOf("done", "error")) {
    delay(delay)
    delay = minOf(delay * 2, 30_000L)
    val status = api.getJobStatus(job.jobId)
    state = status.state
}

// 5. Fetch transcript
val transcript = api.getTranscript(session.sessionId)
```

### Chunked streaming flow

Split recording into ~10s chunks, upload concurrently or sequentially.
Use `chunk_started_ms` to indicate each chunk's start offset from session start.

```kotlin
var chunkIndex = 0
var sessionStartMs = System.currentTimeMillis()

fun onChunkReady(wavBytes: ByteArray, isFinal: Boolean) {
    val startedMs = System.currentTimeMillis() - sessionStartMs
    api.uploadChunk(
        sessionId = session.sessionId,
        fileBytes = wavBytes,
        chunkIndex = chunkIndex++,
        chunkStartedMs = startedMs.toInt(),
        chunkDurationMs = 10_000,
        isFinal = isFinal
    )
    if (isFinal) {
        api.finalize(session.sessionId, runDiarization = true, language = "auto")
    }
}
```

### Authentication setup (OkHttp)

```kotlin
val client = OkHttpClient.Builder()
    .addInterceptor { chain ->
        val request = chain.request().newBuilder()
            .header("Authorization", "Bearer $API_TOKEN")
            .build()
        chain.proceed(request)
    }
    .build()
```

> Store `API_TOKEN` in Android Keystore or `EncryptedSharedPreferences`.
> Never put it in the URL, logs, or screenshots.

---

## Environment Variables (server)

| Variable                   | Default          | Description                                               |
|----------------------------|------------------|-----------------------------------------------------------|
| AUTH_TOKEN                 | —                | Shared secret token (required)                            |
| MAX_CHUNK_MB               | 30               | Max size per WAV chunk in MB                              |
| MAX_SESSION_CHUNKS         | 200              | Max chunks per session                                    |
| RATE_LIMIT_PER_MINUTE      | 60               | API requests per minute per token                         |
| REDIS_HOST                 | redis            | Redis hostname                                            |
| REDIS_QUEUE                | transcription_jobs | Queue name                                             |
| WHISPER_DEVICE             | cuda             | Device mode (GPU-only mode requires `cuda`)               |
| WHISPER_STRICT_CUDA        | 1                | If 1, no CPU fallback; CUDA errors fail jobs and startup  |
| WHISPER_COMPUTE_TYPE_CUDA  | float16          | Primary CUDA compute type (e.g., float16, int8_float16)   |
| WHISPER_COMPUTE_TYPE_CUDA_FALLBACKS | int8_float16,int8 | CUDA compute_type fallbacks (CUDA-only)        |
| WHISPER_COMPUTE_TYPE_CPU   | int8             | CPU compute type (legacy / diagnostics only)              |

---

## Quick GPU / CPU self-test

1. **Check health and GPU status:**

```bash
curl https://api.surv.life/api/v2/health
```

2. **Run a tiny end-to-end transcription in CPU mode (no GPU required):**

```bash
# In your deployment config (e.g., docker-compose), force CPU:
#   WHISPER_DEVICE=cpu

# Then from a client:
SESSION=$(curl -s -X POST https://api.surv.life/api/v2/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mode": "file"}' | jq -r .session_id)

curl -X POST "https://api.surv.life/api/v2/sessions/$SESSION/chunks" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test.wav" \
  -F "chunk_index=0" \
  -F "is_final=true"

curl -X POST "https://api.surv.life/api/v2/sessions/$SESSION/finalize" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"language": "en", "model_size": "tiny"}'
```
