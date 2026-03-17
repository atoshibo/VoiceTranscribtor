"""
Integration tests for API v2.

Usage (against a running server):
    pytest server/tests/test_api_v2.py -v \
        --base-url=https://api.surv.life \
        --token=YOUR_TOKEN

Or run against a local test instance:
    AUTH_TOKEN=testtoken uvicorn main:app --port 8001 &
    pytest server/tests/test_api_v2.py -v \
        --base-url=http://localhost:8001 \
        --token=testtoken

Requirements:
    pip install pytest requests

The tests generate a real minimal WAV file (PCM16, 16kHz, mono, 1s of silence)
so no external audio fixture is needed.
"""
import io
import struct
import time
import wave
import pytest
import requests


# ---------------------------------------------------------------------------
# Pytest CLI options
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption("--base-url", default="http://localhost:8000", help="Server base URL")
    parser.addoption("--token", default="", help="API auth token")


@pytest.fixture(scope="session")
def base_url(request):
    return request.config.getoption("--base-url").rstrip("/")


@pytest.fixture(scope="session")
def token(request):
    return request.config.getoption("--token")


@pytest.fixture(scope="session")
def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# WAV fixture factory
# ---------------------------------------------------------------------------
def make_wav_bytes(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a minimal PCM16 WAV file (silence) in memory."""
    n_frames = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(struct.pack("<" + "h" * n_frames, *([0] * n_frames)))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Health check (no auth)
# ---------------------------------------------------------------------------
class TestHealth:
    def test_health_no_auth(self, base_url):
        r = requests.get(f"{base_url}/api/v2/health", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert data["ok"] is True
        assert "version" in data
        assert "gpu_available" in data

    def test_health_with_auth(self, base_url, auth_headers):
        r = requests.get(f"{base_url}/api/v2/health", headers=auth_headers, timeout=5)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Authentication tests
# ---------------------------------------------------------------------------
class TestAuth:
    def test_missing_token_returns_401(self, base_url):
        r = requests.post(f"{base_url}/api/v2/sessions", json={}, timeout=5)
        assert r.status_code == 401

    def test_wrong_token_returns_403(self, base_url):
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={},
            headers={"Authorization": "Bearer wrong-token-xyz"},
            timeout=5,
        )
        assert r.status_code == 403

    def test_x_api_token_header(self, base_url, token):
        """X-Api-Token header should work as a Bearer alternative."""
        if not token:
            pytest.skip("No token configured")
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={},
            headers={"X-Api-Token": token},
            timeout=5,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"


# ---------------------------------------------------------------------------
# GPU monitoring endpoint
# ---------------------------------------------------------------------------
class TestGPUStatus:
    def test_gpu_status_requires_auth(self, base_url):
        r = requests.get(f"{base_url}/api/v2/system/gpu", timeout=5)
        assert r.status_code == 401

    def test_gpu_status_shape(self, base_url, auth_headers):
        if not auth_headers.get("Authorization", "").endswith(" "):
            pass  # token may be empty, test may 401
        r = requests.get(f"{base_url}/api/v2/system/gpu", headers=auth_headers, timeout=10)
        if r.status_code == 403:
            pytest.skip("No valid token configured")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        # Core fields
        assert "gpu_available" in data
        assert "active_jobs" in data
        assert isinstance(data["active_jobs"], list)
        assert "queue_depth" in data
        assert isinstance(data["queue_depth"], int)
        assert "partial_queue_depth" in data
        assert isinstance(data["partial_queue_depth"], int)
        assert "processing_stats" in data
        assert "active_job_count" in data
        assert "timestamp" in data
        # Diagnostics fields (may be None if worker not running)
        assert "gpu_name" in data
        assert "utilization_percent" in data
        assert "memory_used_mb" in data
        assert "memory_total_mb" in data
        assert "worker_started_at" in data

    def test_gpu_status_active_jobs_have_elapsed(self, base_url, auth_headers):
        """Active jobs in GPU status should include elapsed_s."""
        r = requests.get(f"{base_url}/api/v2/system/gpu", headers=auth_headers, timeout=10)
        if r.status_code != 200:
            pytest.skip("GPU endpoint not available")
        data = r.json()
        for job in data.get("active_jobs", []):
            assert "elapsed_s" in job, f"active_jobs entry missing elapsed_s: {job}"


# ---------------------------------------------------------------------------
# Full integration flow
# ---------------------------------------------------------------------------
class TestSessionFlow:
    """
    Full end-to-end flow:
      create session → upload 2 chunks → finalize → poll → get transcript
    """

    @pytest.fixture(scope="class")
    def session_id(self, base_url, auth_headers):
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={
                "device_id": "TEST:00:00:00:00:01",
                "client_session_id": "test-client-session-001",
                "started_at_utc": "2026-01-01T00:00:00Z",
                "sample_rate_hz": 16000,
                "channels": 1,
                "format": "wav",
                "mode": "stream",
            },
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200, f"Create session failed: {r.text}"
        data = r.json()
        assert "session_id" in data
        assert "upload_url" in data
        return data["session_id"]

    def test_create_session(self, session_id):
        assert session_id is not None
        assert len(session_id) > 8

    def test_upload_chunk_0(self, base_url, auth_headers, session_id):
        wav_bytes = make_wav_bytes(duration_s=1.0)
        r = requests.post(
            f"{base_url}/api/v2/sessions/{session_id}/chunks",
            headers=auth_headers,
            files={"file": ("chunk_0.wav", wav_bytes, "audio/wav")},
            data={
                "chunk_index": "0",
                "chunk_started_ms": "0",
                "chunk_duration_ms": "1000",
                "is_final": "false",
            },
            timeout=30,
        )
        assert r.status_code == 200, f"Upload chunk 0 failed: {r.text}"
        data = r.json()
        assert data["accepted"] is True
        assert data["chunk_index"] == 0
        assert "chunk_count" in data

    def test_session_status_receiving(self, base_url, auth_headers, session_id):
        """After first chunk, session state should be 'receiving'."""
        r = requests.get(
            f"{base_url}/api/v2/sessions/{session_id}/status",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200, f"Get session status failed: {r.text}"
        data = r.json()
        assert data["state"] == "receiving", f"Expected 'receiving', got '{data['state']}'"
        assert data["chunk_count"] == 1
        assert data["first_chunk_at"] is not None
        assert data["last_chunk_at"] is not None
        assert "partial_transcript_available" in data

    def test_upload_chunk_1_final(self, base_url, auth_headers, session_id):
        wav_bytes = make_wav_bytes(duration_s=1.0)
        r = requests.post(
            f"{base_url}/api/v2/sessions/{session_id}/chunks",
            headers=auth_headers,
            files={"file": ("chunk_1.wav", wav_bytes, "audio/wav")},
            data={
                "chunk_index": "1",
                "chunk_started_ms": "1000",
                "chunk_duration_ms": "1000",
                "is_final": "true",
            },
            timeout=30,
        )
        assert r.status_code == 200, f"Upload chunk 1 failed: {r.text}"
        data = r.json()
        assert data["accepted"] is True
        assert data["chunk_index"] == 1

    def test_finalize(self, base_url, auth_headers, session_id):
        r = requests.post(
            f"{base_url}/api/v2/sessions/{session_id}/finalize",
            json={
                "run_diarization": False,
                "language": "auto",
                "model_size": "tiny",
            },
            headers=auth_headers,
            timeout=30,
        )
        assert r.status_code == 200, f"Finalize failed: {r.text}"
        data = r.json()
        assert "job_id" in data
        assert "status_url" in data

    def test_session_status_queued_after_finalize(self, base_url, auth_headers, session_id):
        """After finalize, session status endpoint should show 'queued' or later."""
        r = requests.get(
            f"{base_url}/api/v2/sessions/{session_id}/status",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["state"] in ("queued", "running", "done", "error"), \
            f"Unexpected state: {data['state']}"
        assert data["finalize_requested_at"] is not None
        assert data["queued_at"] is not None

    def test_job_status_queued_or_running(self, base_url, auth_headers, session_id):
        r = requests.get(
            f"{base_url}/api/v2/jobs/{session_id}",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200, f"Get job status failed: {r.text}"
        data = r.json()
        assert data["state"] in ("queued", "running", "done", "error"), \
            f"Unexpected state: {data['state']}"
        # progress is a dict with processing (0-100) and stage fields
        assert isinstance(data["progress"], dict), \
            f"progress should be a dict, got {type(data['progress'])}"
        assert 0 <= data["progress"].get("processing", 0) <= 100
        # Timing fields present
        assert "queued_at" in data

    def test_poll_until_done_or_timeout(self, base_url, auth_headers, session_id):
        """Poll with exponential back-off for up to 120s."""
        deadline = time.time() + 120
        delay = 2
        state = "queued"
        while time.time() < deadline:
            time.sleep(delay)
            delay = min(delay * 2, 15)
            r = requests.get(
                f"{base_url}/api/v2/jobs/{session_id}",
                headers=auth_headers,
                timeout=10,
            )
            assert r.status_code == 200
            data = r.json()
            state = data["state"]
            if state in ("done", "error"):
                break

        assert state in ("queued", "running", "done", "error"), f"Unexpected state: {state}"

    def test_job_timing_fields_when_done(self, base_url, auth_headers, session_id):
        """When job is done, started_at and finished_at should be present."""
        r = requests.get(
            f"{base_url}/api/v2/jobs/{session_id}",
            headers=auth_headers,
            timeout=10,
        )
        data = r.json()
        if data["state"] != "done":
            pytest.skip(f"Job not done yet (state={data['state']})")
        assert data.get("started_at") is not None
        assert data.get("finished_at") is not None

    def test_get_transcript_when_done(self, base_url, auth_headers, session_id):
        """If job is done, transcript should have expected shape."""
        r = requests.get(
            f"{base_url}/api/v2/jobs/{session_id}",
            headers=auth_headers,
            timeout=10,
        )
        state = r.json().get("state")

        if state != "done":
            pytest.skip(f"Job not done yet (state={state}), skipping transcript check")

        r = requests.get(
            f"{base_url}/api/v2/sessions/{session_id}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200, f"Get transcript failed: {r.text}"
        data = r.json()
        assert "session_id" in data
        assert "text" in data
        assert isinstance(data["segments"], list)
        assert isinstance(data["words"], list)
        assert "formats" in data
        assert "srt_url" in data["formats"]
        assert "vtt_url" in data["formats"]

    def test_transcript_returns_202_when_not_done(self, base_url, auth_headers):
        """A freshly created session that was never finalized should return 404 or 202."""
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]
        wav_bytes = make_wav_bytes(0.5)
        requests.post(
            f"{base_url}/api/v2/sessions/{sid}/chunks",
            headers=auth_headers,
            files={"file": ("c.wav", wav_bytes, "audio/wav")},
            data={"chunk_index": "0"},
            timeout=10,
        )
        r = requests.get(
            f"{base_url}/api/v2/sessions/{sid}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code in (202, 404), f"Expected 202 or 404, got {r.status_code}"

        # Cleanup
        requests.delete(
            f"{base_url}/api/v2/sessions/{sid}",
            headers=auth_headers,
            timeout=5,
        )


# ---------------------------------------------------------------------------
# Session status endpoint
# ---------------------------------------------------------------------------
class TestSessionStatus:
    def test_session_status_created_state(self, base_url, auth_headers):
        """A freshly created session with no chunks should be in 'created' state."""
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={"mode": "stream"},
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        sid = r.json()["session_id"]

        r = requests.get(
            f"{base_url}/api/v2/sessions/{sid}/status",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["state"] == "created"
        assert data["chunk_count"] == 0
        assert data["partial_transcript_available"] is False
        assert data["first_chunk_at"] is None
        assert data["finalize_requested_at"] is None

        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)

    def test_session_status_404_for_unknown(self, base_url, auth_headers):
        r = requests.get(
            f"{base_url}/api/v2/sessions/nonexistent-xyz/status",
            headers=auth_headers,
            timeout=5,
        )
        assert r.status_code == 404

    def test_session_status_timing_fields_present(self, base_url, auth_headers):
        """Timing fields should be present in the response."""
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]

        r = requests.get(
            f"{base_url}/api/v2/sessions/{sid}/status",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        for field in ("created_at", "first_chunk_at", "last_chunk_at",
                      "finalize_requested_at", "queued_at", "started_at", "finished_at"):
            assert field in data, f"Missing field: {field}"

        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)

    def test_session_status_chunk_fields(self, base_url, auth_headers):
        """After uploading chunks, chunk tracking fields should be accurate."""
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={"mode": "stream"},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]

        for i in range(2):
            wav_bytes = make_wav_bytes(duration_s=0.5)
            requests.post(
                f"{base_url}/api/v2/sessions/{sid}/chunks",
                headers=auth_headers,
                files={"file": (f"chunk_{i}.wav", wav_bytes, "audio/wav")},
                data={"chunk_index": str(i), "chunk_duration_ms": "500"},
                timeout=10,
            )

        r = requests.get(
            f"{base_url}/api/v2/sessions/{sid}/status",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["chunk_count"] == 2
        assert data["chunks_received"] == 2
        assert data["last_chunk_index"] == 1
        assert "last_processed_chunk_index" in data
        assert "partial_preview" in data
        assert "partial_updated_at" in data
        assert data["total_audio_ms"] >= 0

        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)


# ---------------------------------------------------------------------------
# Partial transcript endpoint
# ---------------------------------------------------------------------------
class TestPartialTranscript:
    def test_partial_404_when_no_partials_yet(self, base_url, auth_headers):
        """A freshly created session should return 404 for partial transcript."""
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={"mode": "stream"},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]

        r = requests.get(
            f"{base_url}/api/v2/sessions/{sid}/transcript/partial",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 404, f"Expected 404, got {r.status_code}"

        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)

    def test_partial_transcript_shape_when_available(self, base_url, auth_headers):
        """
        If partial transcript is available (worker processed it), check the shape.
        This test is skipped if the worker hasn't run yet.
        """
        # Create session and upload enough chunks to trigger a partial
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={"mode": "stream"},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]

        # Upload 5 chunks (default PARTIAL_EVERY_N_CHUNKS=5)
        for i in range(5):
            wav_bytes = make_wav_bytes(duration_s=1.0)
            requests.post(
                f"{base_url}/api/v2/sessions/{sid}/chunks",
                headers=auth_headers,
                files={"file": (f"chunk_{i}.wav", wav_bytes, "audio/wav")},
                data={"chunk_index": str(i), "chunk_duration_ms": "1000"},
                timeout=30,
            )

        # Poll session status for up to 30s for partial to appear
        deadline = time.time() + 30
        partial_available = False
        while time.time() < deadline:
            time.sleep(2)
            rs = requests.get(
                f"{base_url}/api/v2/sessions/{sid}/status",
                headers=auth_headers,
                timeout=10,
            )
            if rs.status_code == 200 and rs.json().get("partial_transcript_available"):
                partial_available = True
                break

        if not partial_available:
            pytest.skip("Partial transcript not generated within 30s (worker may not be running)")

        r = requests.get(
            f"{base_url}/api/v2/sessions/{sid}/transcript/partial",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        data = r.json()
        assert data["provisional"] is True
        assert "text" in data
        assert isinstance(data["segments"], list)
        assert "chunk_count_at_time" in data
        assert "generated_at" in data

        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)


# ---------------------------------------------------------------------------
# Field normalization (Android compat)
# ---------------------------------------------------------------------------
class TestFieldNormalization:
    def test_num_speakers_alias(self, base_url, auth_headers):
        """num_speakers should be accepted as an alias for speaker_count in finalize."""
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]

        wav_bytes = make_wav_bytes(0.5)
        requests.post(
            f"{base_url}/api/v2/sessions/{sid}/chunks",
            headers=auth_headers,
            files={"file": ("c.wav", wav_bytes, "audio/wav")},
            data={"chunk_index": "0"},
            timeout=10,
        )

        # Use num_speakers (Android field) instead of speaker_count
        r = requests.post(
            f"{base_url}/api/v2/sessions/{sid}/finalize",
            json={"language": "en", "model_size": "tiny", "num_speakers": 1},
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200, f"Finalize with num_speakers failed: {r.text}"
        assert "job_id" in r.json()

        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
class TestIdempotency:
    def test_finalize_idempotent(self, base_url, auth_headers):
        """Calling finalize twice should not error out."""
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]
        wav_bytes = make_wav_bytes(0.5)
        requests.post(
            f"{base_url}/api/v2/sessions/{sid}/chunks",
            headers=auth_headers,
            files={"file": ("c.wav", wav_bytes, "audio/wav")},
            data={"chunk_index": "0", "is_final": "true"},
            timeout=10,
        )
        payload = {"run_diarization": False, "language": "en", "model_size": "tiny"}
        r1 = requests.post(
            f"{base_url}/api/v2/sessions/{sid}/finalize",
            json=payload,
            headers=auth_headers,
            timeout=10,
        )
        r2 = requests.post(
            f"{base_url}/api/v2/sessions/{sid}/finalize",
            json=payload,
            headers=auth_headers,
            timeout=10,
        )
        assert r1.status_code == 200
        assert r2.status_code == 200

        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)


# ---------------------------------------------------------------------------
# Delete session
# ---------------------------------------------------------------------------
class TestDelete:
    def test_delete_session(self, base_url, auth_headers):
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]

        r = requests.delete(
            f"{base_url}/api/v2/sessions/{sid}",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        assert r.json()["deleted"] is True

        # Second delete → 404
        r2 = requests.delete(
            f"{base_url}/api/v2/sessions/{sid}",
            headers=auth_headers,
            timeout=10,
        )
        assert r2.status_code == 404


# ---------------------------------------------------------------------------
# Session rollover (continuous BLE — finalize → new session)
# ---------------------------------------------------------------------------
class TestSessionRollover:
    """
    Validate the session rollover pattern:
      1. Create session A, upload chunks, finalize
      2. Create session B immediately
      3. Upload chunks to session B
      4. Verify late chunks to session A are rejected with 409
      5. Verify A and B are completely independent (no chunk cross-contamination)
    """

    def test_rollover_creates_independent_sessions(self, base_url, auth_headers):
        wav_bytes = make_wav_bytes(0.5)

        # --- Session A ---
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={"device_id": "TEST:ROLLOVER:01", "mode": "stream"},
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200, f"Create session A failed: {r.text}"
        sid_a = r.json()["session_id"]

        requests.post(
            f"{base_url}/api/v2/sessions/{sid_a}/chunks",
            headers=auth_headers,
            files={"file": ("c0.wav", wav_bytes, "audio/wav")},
            data={"chunk_index": "0"},
            timeout=10,
        )

        r = requests.post(
            f"{base_url}/api/v2/sessions/{sid_a}/finalize",
            json={"language": "en", "model_size": "tiny"},
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200, f"Finalize A failed: {r.text}"

        # --- Session B (created immediately after A finalizes) ---
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={"device_id": "TEST:ROLLOVER:01", "mode": "stream"},
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200, f"Create session B failed: {r.text}"
        sid_b = r.json()["session_id"]
        assert sid_b != sid_a, "Session B must have a different ID than session A"

        # Upload to B — must succeed
        r = requests.post(
            f"{base_url}/api/v2/sessions/{sid_b}/chunks",
            headers=auth_headers,
            files={"file": ("c0.wav", wav_bytes, "audio/wav")},
            data={"chunk_index": "0"},
            timeout=10,
        )
        assert r.status_code == 200, f"Upload to session B failed: {r.text}"

        # B has exactly 1 chunk
        rs = requests.get(
            f"{base_url}/api/v2/sessions/{sid_b}/status",
            headers=auth_headers,
            timeout=10,
        )
        assert rs.json()["chunk_count"] == 1, "Session B should have exactly 1 chunk"

        # Cleanup
        requests.delete(f"{base_url}/api/v2/sessions/{sid_a}", headers=auth_headers, timeout=5)
        requests.delete(f"{base_url}/api/v2/sessions/{sid_b}", headers=auth_headers, timeout=5)

    def test_late_chunk_to_finalized_session_returns_409(self, base_url, auth_headers):
        """Uploading a chunk to an already-finalized session must return 409."""
        wav_bytes = make_wav_bytes(0.5)

        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]

        requests.post(
            f"{base_url}/api/v2/sessions/{sid}/chunks",
            headers=auth_headers,
            files={"file": ("c0.wav", wav_bytes, "audio/wav")},
            data={"chunk_index": "0"},
            timeout=10,
        )
        requests.post(
            f"{base_url}/api/v2/sessions/{sid}/finalize",
            json={"model_size": "tiny"},
            headers=auth_headers,
            timeout=10,
        )

        # Now try a late chunk
        r = requests.post(
            f"{base_url}/api/v2/sessions/{sid}/chunks",
            headers=auth_headers,
            files={"file": ("late.wav", wav_bytes, "audio/wav")},
            data={"chunk_index": "1"},
            timeout=10,
        )
        assert r.status_code == 409, \
            f"Expected 409 for late chunk to finalized session, got {r.status_code}: {r.text}"
        assert "finalized" in r.text.lower() or "new session" in r.text.lower(), \
            f"409 response should mention 'finalized' or 'new session': {r.text}"

        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)


# ---------------------------------------------------------------------------
# Transcript completeness
# ---------------------------------------------------------------------------
class TestTranscriptCompleteness:
    """
    Validate that the final transcript endpoint returns complete, ordered data.
    These tests skip if the job hasn't finished; run after TestSessionFlow completes.
    """

    @pytest.fixture(scope="class")
    def completed_session(self, base_url, auth_headers):
        """Create a session, upload chunks, finalize, wait for completion."""
        wav_bytes = make_wav_bytes(duration_s=2.0)

        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={"device_id": "TEST:COMPLETENESS:01"},
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        sid = r.json()["session_id"]

        for i in range(2):
            requests.post(
                f"{base_url}/api/v2/sessions/{sid}/chunks",
                headers=auth_headers,
                files={"file": (f"c{i}.wav", wav_bytes, "audio/wav")},
                data={"chunk_index": str(i), "chunk_duration_ms": "2000"},
                timeout=30,
            )

        r = requests.post(
            f"{base_url}/api/v2/sessions/{sid}/finalize",
            json={"language": "en", "model_size": "tiny", "run_diarization": False},
            headers=auth_headers,
            timeout=30,
        )
        assert r.status_code == 200

        # Poll for done (up to 120s)
        deadline = time.time() + 120
        state = "queued"
        while time.time() < deadline:
            time.sleep(3)
            rj = requests.get(
                f"{base_url}/api/v2/jobs/{sid}",
                headers=auth_headers,
                timeout=10,
            )
            state = rj.json().get("state", state)
            if state in ("done", "error"):
                break

        yield {"session_id": sid, "state": state}
        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)

    def test_transcript_text_is_string(self, base_url, auth_headers, completed_session):
        if completed_session["state"] != "done":
            pytest.skip(f"Job not done (state={completed_session['state']})")
        sid = completed_session["session_id"]
        r = requests.get(
            f"{base_url}/api/v2/sessions/{sid}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["text"], str), "text must be a string"

    def test_transcript_segments_sorted_by_start(self, base_url, auth_headers, completed_session):
        if completed_session["state"] != "done":
            pytest.skip(f"Job not done (state={completed_session['state']})")
        sid = completed_session["session_id"]
        r = requests.get(
            f"{base_url}/api/v2/sessions/{sid}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        segments = r.json().get("segments", [])
        starts = [s["start_ms"] for s in segments]
        assert starts == sorted(starts), f"Segments not sorted by start_ms: {starts}"

    def test_transcript_no_empty_segment_text(self, base_url, auth_headers, completed_session):
        if completed_session["state"] != "done":
            pytest.skip(f"Job not done (state={completed_session['state']})")
        sid = completed_session["session_id"]
        r = requests.get(
            f"{base_url}/api/v2/sessions/{sid}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        segments = r.json().get("segments", [])
        for seg in segments:
            assert seg.get("text", "").strip() != "", \
                f"Empty text segment found: {seg}"

    def test_transcript_text_not_empty_when_segments_exist(
        self, base_url, auth_headers, completed_session
    ):
        if completed_session["state"] != "done":
            pytest.skip(f"Job not done (state={completed_session['state']})")
        sid = completed_session["session_id"]
        r = requests.get(
            f"{base_url}/api/v2/sessions/{sid}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        segments = data.get("segments", [])
        # If whisper found any speech, top-level text must not be empty
        if segments:
            assert data["text"].strip() != "", \
                "top-level text is empty but segments were returned"

    def test_transcript_formats_present(self, base_url, auth_headers, completed_session):
        if completed_session["state"] != "done":
            pytest.skip(f"Job not done (state={completed_session['state']})")
        sid = completed_session["session_id"]
        r = requests.get(
            f"{base_url}/api/v2/sessions/{sid}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "formats" in data
        assert "srt_url" in data["formats"]
        assert "vtt_url" in data["formats"]


# ---------------------------------------------------------------------------
# Model size validation
# ---------------------------------------------------------------------------
class TestModelSizeValidation:
    def test_valid_model_sizes_accepted(self, base_url, auth_headers):
        """All allowed model sizes should not trigger a 400."""
        allowed = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
        for model in allowed:
            r = requests.post(
                f"{base_url}/api/v2/sessions",
                json={},
                headers=auth_headers,
                timeout=10,
            )
            sid = r.json()["session_id"]

            wav_bytes = make_wav_bytes(0.5)
            requests.post(
                f"{base_url}/api/v2/sessions/{sid}/chunks",
                headers=auth_headers,
                files={"file": ("c.wav", wav_bytes, "audio/wav")},
                data={"chunk_index": "0"},
                timeout=10,
            )

            r = requests.post(
                f"{base_url}/api/v2/sessions/{sid}/finalize",
                json={"model_size": model},
                headers=auth_headers,
                timeout=10,
            )
            assert r.status_code == 200, \
                f"model_size='{model}' should be accepted, got {r.status_code}: {r.text}"

            requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)

    def test_invalid_model_size_returns_400(self, base_url, auth_headers):
        """An unrecognized model_size must return 400."""
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]

        wav_bytes = make_wav_bytes(0.5)
        requests.post(
            f"{base_url}/api/v2/sessions/{sid}/chunks",
            headers=auth_headers,
            files={"file": ("c.wav", wav_bytes, "audio/wav")},
            data={"chunk_index": "0"},
            timeout=10,
        )

        r = requests.post(
            f"{base_url}/api/v2/sessions/{sid}/finalize",
            json={"model_size": "turbo-9000"},
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 400, \
            f"Invalid model_size should return 400, got {r.status_code}: {r.text}"
        assert "model_size" in r.text.lower() or "allowed" in r.text.lower(), \
            f"400 response should mention model_size or allowed values: {r.text}"

        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)

    def test_null_model_size_uses_default(self, base_url, auth_headers):
        """Omitting model_size or sending null should succeed (uses default 'base')."""
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]

        wav_bytes = make_wav_bytes(0.5)
        requests.post(
            f"{base_url}/api/v2/sessions/{sid}/chunks",
            headers=auth_headers,
            files={"file": ("c.wav", wav_bytes, "audio/wav")},
            data={"chunk_index": "0"},
            timeout=10,
        )

        r = requests.post(
            f"{base_url}/api/v2/sessions/{sid}/finalize",
            json={"model_size": None},
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200, \
            f"null model_size should succeed, got {r.status_code}: {r.text}"

        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_upload_to_nonexistent_session(self, base_url, auth_headers):
        wav_bytes = make_wav_bytes(0.1)
        r = requests.post(
            f"{base_url}/api/v2/sessions/nonexistent-session-id/chunks",
            headers=auth_headers,
            files={"file": ("c.wav", wav_bytes, "audio/wav")},
            data={"chunk_index": "0"},
            timeout=10,
        )
        assert r.status_code == 404

    def test_finalize_with_no_chunks(self, base_url, auth_headers):
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]

        r = requests.post(
            f"{base_url}/api/v2/sessions/{sid}/finalize",
            json={},
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 400

        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)


# ---------------------------------------------------------------------------
# Transcript quality report in final response
# ---------------------------------------------------------------------------
class TestTranscriptQualityReport:
    """
    Integration tests for the quality_report field added to
    GET /api/v2/sessions/{id}/transcript.

    Semantics being verified:
    - quality_report key is always present in the response (null until worker writes it)
    - reading_text key is always present (null until quality_report available)
    - When quality_report is populated, it has the expected structure
    - text (canonical) is never empty when segments exist
    - reading_text <= text (same or shorter, never adds content)
    - Partial transcript stays provisional=true and does NOT contain quality_report
    """

    @pytest.fixture(scope="class")
    def done_session(self, base_url, auth_headers):
        """Create + finalize a session and wait for done."""
        wav_bytes = make_wav_bytes(duration_s=2.0)

        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={"device_id": "TEST:QUALITY:01"},
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        sid = r.json()["session_id"]

        for i in range(2):
            requests.post(
                f"{base_url}/api/v2/sessions/{sid}/chunks",
                headers=auth_headers,
                files={"file": (f"c{i}.wav", wav_bytes, "audio/wav")},
                data={"chunk_index": str(i), "chunk_duration_ms": "2000"},
                timeout=30,
            )

        requests.post(
            f"{base_url}/api/v2/sessions/{sid}/finalize",
            json={"language": "en", "model_size": "tiny", "run_diarization": False},
            headers=auth_headers,
            timeout=30,
        )

        # Poll for done (up to 120s)
        deadline = time.time() + 120
        state = "queued"
        while time.time() < deadline:
            time.sleep(3)
            rj = requests.get(
                f"{base_url}/api/v2/jobs/{sid}",
                headers=auth_headers,
                timeout=10,
            )
            state = rj.json().get("state", state)
            if state in ("done", "error"):
                break

        yield {"session_id": sid, "state": state}
        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)

    def test_transcript_response_has_quality_report_key(
        self, base_url, auth_headers, done_session
    ):
        if done_session["state"] != "done":
            pytest.skip(f"Job not done (state={done_session['state']})")
        r = requests.get(
            f"{base_url}/api/v2/sessions/{done_session['session_id']}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "quality_report" in data, "quality_report key missing from transcript response"
        assert "reading_text" in data, "reading_text key missing from transcript response"

    def test_quality_report_structure_when_present(
        self, base_url, auth_headers, done_session
    ):
        if done_session["state"] != "done":
            pytest.skip(f"Job not done (state={done_session['state']})")
        r = requests.get(
            f"{base_url}/api/v2/sessions/{done_session['session_id']}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        data = r.json()
        qr = data.get("quality_report")
        if qr is None:
            pytest.skip("quality_report is null (worker may not have produced it yet)")

        for key in ("analysis_version", "segment_count", "flagged_segment_count",
                    "clean_segment_count", "issues", "suspected_missing_intervals",
                    "continuity", "reading_text", "analyzed_at"):
            assert key in qr, f"quality_report missing key: {key}"

        assert isinstance(qr["issues"], list)
        assert isinstance(qr["suspected_missing_intervals"], list)
        assert isinstance(qr["continuity"], dict)
        assert qr["analysis_version"] == "1.1"

    def test_canonical_text_present_and_not_empty_when_segments_exist(
        self, base_url, auth_headers, done_session
    ):
        if done_session["state"] != "done":
            pytest.skip(f"Job not done (state={done_session['state']})")
        r = requests.get(
            f"{base_url}/api/v2/sessions/{done_session['session_id']}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        data = r.json()
        segments = data.get("segments", [])
        if segments:
            assert data["text"].strip() != "", (
                "text is empty but segments were returned — text assembly is broken"
            )

    def test_reading_text_subset_of_text(
        self, base_url, auth_headers, done_session
    ):
        """reading_text should be equal to or shorter than text (never adds words)."""
        if done_session["state"] != "done":
            pytest.skip(f"Job not done (state={done_session['state']})")
        r = requests.get(
            f"{base_url}/api/v2/sessions/{done_session['session_id']}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        data = r.json()
        reading = data.get("reading_text") or ""
        full_text = data.get("text") or ""
        # reading_text must not contain words absent from text
        if reading and full_text:
            reading_words = set(reading.lower().split())
            full_words = set(full_text.lower().split())
            extra = reading_words - full_words
            assert not extra, (
                f"reading_text contains words not in text: {extra}"
            )

    def test_quality_report_issues_have_required_fields(
        self, base_url, auth_headers, done_session
    ):
        if done_session["state"] != "done":
            pytest.skip(f"Job not done (state={done_session['state']})")
        r = requests.get(
            f"{base_url}/api/v2/sessions/{done_session['session_id']}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        data = r.json()
        qr = data.get("quality_report")
        if not qr or not qr.get("issues"):
            pytest.skip("No issues in quality_report — nothing to validate")
        for issue in qr["issues"]:
            for key in ("type", "confidence", "segment_index", "start_ms", "end_ms",
                        "text", "reason"):
                assert key in issue, f"Issue entry missing key '{key}': {issue}"
            assert issue["type"] in (
                "segmentation_error", "missing_audio_continuity",
                "phonetic_hallucination", "syntax_collapse", "low_information_noise",
                "speaker_mixing",
            ), f"Unknown issue type: {issue['type']}"
            assert issue["confidence"] in ("low", "medium", "high")

    def test_partial_transcript_is_provisional_no_quality_report(
        self, base_url, auth_headers
    ):
        """
        Partial transcript must remain provisional=true and must NOT contain
        a quality_report field — quality analysis is final-only.
        """
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={"mode": "stream"},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]

        for i in range(5):
            wav_bytes = make_wav_bytes(duration_s=1.0)
            requests.post(
                f"{base_url}/api/v2/sessions/{sid}/chunks",
                headers=auth_headers,
                files={"file": (f"c{i}.wav", wav_bytes, "audio/wav")},
                data={"chunk_index": str(i), "chunk_duration_ms": "1000"},
                timeout=30,
            )

        # Poll up to 30s for partial
        deadline = time.time() + 30
        partial_available = False
        while time.time() < deadline:
            time.sleep(2)
            rs = requests.get(
                f"{base_url}/api/v2/sessions/{sid}/status",
                headers=auth_headers,
                timeout=10,
            )
            if rs.status_code == 200 and rs.json().get("partial_transcript_available"):
                partial_available = True
                break

        if not partial_available:
            requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)
            pytest.skip("Partial transcript not generated within 30s (worker may not be running)")

        r = requests.get(
            f"{base_url}/api/v2/sessions/{sid}/transcript/partial",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert data["provisional"] is True
        assert "quality_report" not in data, (
            "Partial transcript must NOT contain quality_report — it is final-only"
        )

        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)


# ---------------------------------------------------------------------------
# TestRawCleanTranscript
# ---------------------------------------------------------------------------

class TestRawCleanTranscript:
    """
    Verify that the transcript endpoint exposes the new raw/clean split fields
    while keeping backward-compat fields intact.
    Requires a fully-done session (done_session fixture).
    """

    def test_raw_text_present(self, base_url, auth_headers, done_session):
        if done_session["state"] != "done":
            pytest.skip(f"Job not done (state={done_session['state']})")
        r = requests.get(
            f"{base_url}/api/v2/sessions/{done_session['session_id']}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "raw_text" in data, "raw_text field missing from transcript response"

    def test_clean_text_key_present(self, base_url, auth_headers, done_session):
        """clean_text key must be present (may be null if worker hasn't finished yet)."""
        if done_session["state"] != "done":
            pytest.skip(f"Job not done (state={done_session['state']})")
        r = requests.get(
            f"{base_url}/api/v2/sessions/{done_session['session_id']}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "clean_text" in data, "clean_text field missing from transcript response"

    def test_paragraphs_key_present(self, base_url, auth_headers, done_session):
        """paragraphs key must be present (may be null if worker hasn't finished yet)."""
        if done_session["state"] != "done":
            pytest.skip(f"Job not done (state={done_session['state']})")
        r = requests.get(
            f"{base_url}/api/v2/sessions/{done_session['session_id']}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "paragraphs" in data, "paragraphs field missing from transcript response"

    def test_source_integrity_key_present(self, base_url, auth_headers, done_session):
        """source_integrity key must be present at top level of transcript response."""
        if done_session["state"] != "done":
            pytest.skip(f"Job not done (state={done_session['state']})")
        r = requests.get(
            f"{base_url}/api/v2/sessions/{done_session['session_id']}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "source_integrity" in data, "source_integrity missing from transcript response"

    def test_text_field_unchanged_backward_compat(self, base_url, auth_headers, done_session):
        """Legacy `text` field must still be present and equal to raw_text."""
        if done_session["state"] != "done":
            pytest.skip(f"Job not done (state={done_session['state']})")
        r = requests.get(
            f"{base_url}/api/v2/sessions/{done_session['session_id']}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert "text" in data, "Backward-compat `text` field missing"
        # raw_text must equal text
        assert data.get("raw_text") == data.get("text"), (
            "raw_text and text must be identical for backward compatibility"
        )

    def test_segments_have_corruption_flags_field(self, base_url, auth_headers, done_session):
        """Every segment in the transcript response must have a corruption_flags list."""
        if done_session["state"] != "done":
            pytest.skip(f"Job not done (state={done_session['state']})")
        r = requests.get(
            f"{base_url}/api/v2/sessions/{done_session['session_id']}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        segments = data.get("segments", [])
        if not segments:
            pytest.skip("No segments returned — cannot verify corruption_flags field")
        for i, seg in enumerate(segments):
            assert "corruption_flags" in seg, (
                f"segments[{i}] missing corruption_flags field"
            )
            assert isinstance(seg["corruption_flags"], list), (
                f"segments[{i}].corruption_flags must be a list"
            )

    def test_paragraphs_structure_when_populated(self, base_url, auth_headers, done_session):
        """If paragraphs is non-null and non-empty, each entry must have text and segment_indices."""
        if done_session["state"] != "done":
            pytest.skip(f"Job not done (state={done_session['state']})")
        r = requests.get(
            f"{base_url}/api/v2/sessions/{done_session['session_id']}/transcript",
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        paragraphs = data.get("paragraphs")
        if not paragraphs:
            pytest.skip("paragraphs is null or empty — cannot verify structure")
        for i, para in enumerate(paragraphs):
            assert "text" in para, f"paragraphs[{i}] missing text"
            assert "segment_indices" in para, f"paragraphs[{i}] missing segment_indices"
            assert isinstance(para["segment_indices"], list), (
                f"paragraphs[{i}].segment_indices must be a list"
            )


# ---------------------------------------------------------------------------
# TestContinuityMetadata
# ---------------------------------------------------------------------------

class TestContinuityMetadata:
    """
    Verify that the API accepts optional continuity metadata fields on chunk
    upload and session_integrity on finalize without rejecting the request.
    These fields are additive — existing clients that omit them are unaffected.
    """

    def _create_session(self, base_url, auth_headers):
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={"label": "continuity-meta-test"},
            headers=auth_headers,
            timeout=10,
        )
        assert r.status_code == 200
        return r.json()["session_id"]

    def _make_wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 160)  # 10ms silence
        return buf.getvalue()

    def test_chunk_upload_with_continuity_fields_accepted(self, base_url, auth_headers):
        """Chunk upload with all continuity form fields must return 200."""
        sid = self._create_session(base_url, auth_headers)
        try:
            wav = self._make_wav_bytes()
            r = requests.post(
                f"{base_url}/api/v2/sessions/{sid}/chunks",
                headers=auth_headers,
                files={"file": ("chunk.wav", wav, "audio/wav")},
                data={
                    "chunk_index": "0",
                    "dropped_frames": "3",
                    "decode_failure": "false",
                    "gap_before_ms": "250",
                    "source_degraded": "false",
                },
                timeout=15,
            )
            assert r.status_code == 200, (
                f"Chunk upload with continuity fields rejected: {r.status_code} {r.text}"
            )
        finally:
            requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)

    def test_chunk_upload_without_continuity_fields_still_works(self, base_url, auth_headers):
        """Backward-compat: chunk upload omitting all continuity fields must still return 200."""
        sid = self._create_session(base_url, auth_headers)
        try:
            wav = self._make_wav_bytes()
            r = requests.post(
                f"{base_url}/api/v2/sessions/{sid}/chunks",
                headers=auth_headers,
                files={"file": ("chunk.wav", wav, "audio/wav")},
                data={"chunk_index": "0"},
                timeout=15,
            )
            assert r.status_code == 200, (
                f"Chunk upload without continuity fields rejected: {r.status_code} {r.text}"
            )
        finally:
            requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)

    def test_chunk_upload_decode_failure_true(self, base_url, auth_headers):
        """decode_failure=true must be accepted without error."""
        sid = self._create_session(base_url, auth_headers)
        try:
            wav = self._make_wav_bytes()
            r = requests.post(
                f"{base_url}/api/v2/sessions/{sid}/chunks",
                headers=auth_headers,
                files={"file": ("chunk.wav", wav, "audio/wav")},
                data={
                    "chunk_index": "0",
                    "decode_failure": "true",
                    "source_degraded": "true",
                    "dropped_frames": "12",
                    "gap_before_ms": "1500",
                },
                timeout=15,
            )
            assert r.status_code == 200, (
                f"decode_failure=true chunk upload rejected: {r.status_code} {r.text}"
            )
        finally:
            requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)

    def test_finalize_with_session_integrity_accepted(self, base_url, auth_headers):
        """Finalize body with session_integrity dict must return 200."""
        sid = self._create_session(base_url, auth_headers)
        try:
            wav = self._make_wav_bytes()
            requests.post(
                f"{base_url}/api/v2/sessions/{sid}/chunks",
                headers=auth_headers,
                files={"file": ("chunk.wav", wav, "audio/wav")},
                data={"chunk_index": "0"},
                timeout=15,
            )
            r = requests.post(
                f"{base_url}/api/v2/sessions/{sid}/finalize",
                json={
                    "session_integrity": {
                        "total_dropped_frames": 5,
                        "any_decode_failure": False,
                        "recording_interrupted": False,
                        "note": "test run",
                    }
                },
                headers=auth_headers,
                timeout=10,
            )
            assert r.status_code == 200, (
                f"Finalize with session_integrity rejected: {r.status_code} {r.text}"
            )
        finally:
            requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)

    def test_finalize_without_session_integrity_still_works(self, base_url, auth_headers):
        """Backward-compat: finalize without session_integrity must still return 200."""
        sid = self._create_session(base_url, auth_headers)
        try:
            wav = self._make_wav_bytes()
            requests.post(
                f"{base_url}/api/v2/sessions/{sid}/chunks",
                headers=auth_headers,
                files={"file": ("chunk.wav", wav, "audio/wav")},
                data={"chunk_index": "0"},
                timeout=15,
            )
            r = requests.post(
                f"{base_url}/api/v2/sessions/{sid}/finalize",
                json={},
                headers=auth_headers,
                timeout=10,
            )
            assert r.status_code == 200, (
                f"Finalize without session_integrity rejected: {r.status_code} {r.text}"
            )
        finally:
            requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)
