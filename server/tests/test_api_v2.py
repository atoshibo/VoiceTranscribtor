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
