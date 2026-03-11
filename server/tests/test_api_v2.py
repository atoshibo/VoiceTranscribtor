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
        assert 0.0 <= data["progress"] <= 1.0

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

        # It's OK if we hit timeout (worker may not be running in test env)
        # but state should be one of the valid values
        assert state in ("queued", "running", "done", "error"), f"Unexpected state: {state}"

    def test_get_transcript_when_done(self, base_url, auth_headers, session_id):
        """If job is done, transcript should have expected shape."""
        # Check current state
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
        # Create a new session but don't finalize it
        r = requests.post(
            f"{base_url}/api/v2/sessions",
            json={},
            headers=auth_headers,
            timeout=10,
        )
        sid = r.json()["session_id"]
        # Upload one chunk so state is valid
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
        # Not done yet → 202
        assert r.status_code in (202, 404), f"Expected 202 or 404, got {r.status_code}"

        # Cleanup
        requests.delete(
            f"{base_url}/api/v2/sessions/{sid}",
            headers=auth_headers,
            timeout=5,
        )


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

        # Cleanup
        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)


# ---------------------------------------------------------------------------
# Delete session
# ---------------------------------------------------------------------------
class TestDelete:
    def test_delete_session(self, base_url, auth_headers):
        # Create and immediately delete
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

        # Cleanup
        requests.delete(f"{base_url}/api/v2/sessions/{sid}", headers=auth_headers, timeout=5)
