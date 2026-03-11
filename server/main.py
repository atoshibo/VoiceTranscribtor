"""
VoiceRecordTranscriptor — web service.

Serves API v2 (Android/Omi client), diagnostics, and a minimal browser debug UI.
API auth: Authorization: Bearer / X-Api-Token headers only.
Browser UI: token stored in localStorage, never in URL.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
import os
import json
import time
from pathlib import Path

import redis as _redis_lib

from api_v2 import router as api_v2_router

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AUTH_TOKEN = os.getenv("AUTH_TOKEN")
SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", "/data/sessions"))
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Paths that never require auth (public)
_PUBLIC_PATHS = {"/", "/health", "/api/health", "/api/v2/health"}

app = FastAPI(title="VoiceRecordTranscriptor", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth middleware — API only (browser UI handles its own auth via JS)
# ---------------------------------------------------------------------------
def _is_authorized(request: StarletteRequest) -> bool:
    if not AUTH_TOKEN:
        return False
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and auth[7:] == AUTH_TOKEN:
        return True
    if request.headers.get("X-Api-Token", "") == AUTH_TOKEN:
        return True
    return False


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        path = request.url.path
        # Public: root UI page and health checks
        if path in _PUBLIC_PATHS or path.startswith("/api/v2/health"):
            return await call_next(request)
        # API calls require header auth
        if path.startswith("/api/"):
            if not _is_authorized(request):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required. Use 'Authorization: Bearer <token>' header."},
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)


app.add_middleware(AuthMiddleware)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        r = _redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.ping()
        print(f"[STARTUP] Redis OK at {REDIS_HOST}:{REDIS_PORT}")
    except Exception as e:
        print(f"[STARTUP] Redis not available: {e}")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
@app.get("/api/health")
async def health():
    return {"ok": True, "service": "web"}


# ---------------------------------------------------------------------------
# Sessions list (for browser UI)
# ---------------------------------------------------------------------------
@app.get("/api/sessions")
async def list_sessions():
    """List recent sessions, newest first. Max 100."""
    sessions = []
    if not SESSIONS_DIR.exists():
        return {"sessions": []}
    for d in sorted(SESSIONS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[:100]:
        if not d.is_dir():
            continue
        entry = {"session_id": d.name, "is_v2": (d / "v2_session.json").exists()}
        # Read status
        sp = d / "status.json"
        if sp.exists():
            try:
                s = json.loads(sp.read_text(encoding="utf-8"))
                entry["status"] = s.get("status", "unknown")
                entry["updated_at"] = s.get("updated_at")
                entry["progress"] = s.get("progress", {})
            except Exception:
                entry["status"] = "unreadable"
        # Read creation time from v2_session.json or mtime
        vsp = d / "v2_session.json"
        if vsp.exists():
            try:
                v = json.loads(vsp.read_text(encoding="utf-8"))
                entry["created_at"] = v.get("created_at")
                entry["device_id"] = v.get("device_id")
                entry["mode"] = v.get("mode")
            except Exception:
                pass
        if "created_at" not in entry:
            entry["created_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.localtime(d.stat().st_mtime)
            )
        sessions.append(entry)
    return {"sessions": sessions}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@app.get("/api/diagnostics")
async def diagnostics():
    result = {"redis": {"ok": False, "error": None}, "sessions_dir": str(SESSIONS_DIR), "gpu": {}}
    try:
        r = _redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT)
        r.ping()
        result["redis"]["ok"] = True
        result["redis"]["queue_len"] = r.llen(os.getenv("REDIS_QUEUE", "transcription_jobs"))
    except Exception as e:
        result["redis"]["error"] = str(e)
    try:
        from transcription import get_device_diagnostics  # type: ignore
        result["gpu"] = get_device_diagnostics()
    except ImportError:
        result["gpu"] = {"note": "transcription module not in web container"}
    except Exception as e:
        result["gpu"] = {"error": str(e)}
    return result


@app.get("/api/selftest")
async def selftest():
    try:
        from transcription import run_smoke_test  # type: ignore
        result = run_smoke_test(model_size="tiny")
        return {"ok": result.get("success", False), "details": result}
    except ImportError:
        return {"ok": False, "details": "transcription module not in web container"}
    except Exception as e:
        return {"ok": False, "details": str(e)}


# ---------------------------------------------------------------------------
# Browser debug UI — single-page app, token in localStorage
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(_UI_HTML)


_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VoiceTranscript</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; font-size: 14px; background: #0f1117; color: #e0e0e0; min-height: 100vh; }
a { color: #7eb6ff; }
.page { max-width: 960px; margin: 0 auto; padding: 24px 16px; }
h1 { font-size: 18px; font-weight: 600; color: #fff; }

/* token gate */
#gate { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 60vh; gap: 12px; }
#gate h2 { font-size: 16px; color: #aaa; }
#gate input { width: 320px; padding: 10px 14px; border-radius: 6px; border: 1px solid #333; background: #1a1d27; color: #fff; font-size: 14px; outline: none; }
#gate input:focus { border-color: #4a7eff; }
#gate .btn-primary { padding: 10px 28px; border-radius: 6px; border: none; background: #4a7eff; color: #fff; cursor: pointer; font-size: 14px; }
#gate .btn-primary:hover { background: #3a6eff; }

/* top bar */
#topbar { display: flex; align-items: center; gap: 10px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid #222; flex-wrap: wrap; }
#topbar h1 { flex: 1; }
.pill { padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; white-space: nowrap; }
.pill.ok { background: #1a3a1a; color: #4caf50; }
.pill.err { background: #3a1a1a; color: #f44336; }
.pill.unknown { background: #2a2a2a; color: #888; }

/* buttons */
.btn { padding: 6px 14px; border-radius: 5px; border: 1px solid #333; background: #1e2130; color: #ccc; cursor: pointer; font-size: 13px; white-space: nowrap; }
.btn:hover { background: #2a2f45; color: #fff; }
.btn.primary { background: #1a3a6a; border-color: #2a5aaa; color: #7eb6ff; }
.btn.primary:hover { background: #1e4a8a; }
.btn.danger { border-color: #552222; color: #f88; }
.btn.danger:hover { background: #3a1a1a; }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }

/* two-column layout */
.layout { display: grid; grid-template-columns: 340px 1fr; gap: 20px; align-items: start; }
@media (max-width: 700px) { .layout { grid-template-columns: 1fr; } }

/* upload card */
.card { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 8px; padding: 16px; }
.card h2 { font-size: 13px; font-weight: 600; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 14px; }
.form-row { margin-bottom: 12px; }
.form-row label { display: block; font-size: 12px; color: #777; margin-bottom: 4px; }
.form-row select, .form-row input[type=number] {
  width: 100%; padding: 7px 10px; border-radius: 5px; border: 1px solid #333;
  background: #0f1117; color: #ddd; font-size: 13px; outline: none;
}
.form-row select:focus, .form-row input:focus { border-color: #4a7eff; }
.drop-zone {
  border: 2px dashed #333; border-radius: 6px; padding: 24px 12px;
  text-align: center; color: #555; cursor: pointer; transition: border-color .15s, color .15s;
  margin-bottom: 12px;
}
.drop-zone.hover, .drop-zone.has-file { border-color: #4a7eff; color: #7eb6ff; }
.drop-zone.has-file { background: #0d1520; }
#upload-progress { margin-top: 10px; font-size: 12px; color: #888; min-height: 18px; }
#upload-progress .prog-bar { height: 4px; background: #222; border-radius: 2px; margin-top: 6px; }
#upload-progress .prog-fill { height: 100%; background: #4a7eff; border-radius: 2px; transition: width .3s; }

/* sessions panel */
.toolbar { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
.toolbar input { flex: 1; min-width: 160px; padding: 7px 12px; border-radius: 5px; border: 1px solid #333; background: #1a1d27; color: #fff; font-size: 13px; outline: none; }
.toolbar input:focus { border-color: #4a7eff; }

table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 7px 10px; border-bottom: 1px solid #222; color: #555; font-weight: 500; font-size: 12px; }
td { padding: 7px 10px; border-bottom: 1px solid #181b24; vertical-align: middle; }
.status-done { color: #4caf50; }
.status-running, .status-queued { color: #ffa726; }
.status-error { color: #f44336; }
.clickable { cursor: pointer; color: #7eb6ff; font-family: monospace; font-size: 12px; }
.clickable:hover { text-decoration: underline; }

/* detail panel */
#detail { margin-top: 16px; background: #1a1d27; border-radius: 8px; border: 1px solid #2a2d3a; padding: 16px; display: none; }
#detail h3 { font-size: 13px; font-weight: 600; color: #aaa; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }
pre { background: #0f1117; border-radius: 5px; padding: 12px; font-size: 12px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; color: #ccc; border: 1px solid #222; max-height: 360px; overflow-y: auto; }
.section-title { font-size: 11px; color: #555; margin: 14px 0 4px; text-transform: uppercase; letter-spacing: 0.06em; }
.actions { display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap; }
.transcript-text { background: #0f1117; border-radius: 5px; padding: 12px; font-size: 13px; line-height: 1.7; color: #ddd; border: 1px solid #222; max-height: 280px; overflow-y: auto; }

/* message bar */
#msg { padding: 8px 12px; border-radius: 5px; font-size: 13px; margin-bottom: 12px; display: none; }
#msg.info { background: #1a2a3a; color: #7eb6ff; border: 1px solid #2a3a4a; }
#msg.err { background: #2a1a1a; color: #f88; border: 1px solid #4a2a2a; }
#msg.ok { background: #1a2a1a; color: #7ecf7e; border: 1px solid #2a4a2a; }
</style>
</head>
<body>
<div class="page">

<!-- TOKEN GATE -->
<div id="gate">
  <h2>VoiceTranscript</h2>
  <p style="color:#555;font-size:12px;margin-bottom:4px">Enter your API token to continue</p>
  <input type="password" id="token-input" placeholder="Paste token here..." autocomplete="off">
  <button class="btn-primary" onclick="saveToken()">Connect</button>
</div>

<!-- DASHBOARD -->
<div id="dashboard" style="display:none">
  <div id="topbar">
    <h1>VoiceTranscript</h1>
    <span id="gpu-pill" class="pill unknown">GPU …</span>
    <span id="redis-pill" class="pill unknown">Redis …</span>
    <button class="btn" onclick="refreshAll()">↻ Refresh</button>
    <button class="btn danger" onclick="clearToken()">Sign out</button>
  </div>

  <div id="msg"></div>

  <div class="layout">

    <!-- LEFT: upload + options -->
    <div>
      <div class="card">
        <h2>New Transcription</h2>

        <div class="drop-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
          <div id="drop-label">Click or drag an audio file here</div>
          <div style="font-size:11px;color:#444;margin-top:4px">WAV, MP3, M4A, OGG, FLAC, …</div>
        </div>
        <input type="file" id="file-input" accept="audio/*,video/*" style="display:none" onchange="onFileSelected(this)">

        <div class="form-row">
          <label>Language</label>
          <select id="opt-lang">
            <option value="">Auto-detect</option>
            <option value="en">English</option>
            <option value="ru">Russian</option>
            <option value="de">German</option>
            <option value="fr">French</option>
            <option value="es">Spanish</option>
            <option value="zh">Chinese</option>
            <option value="ja">Japanese</option>
            <option value="ar">Arabic</option>
            <option value="pt">Portuguese</option>
            <option value="it">Italian</option>
            <option value="ko">Korean</option>
            <option value="tr">Turkish</option>
            <option value="pl">Polish</option>
            <option value="uk">Ukrainian</option>
          </select>
        </div>

        <div class="form-row">
          <label>Whisper model</label>
          <select id="opt-model">
            <option value="tiny">tiny — fastest, least accurate</option>
            <option value="base">base</option>
            <option value="small" selected>small — recommended</option>
            <option value="medium">medium — slower, more accurate</option>
            <option value="large-v2">large-v2 — best quality</option>
            <option value="large-v3">large-v3</option>
          </select>
        </div>

        <div class="form-row">
          <label>Speakers (diarization)</label>
          <select id="opt-speakers">
            <option value="0">Auto-detect</option>
            <option value="1">1 speaker</option>
            <option value="2" selected>2 speakers</option>
            <option value="3">3 speakers</option>
            <option value="4">4 speakers</option>
            <option value="5">5 speakers</option>
            <option value="6">6+ speakers</option>
          </select>
        </div>

        <div id="upload-progress"></div>
        <button class="btn primary" id="submit-btn" onclick="submitUpload()" style="width:100%;margin-top:4px" disabled>
          Upload &amp; Transcribe
        </button>
      </div>
    </div>

    <!-- RIGHT: sessions list + detail -->
    <div>
      <div class="toolbar">
        <input id="lookup-id" placeholder="Session / job ID…" onkeydown="if(event.key==='Enter')lookupSession()">
        <button class="btn" onclick="lookupSession()">Look up</button>
        <button class="btn" onclick="loadSessions()">↻ Sessions</button>
      </div>

      <table id="sessions-table">
        <thead><tr>
          <th>Session ID</th>
          <th>Status</th>
          <th>Model</th>
          <th>Created</th>
        </tr></thead>
        <tbody id="sessions-body"></tbody>
      </table>

      <div id="detail">
        <h3 id="detail-title">Session detail</h3>
        <div id="detail-body"></div>
      </div>
    </div>

  </div><!-- .layout -->
</div><!-- #dashboard -->

</div><!-- .page -->

<script>
const BASE = '';
let _selectedFile = null;
let _pollTimer = null;

// ── Auth ──────────────────────────────────────────────────────────────
function getToken() { return localStorage.getItem('vts_token') || ''; }

function saveToken() {
  const t = document.getElementById('token-input').value.trim();
  if (!t) return;
  localStorage.setItem('vts_token', t);
  showDashboard();
}

function clearToken() {
  localStorage.removeItem('vts_token');
  document.getElementById('dashboard').style.display = 'none';
  document.getElementById('gate').style.display = 'flex';
  document.getElementById('token-input').value = '';
}

function showDashboard() {
  document.getElementById('gate').style.display = 'none';
  document.getElementById('dashboard').style.display = 'block';
  refreshAll();
}

// ── HTTP helpers ──────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const headers = { 'Authorization': 'Bearer ' + getToken(), ...(opts.headers || {}) };
  if (!(opts.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }
  const res = await fetch(BASE + path, { ...opts, headers });
  if (res.status === 401 || res.status === 403) {
    showMsg('Token rejected — check your token.', 'err');
    return null;
  }
  return res;
}

function showMsg(text, type = 'info') {
  const el = document.getElementById('msg');
  el.textContent = text;
  el.className = type;
  el.style.display = 'block';
  if (type !== 'err') setTimeout(() => { el.style.display = 'none'; }, 7000);
}

// ── Health / status bar ───────────────────────────────────────────────
async function refreshAll() { loadHealth(); loadSessions(); }

async function loadHealth() {
  try {
    const res = await fetch(BASE + '/api/v2/health');
    const d = await res.json();
    const gp = document.getElementById('gpu-pill');
    gp.textContent = d.gpu_available ? ('GPU ✓ ' + d.selected_compute_type) : 'GPU ✗';
    gp.className = 'pill ' + (d.gpu_available ? 'ok' : 'err');

    const dr = await apiFetch('/api/diagnostics');
    if (dr) {
      const dd = await dr.json();
      const rp = document.getElementById('redis-pill');
      const ok = dd.redis && dd.redis.ok;
      rp.textContent = ok ? ('Redis ✓ q:' + (dd.redis.queue_len ?? '?')) : 'Redis ✗';
      rp.className = 'pill ' + (ok ? 'ok' : 'err');
    }
  } catch(e) { console.error('health', e); }
}

// ── Sessions list ─────────────────────────────────────────────────────
async function loadSessions() {
  const res = await apiFetch('/api/sessions');
  if (!res) return;
  const d = await res.json();
  const tbody = document.getElementById('sessions-body');
  tbody.innerHTML = '';
  for (const s of (d.sessions || [])) {
    const cls = {done:'status-done',error:'status-error',running:'status-running',queued:'status-queued'}[s.status] || '';
    const tr = document.createElement('tr');
    const model = s.progress && s.progress.model ? s.progress.model : (s.mode || '—');
    tr.innerHTML = `
      <td><span class="clickable" onclick="openSession('${s.session_id}')" title="${s.session_id}">${s.session_id.slice(0,8)}…</span></td>
      <td class="${cls}">${s.status || '—'}</td>
      <td style="color:#666;font-size:12px">${model}</td>
      <td style="color:#555;font-size:12px">${(s.created_at||'').replace('T',' ').slice(0,16)}</td>`;
    tbody.appendChild(tr);
  }
  if (!d.sessions || !d.sessions.length) {
    tbody.innerHTML = '<tr><td colspan="4" style="color:#444;text-align:center;padding:24px">No sessions yet</td></tr>';
  }
}

// ── Drag & drop / file select ─────────────────────────────────────────
function onFileSelected(input) {
  if (input.files && input.files[0]) setFile(input.files[0]);
}

function setFile(file) {
  _selectedFile = file;
  const dz = document.getElementById('drop-zone');
  dz.classList.add('has-file');
  document.getElementById('drop-label').textContent = file.name + ' (' + (file.size/1048576).toFixed(1) + ' MB)';
  document.getElementById('submit-btn').disabled = false;
}

(function initDrop() {
  const dz = document.getElementById('drop-zone');
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('hover'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('hover'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('hover');
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });
})();

// ── Upload & transcribe ───────────────────────────────────────────────
async function submitUpload() {
  if (!_selectedFile) return;
  const lang = document.getElementById('opt-lang').value;
  const model = document.getElementById('opt-model').value;
  const numSpk = parseInt(document.getElementById('opt-speakers').value, 10);

  const btn = document.getElementById('submit-btn');
  btn.disabled = true;
  setProgress('Creating session…', 0);

  try {
    // 1. Create session
    const sr = await apiFetch('/api/v2/sessions', {
      method: 'POST',
      body: JSON.stringify({ mode: 'upload' }),
    });
    if (!sr || !sr.ok) { setProgress('Failed to create session.'); btn.disabled = false; return; }
    const { session_id } = await sr.json();

    // 2. Upload file as single chunk
    setProgress('Uploading…', 10);
    const form = new FormData();
    form.append('file', _selectedFile);
    form.append('chunk_index', '0');
    form.append('is_final', 'true');
    const ur = await apiFetch(`/api/v2/sessions/${session_id}/chunks`, {
      method: 'POST',
      body: form,
      headers: {},  // let browser set multipart boundary
    });
    if (!ur || !ur.ok) {
      const err = ur ? await ur.text() : 'upload failed';
      setProgress('Upload failed: ' + err);
      btn.disabled = false; return;
    }

    // 3. Finalize
    setProgress('Queuing job…', 60);
    const body = { model_size: model };
    if (lang) body.language = lang;
    if (numSpk > 0) body.num_speakers = numSpk;
    const fr = await apiFetch(`/api/v2/sessions/${session_id}/finalize`, {
      method: 'POST',
      body: JSON.stringify(body),
    });
    if (!fr || !fr.ok) { setProgress('Finalize failed.'); btn.disabled = false; return; }
    const { job_id } = await fr.json();

    setProgress('Job queued — polling…', 80);
    showMsg('Job started: ' + job_id, 'ok');
    loadSessions();

    // 4. Poll until done / error
    pollJob(job_id, session_id, btn);

  } catch(e) {
    setProgress('Error: ' + e.message);
    btn.disabled = false;
  }
}

function setProgress(text, pct) {
  const el = document.getElementById('upload-progress');
  if (pct !== undefined) {
    el.innerHTML = `<span>${text}</span><div class="prog-bar"><div class="prog-fill" style="width:${pct}%"></div></div>`;
  } else {
    el.textContent = text;
  }
}

function pollJob(jobId, sessionId, btn) {
  clearTimeout(_pollTimer);
  async function tick() {
    const res = await apiFetch(`/api/v2/jobs/${jobId}`);
    if (!res) return;
    const d = await res.json();
    const pct = d.progress ? Math.round((d.progress.processing || 0)) : null;
    const state = d.state || 'unknown';
    if (state === 'done') {
      setProgress('Done!', 100);
      btn.disabled = false;
      loadSessions();
      openSession(sessionId);
    } else if (state === 'error') {
      setProgress('Worker error — see session detail.');
      btn.disabled = false;
      loadSessions();
      openSession(sessionId);
    } else {
      setProgress(state + (pct !== null ? ' ' + pct + '%' : '…'), pct || 85);
      _pollTimer = setTimeout(tick, 2500);
    }
  }
  _pollTimer = setTimeout(tick, 2000);
}

// ── Session detail ────────────────────────────────────────────────────
function lookupSession() {
  const id = document.getElementById('lookup-id').value.trim();
  if (id) openSession(id);
}

async function openSession(id) {
  document.getElementById('lookup-id').value = id;
  const detail = document.getElementById('detail');
  detail.style.display = 'block';
  document.getElementById('detail-title').textContent = 'Session: ' + id;
  const body = document.getElementById('detail-body');
  body.innerHTML = '<p style="color:#555;font-size:12px">Loading…</p>';
  detail.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  const jobRes = await apiFetch(`/api/v2/jobs/${id}`);
  let jobData = jobRes && jobRes.ok ? await jobRes.json() : null;

  let html = '';

  if (jobData) {
    html += `<div class="section-title">Job Status</div>`;
    html += `<pre>${escHtml(JSON.stringify(jobData, null, 2))}</pre>`;
  } else {
    html += `<p style="color:#555;font-size:12px">No job data found.</p>`;
  }

  if (jobData && jobData.state === 'done') {
    const txRes = await apiFetch(`/api/v2/sessions/${id}/transcript`);
    if (txRes && txRes.ok) {
      const tx = await txRes.json();
      html += `<div class="section-title">Transcript</div>`;
      html += `<div class="transcript-text">${escHtml(tx.text || '(empty)')}</div>`;
      if (tx.segments && tx.segments.length) {
        html += `<div class="section-title">Segments (${tx.segments.length})</div><pre>`;
        html += tx.segments.slice(0, 40).map(s =>
          '[' + (s.start_ms/1000).toFixed(2) + 's] ' + s.speaker + ': ' + s.text
        ).join('\\n');
        if (tx.segments.length > 40) html += '\\n… (' + (tx.segments.length - 40) + ' more)';
        html += '</pre>';
      }
    }
  }

  if (jobData && jobData.state === 'error') {
    const errRes = await apiFetch(`/api/v2/jobs/${id}/error`);
    if (errRes && errRes.ok) {
      const err = await errRes.json();
      html += `<div class="section-title">Error Detail</div>`;
      html += `<pre style="color:#f88">${escHtml(JSON.stringify(err, null, 2))}</pre>`;
    }
  }

  html += `<div class="actions">
    <button class="btn" onclick="openSession('${id}')">↻ Refresh</button>`;
  if (jobData && jobData.state === 'done') {
    html += `<button class="btn" onclick="downloadSubtitle('${id}','srt')">↓ SRT</button>`;
    html += `<button class="btn" onclick="downloadSubtitle('${id}','vtt')">↓ VTT</button>`;
  }
  html += `<button class="btn danger" onclick="deleteSession('${id}')">Delete</button>`;
  html += `</div>`;

  body.innerHTML = html;
}

async function deleteSession(id) {
  if (!confirm('Delete session ' + id + '? This cannot be undone.')) return;
  const res = await apiFetch(`/api/v2/sessions/${id}`, { method: 'DELETE' });
  if (res && res.ok) {
    showMsg('Session deleted.', 'ok');
    document.getElementById('detail').style.display = 'none';
    loadSessions();
  } else {
    showMsg('Delete failed.', 'err');
  }
}

async function downloadSubtitle(id, fmt) {
  const res = await apiFetch(`/api/v2/sessions/${id}/subtitle.${fmt}`);
  if (!res || !res.ok) { showMsg('Failed to download ' + fmt.toUpperCase(), 'err'); return; }
  const text = await res.text();
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = id + '.' + fmt; a.click();
  URL.revokeObjectURL(url);
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Boot ──────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('token-input').addEventListener('keydown', e => {
    if (e.key === 'Enter') saveToken();
  });
  if (getToken()) showDashboard();
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Mount API v2
# ---------------------------------------------------------------------------
app.include_router(api_v2_router)
