"""Web UI sidecar — watch the loop and tune it live, from a browser.

This is the websocket overlay the spec calls for (§7: a sidecar over a
websocket, not the core). `WebMonitor` implements the same `Monitor` protocol as
the Rich dashboard, so the loop couples to neither — it just emits `TickReport`s.
On top of the read stream this adds a write channel: the browser sends `set`
messages that mutate the shared `ControlState`, which the cognition loop applies
on its next tick.

Served as a single self-contained HTML page (vanilla JS, no build step) plus a
`/ws` endpoint. `aiohttp` is imported lazily so it stays an optional extra:
    pip install -e ".[web]"
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import TYPE_CHECKING, Any

from .logging import get_logger
from .monitor import TickReport, format_clock, reason_kind

if TYPE_CHECKING:
    from aiohttp import web

    from .control import ControlState

log = get_logger(__name__)


class WebMonitor:
    """Monitor that broadcasts ticks over websockets and accepts live param edits."""

    def __init__(
        self,
        controls: ControlState,
        persona_name: str,
        platform: str,
        *,
        stream_info: dict[str, object] | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        history: int = 60,
    ) -> None:
        self.controls = controls
        self.persona_name = persona_name
        self.platform = platform
        self.stream_info = stream_info or {}
        self.host = host
        self.port = port
        self._clients: set[web.WebSocketResponse] = set()
        self._messages: deque[dict[str, Any]] = deque(maxlen=history)
        self._latest_tick: dict[str, Any] | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._serve_error: BaseException | None = None
        self._runner: web.AppRunner | None = None
        self._start_t: float | None = None
        self._tick_no = 0

    # --- Monitor protocol ---
    def start(self) -> None:
        # Called from within the running loop; spin the server up as a task.
        self._serve_error = None
        self._serve_task = asyncio.create_task(self._serve(), name="webui")
        self._serve_task.add_done_callback(self._remember_serve_failure)

    def stop(self) -> None:
        if self._serve_task is not None:
            self._serve_task.cancel()

    async def wait_stopped(self) -> None:
        if self._serve_task is None:
            return
        try:
            await self._serve_task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            # The done callback normally logs and stores this first; keep cleanup
            # idempotent if the await wins the race.
            if self._serve_error is None:
                self._serve_error = exc
                log.error("web UI server failed", exc_info=(type(exc), exc, exc.__traceback__))

    def on_tick(self, report: TickReport) -> None:
        if (
            self._serve_task is not None
            and self._serve_task.done()
            and self._serve_error is None
        ):
            self._remember_serve_failure(self._serve_task)
        if self._serve_error is not None:
            raise RuntimeError("web UI server failed") from self._serve_error
        if self._start_t is None:
            self._start_t = report.t
        self._tick_no += 1
        payload = self._tick_payload(report)
        self._latest_tick = payload
        if report.posted:
            self._messages.appendleft(
                {"kind": "posted", "text": report.posted, "clock": payload["clock"]}
            )
        elif report.dropped:
            self._messages.appendleft(
                {"kind": "dropped", "text": report.dropped, "clock": payload["clock"]}
            )
        self._broadcast({"type": "tick", **payload})

    def _remember_serve_failure(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        if self._serve_error is not None:
            return
        exc = task.exception()
        if exc is None:
            return
        self._serve_error = exc
        log.error("web UI server failed", exc_info=(type(exc), exc, exc.__traceback__))

    # --- payload builders ---
    def _tick_payload(self, report: TickReport) -> dict[str, Any]:
        d = report.decision
        elapsed = 0.0 if self._start_t is None else report.t - self._start_t
        return {
            "tick": self._tick_no,
            "clock": format_clock(elapsed),
            "should_reply": d.should_reply,
            "score": round(d.score, 3),
            "threshold": round(d.threshold, 3),
            "reasons": list(d.reasons),
            "reason_kinds": {reason: reason_kind(reason) for reason in d.reasons},
            "mood": round(report.mood, 3),
            "n_events": report.n_events,
            "scene": report.scene_summary,
            "transcript": report.transcript_tail,
            "chat": [{"author": c.author, "text": c.text} for c in report.recent_chat],
            "episodic_summary": report.episodic_summary,
            "episodic_history": list(report.episodic_history),
            "semantic_facts": list(report.semantic_facts),
            "posted": report.posted,
            "dropped": report.dropped,
        }

    def _init_payload(self) -> dict[str, Any]:
        return {
            "type": "init",
            "persona": self.persona_name,
            "platform": self.platform,
            "stream": self.stream_info,
            "schema": self.controls.schema(),
            "controls": self.controls.values(),
            "messages": list(self._messages),
            "tick": self._latest_tick,
        }

    # --- websocket plumbing ---
    def _broadcast(self, obj: dict[str, Any]) -> None:
        if not self._clients:
            return
        data = json.dumps(obj)
        for ws in list(self._clients):
            asyncio.create_task(self._safe_send(ws, data))

    async def _safe_send(self, ws: web.WebSocketResponse, data: str) -> None:
        try:
            await ws.send_str(data)
        except Exception:
            self._clients.discard(ws)

    def _handle_client_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        if msg.get("type") == "set" and "key" in msg:
            self.controls.set(str(msg["key"]), msg.get("value"))
            # Echo the resolved state so every client stays in sync.
            self._broadcast({"type": "state", "controls": self.controls.values()})

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        from aiohttp import WSMsgType, web

        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        self._clients.add(ws)
        await ws.send_str(json.dumps(self._init_payload()))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    self._handle_client_message(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)
        return ws

    async def _index(self, _request: web.Request) -> web.Response:
        from aiohttp import web

        return web.Response(text=_INDEX_HTML, content_type="text/html")

    async def _serve(self) -> None:
        try:
            from aiohttp import web
        except ImportError as exc:  # pragma: no cover - guarded at construction
            raise SystemExit("--web needs the 'web' extra: pip install -e '.[web]'") from exc

        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/ws", self._ws_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self._runner = runner
        log.info("web UI on http://%s:%d", self.host, self.port)
        try:
            await asyncio.Event().wait()  # serve until the task is cancelled
        finally:
            await runner.cleanup()


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Lingus · live tuner</title>
<style>
  :root {
    --bg:#0f1115; --panel:#181b22; --panel2:#1f232c; --ink:#e6e8ee; --dim:#8b93a3;
    --accent:#b388ff; --green:#5ad17a; --red:#ff6b6b; --amber:#ffc857; --blue:#5aa9ff;
    --line:#2a2f3a;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  header { display:flex; align-items:center; gap:12px; padding:12px 18px;
    border-bottom:1px solid var(--line); background:var(--panel); }
  header .name { font-weight:700; color:var(--accent); }
  header .dim { color:var(--dim); }
  #conn { margin-left:auto; font-size:12px; }
  #conn.up { color:var(--green); } #conn.down { color:var(--red); }
  main { display:grid; grid-template-columns:320px 1fr; gap:14px; padding:14px 18px; align-items:start; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
  .panel h2 { margin:0 0 10px; font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--dim); }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:900px){ main{grid-template-columns:1fr;} .grid2{grid-template-columns:1fr;} }

  .fab { position:fixed; left:18px; bottom:18px; z-index:30; width:52px; height:52px;
    border-radius:50%; border:1px solid #6f55b6; background:var(--accent); color:#140d20;
    font-weight:900; font-size:22px; cursor:pointer; box-shadow:0 14px 32px rgba(0,0,0,.45); }
  .scrim { position:fixed; inset:0; z-index:40; background:rgba(0,0,0,.52); opacity:0;
    pointer-events:none; transition:opacity .18s; }
  .scrim.open { opacity:1; pointer-events:auto; }
  .sheet { position:fixed; z-index:50; inset:0 auto 0 0; width:min(420px, 92vw);
    background:var(--panel); border-right:1px solid var(--line);
    transform:translateX(-104%); transition:transform .2s ease; padding:16px;
    overflow:auto; box-shadow:18px 0 48px rgba(0,0,0,.45); }
  .sheet.open { transform:translateX(0); }
  .sheet-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
  .sheet-head h2 { margin:0; font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--dim); }
  .icon-btn { width:34px; height:34px; border-radius:8px; border:1px solid var(--line);
    background:var(--panel2); color:var(--ink); cursor:pointer; font-size:20px; line-height:1; }

  .profile-name { font-size:22px; font-weight:800; color:var(--ink); margin-bottom:10px; overflow-wrap:anywhere; }
  .profile-meta { display:grid; grid-template-columns:84px 1fr; gap:6px 10px; margin-bottom:16px; }
  .profile-meta .k, .research .k { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
  .profile-meta .v { min-width:0; overflow-wrap:anywhere; }
  .profile-meta a { color:var(--blue); text-decoration:none; }
  .profile-meta a:hover { text-decoration:underline; }
  .research .summary { color:var(--ink); white-space:pre-wrap; overflow-wrap:anywhere; margin-bottom:10px; }
  .research ul { margin:0 0 12px 0; padding-left:18px; color:var(--ink); }
  .research li { margin-bottom:6px; }
  .sources { display:flex; flex-wrap:wrap; gap:6px; }
  .sources a { max-width:100%; color:var(--dim); background:var(--panel2); border:1px solid var(--line);
    border-radius:8px; padding:3px 7px; text-decoration:none; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

  /* toggle */
  .toggle { display:flex; align-items:center; gap:10px; margin-bottom:14px; }
  .toggle button { border:none; cursor:pointer; border-radius:20px; width:54px; height:28px;
    background:#3a3f4b; position:relative; transition:.15s; }
  .toggle button .knob { position:absolute; top:3px; left:3px; width:22px; height:22px;
    border-radius:50%; background:#fff; transition:.15s; }
  .toggle button.on { background:var(--green); }
  .toggle button.on .knob { left:29px; }
  .toggle .label { font-weight:700; }

  .ctl { margin:12px 0; }
  .ctl .row { display:flex; justify-content:space-between; align-items:baseline; }
  .ctl label { color:var(--ink); }
  .ctl .val { color:var(--accent); font-weight:700; }
  .ctl .help { color:var(--dim); font-size:11px; margin-top:2px; }
  input[type=range] { width:100%; margin-top:6px; accent-color:var(--accent); }
  .deriv { color:var(--dim); font-size:11px; margin-top:8px; border-top:1px dashed var(--line); padding-top:8px; }

  /* arbiter */
  .verdict { font-size:20px; font-weight:800; }
  .verdict.speak { color:var(--green); } .verdict.hold { color:var(--dim); }
  .bar { position:relative; height:22px; background:var(--panel2); border-radius:6px; margin:8px 0; overflow:hidden; }
  .bar .fill { position:absolute; top:0; left:0; bottom:0; background:linear-gradient(90deg,#4a6cf7,#b388ff); }
  .bar .thr { position:absolute; top:-2px; bottom:-2px; width:2px; background:var(--red); }
  .bar .cap { position:absolute; right:6px; top:2px; font-size:11px; color:var(--ink); }
  .meta { display:flex; gap:18px; color:var(--dim); font-size:12px; margin-top:6px; flex-wrap:wrap; }
  .meta b { color:var(--ink); }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
  .chip { font-size:11px; padding:2px 8px; border-radius:10px; background:var(--panel2); color:var(--dim); }
  .chip.pos { color:#0c1a10; background:var(--green); }
  .chip.block { color:#1a0c0c; background:var(--red); }
  .chip.info { color:#1a160c; background:var(--amber); }

  .ctx .k { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.06em; }
  .ctx .v { margin:4px 0 12px; white-space:pre-wrap; word-break:break-word; }
  .memlist { margin:4px 0 12px; padding-left:18px; color:var(--ink); }
  .memlist li { margin-bottom:6px; overflow-wrap:anywhere; }
  .chatline b { color:var(--blue); }

  #log { max-height:340px; overflow:auto; }
  .msg { padding:7px 9px; border-radius:7px; margin-bottom:6px; background:var(--panel2); }
  .msg.posted { border-left:3px solid var(--green); }
  .msg.dropped { border-left:3px solid var(--amber); color:var(--dim); }
  .msg .t { color:var(--dim); font-size:11px; margin-right:6px; }
  .empty { color:var(--dim); }
</style>
</head>
<body>
<header>
  <span class="name" id="persona">Lingus</span>
  <span class="dim" id="platform"></span>
  <span class="dim" id="uptime"></span>
  <span id="conn" class="down">● connecting…</span>
</header>
<button class="fab" id="controlsFab" aria-label="open controls" title="Controls">☰</button>
<div class="scrim" id="sheetScrim"></div>
<aside class="sheet" id="controlSheet" aria-hidden="true">
  <div class="sheet-head">
    <h2>controls</h2>
    <button class="icon-btn" id="closeSheet" aria-label="close controls">×</button>
  </div>
  <div class="toggle">
    <button id="chatBtn" aria-label="toggle chat"><span class="knob"></span></button>
    <span class="label" id="chatLabel">chatting…</span>
  </div>
  <div id="params"></div>
  <div class="deriv" id="deriv"></div>
</aside>
<main>
  <!-- LEFT: stream profile -->
  <section class="panel stream-profile">
    <h2>stream</h2>
    <div class="profile-name" id="streamer">—</div>
    <div class="profile-meta">
      <div class="k">platform</div><div class="v" id="streamPlatform">—</div>
      <div class="k">live</div><div class="v"><a id="liveLink" target="_blank" rel="noopener">—</a></div>
    </div>
    <div class="research">
      <h2>web search context</h2>
      <div class="summary" id="researchSummary">—</div>
      <div class="k">facts</div>
      <ul id="researchFacts"><li>—</li></ul>
      <div class="k">sources</div>
      <div class="sources" id="researchSources">—</div>
    </div>
  </section>

  <!-- RIGHT: live view -->
  <div style="display:grid; gap:14px;">
    <section class="panel">
      <h2>arbiter — should I speak?</h2>
      <div class="verdict hold" id="verdict">waiting…</div>
      <div class="bar"><div class="fill" id="fill"></div><div class="thr" id="thr"></div>
        <div class="cap" id="barcap"></div></div>
      <div class="chips" id="reasons"></div>
      <div class="meta">
        <span>mood <b id="mood">0.00</b></span>
        <span>events <b id="events">0</b></span>
        <span>tick <b id="tick">0</b></span>
      </div>
    </section>

    <div class="grid2">
      <section class="panel ctx">
        <h2>context collected</h2>
        <div class="k">scene</div><div class="v" id="scene">—</div>
        <div class="k">speech (asr)</div><div class="v" id="speech">—</div>
        <div class="k">chat</div><div class="v" id="chat">—</div>
        <div class="k">stream so far</div><div class="v" id="episodicSummary">—</div>
        <div class="k">past stream memories</div><ul class="memlist" id="episodicHistory"><li>—</li></ul>
        <div class="k">known facts</div><ul class="memlist" id="semanticFacts"><li>—</li></ul>
      </section>
      <section class="panel">
        <h2>bot messages</h2>
        <div id="log"><div class="empty">no posts yet — watching</div></div>
      </section>
    </div>
  </div>
</main>

<script>
const $ = (id) => document.getElementById(id);
let ws, schema = [], controls = {}, stream = {};

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { $("conn").className = "up"; $("conn").textContent = "● live"; };
  ws.onclose = () => { $("conn").className = "down"; $("conn").textContent = "● disconnected — retrying";
    setTimeout(connect, 1500); };
  ws.onmessage = (ev) => handle(JSON.parse(ev.data));
}

function send(key, value) { ws && ws.readyState === 1 && ws.send(JSON.stringify({type:"set", key, value})); }

function handle(m) {
  if (m.type === "init") {
    $("persona").textContent = m.persona;
    $("platform").textContent = "· " + m.platform;
    stream = m.stream || {};
    renderStream();
    schema = m.schema; controls = m.controls;
    buildParams();
    renderControls();
    renderMessages(m.messages || []);
    if (m.tick) renderTick(m.tick);
  } else if (m.type === "state") {
    controls = m.controls; renderControls();
  } else if (m.type === "tick") {
    renderTick(m); pushMessage(m);
  }
}

function buildParams() {
  const box = $("params"); box.innerHTML = "";
  for (const s of schema) {
    if (s.kind === "bool") {
      if (s.key === "chat_enabled") continue; // master switch lives at the top of the sheet
      const wrap = document.createElement("div"); wrap.className = "toggle";
      wrap.innerHTML = `<button id="b_${s.key}" aria-label="toggle ${s.key}"><span class="knob"></span></button>
        <span class="label" id="l_${s.key}">${s.label}</span>`;
      box.appendChild(wrap);
      if (s.help) { const h = document.createElement("div"); h.className = "help";
        h.style.marginTop = "-8px"; h.textContent = s.help; box.appendChild(h); }
      $("b_"+s.key).onclick = () => send(s.key, !controls[s.key]);
      continue;
    }
    const wrap = document.createElement("div"); wrap.className = "ctl";
    wrap.innerHTML = `<div class="row"><label>${s.label}</label><span class="val" id="v_${s.key}"></span></div>
      <input type="range" id="r_${s.key}" min="${s.min}" max="${s.max}" step="${s.step}">
      <div class="help">${s.help}</div>`;
    box.appendChild(wrap);
    const r = $("r_"+s.key);
    r.addEventListener("input", () => { $("v_"+s.key).textContent = fmt(s, r.value);
      controls[s.key] = +r.value; renderDeriv(); });
    r.addEventListener("change", () => send(s.key, +r.value));
  }
  $("chatBtn").onclick = () => send("chat_enabled", !controls.chat_enabled);
}

function fmt(s, v) { return s.kind === "int" ? String(Math.round(v)) : (+v).toFixed(2); }

function titleCase(s) {
  s = s == null ? "" : String(s);
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : "—";
}

function hostLabel(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); }
  catch { return url || "—"; }
}

function safeHref(url) {
  try {
    const parsed = new URL(url, location.href);
    return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : "";
  } catch { return ""; }
}

function renderStream() {
  $("streamer").textContent = stream.nickname || "unknown streamer";
  $("streamPlatform").textContent = titleCase(stream.platform);
  const live = $("liveLink");
  const liveHref = safeHref(stream.live_url);
  if (liveHref) {
    live.href = liveHref;
    live.textContent = hostLabel(liveHref);
  } else {
    live.removeAttribute("href");
    live.textContent = "—";
  }
  $("researchSummary").textContent = stream.summary || "—";
  const facts = Array.isArray(stream.facts) ? stream.facts : [];
  $("researchFacts").innerHTML = facts.length
    ? facts.map(f => `<li>${esc(f)}</li>`).join("")
    : "<li>—</li>";
  const sources = Array.isArray(stream.source_urls) ? stream.source_urls : [];
  const sourceLinks = sources.map((u, i) => ({ href: safeHref(u), i })).filter(s => s.href);
  $("researchSources").innerHTML = sourceLinks.length
    ? sourceLinks.map(s => `<a href="${escAttr(s.href)}" target="_blank" rel="noopener">source ${s.i + 1}</a>`).join("")
    : "—";
}

function renderControls() {
  for (const s of schema) {
    if (s.kind === "bool") {
      if (s.key === "chat_enabled") continue;
      const b = $("b_"+s.key); if (!b) continue;
      const bon = !!controls[s.key];
      b.className = bon ? "on" : "";
      const l = $("l_"+s.key); if (l) l.style.color = bon ? "var(--green)" : "var(--red)";
      continue;
    }
    const r = $("r_"+s.key); if (!r) continue;
    r.value = controls[s.key]; $("v_"+s.key).textContent = fmt(s, controls[s.key]);
  }
  const on = !!controls.chat_enabled;
  $("chatBtn").className = on ? "on" : "";
  $("chatLabel").textContent = on ? "chatting ON" : "muted";
  $("chatLabel").style.color = on ? "var(--green)" : "var(--red)";
  renderDeriv();
}

function renderDeriv() {
  $("deriv").innerHTML =
    `resolved → fire threshold <b>${(controls._effective_threshold ?? 0).toFixed(2)}</b>` +
    ` · cooldown <b>${(controls._effective_cooldown ?? 0).toFixed(1)}s</b>`;
}

function renderTick(t) {
  const v = $("verdict");
  v.textContent = t.should_reply ? "SPEAK ✓" : "hold";
  v.className = "verdict " + (t.should_reply ? "speak" : "hold");
  const hi = Math.max(t.score, t.threshold, 1) * 1.1;
  $("fill").style.width = Math.min(100, (t.score / hi) * 100) + "%";
  $("thr").style.left = Math.min(100, (t.threshold / hi) * 100) + "%";
  $("barcap").textContent = `score ${t.score.toFixed(2)} / thr ${t.threshold.toFixed(2)}`;
  $("mood").textContent = (t.mood >= 0 ? "+" : "") + t.mood.toFixed(2);
  $("events").textContent = t.n_events;
  $("tick").textContent = t.tick;
  $("uptime").textContent = "· " + t.clock;
  const kinds = t.reason_kinds || {};
  $("reasons").innerHTML = (t.reasons.length ? t.reasons : ["—"]).map(r => {
    const kind = kinds[r];
    const c = kind === "positive" ? "pos" : kind === "blocking" ? "block" : (r === "—" ? "" : "info");
    return `<span class="chip ${c}">${r}</span>`; }).join("");
  $("scene").textContent = t.scene || "—";
  $("speech").textContent = t.transcript || "—";
  $("chat").innerHTML = t.chat && t.chat.length
    ? t.chat.map(c => `<div class="chatline"><b>${esc(c.author)}:</b> ${esc(c.text)}</div>`).join("")
    : "—";
  $("episodicSummary").textContent = t.episodic_summary || "—";
  renderList("episodicHistory", t.episodic_history);
  renderList("semanticFacts", t.semantic_facts);
}

function renderList(id, items) {
  const list = Array.isArray(items) ? items : [];
  $(id).innerHTML = list.length ? list.map(item => `<li>${esc(item)}</li>`).join("") : "<li>—</li>";
}

function pushMessage(t) {
  if (!t.posted && !t.dropped) return;
  renderMessages([{ kind: t.posted ? "posted" : "dropped",
    text: t.posted || t.dropped, clock: t.clock }], true);
}

function renderMessages(list, prepend) {
  const log = $("log");
  if (!prepend) log.innerHTML = "";
  if (!prepend && list.length === 0) { log.innerHTML = `<div class="empty">no posts yet — watching</div>`; return; }
  const frag = list.map(m =>
    `<div class="msg ${m.kind}"><span class="t">${m.clock||""}</span>${esc(m.text)}</div>`).join("");
  if (prepend) { if (log.querySelector(".empty")) log.innerHTML = ""; log.insertAdjacentHTML("afterbegin", frag); }
  else log.innerHTML = frag;
}

function esc(s){ return (s==null?"":String(s)).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function escAttr(s){ return esc(s).replace(/"/g, "&quot;"); }
function setSheet(open) {
  $("controlSheet").classList.toggle("open", open);
  $("sheetScrim").classList.toggle("open", open);
  $("controlSheet").setAttribute("aria-hidden", open ? "false" : "true");
}
$("controlsFab").onclick = () => setSheet(true);
$("closeSheet").onclick = () => setSheet(false);
$("sheetScrim").onclick = () => setSheet(false);
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") setSheet(false); });
connect();
</script>
</body>
</html>"""
