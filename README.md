# Lingus

<p><image alt="lingus logo" src="./logo.png"></p>

---

A real-time, characterful live-stream interaction bot. It perceives a live stream (video + audio + chat), fuses it into a shared world-state, decides *when* to speak (arbiter) and *what* to say (generator) and posts to chat, optimized for **personality** over raw coverage.

## Architecture

```
VIDEO  AUDIO  CHAT   →  WORLD STATE  →  ARBITER  →  GENERATOR  →  CHAT
(perception)            (blackboard)    (when?)     (what?)       (filter)   (post)
```

- **Adapters** (`adapters/`) abstract the platform: file-replay + YouTube (Twitch later).
- **Model backends** (`models/`) abstract the models: small ones local (ASR, VLM),
  the generator hosted (OpenAI-compatible — GPT-5.5 / Grok).

## Setup

```bash
source venv/bin/activate
pip install -e ".[dev]"          # core + dev tooling
# later phases:
# pip install -e ".[asr,youtube,llm]"
cp .env.example .env             # fill in keys when wiring Phase 1
```

## Run (offline)

```bash
python -m lingus.app --segment tests/samples/demo
```

Drives the loop from a recorded segment with no network or API keys. You should see
the world-state populate with chat + transcript events, then a simple offline
cognition tick may post a deterministic bot reply.

```bash
python -m lingus.app --segment tests/samples/cake --speed 100
```

This sample replays scene + speech context for a chocolate-cake stain and logs the
bot's reply through the file replay chat adapter.

## Test

```bash
pytest
ruff check src tests
```

## Build status

- [x] **Phase 0** — skeleton: config, world-state, persona schema, adapter/model ABCs, offline loop
- [x] **Phase 0.5** — context snapshot, simple arbiter, deterministic offline reply loop
- [~] **Phase 1** — speech → reply → chat: capture + local ASR + hosted/template generator + governor wired and validated live in observe mode. Remaining: validate the real LLM generator live; post path lands with the Twitch adapter
- [~] **Phase 2** — chat perception: `ChatTrendDetector` (hype/pile-on) built and wired into the arbiter. Remaining: real YouTube live-chat ingestion (`ObserveChatAdapter.incoming` yields nothing)
- [x] **Phase 3** — memory: working + self-memory + dedup/bit-fatigue + episodic summarization + semantic (durable cross-stream facts)
- [ ] **Phase 4** — video (frame gating + VLM scene state) — `youtube.py` `video_frames()` is a stub
- [ ] **Phase 5** — hardening: output moderation pass (incl. the regex spam/offensive filter), staleness/barge-in, per-signal cooldowns. The `ModerationBackend` ABC + `moderation:` config exist but are **not** wired into the post path yet — the non-negotiable gate before any real posting
- [ ] **Phase 6** — eval loop (record / replay / judge)
- [ ] **Final** — Twitch adapter

Legend: `[x]` done · `[~]` partial · `[ ]` not started.
```
