# Sales Copilot — live closing probability with Jev

A live sales-call copilot. As each utterance of a call arrives (replayed transcript, typed text, or the
browser microphone), the app asks TypeSafe's **Jev** (`jev-1.13.0`, a System One model that returns typed
decisions instead of text) ~37 atomic questions in **one request**, and within a few hundred milliseconds updates:

1. **Closing probability** (0–100 %) as a live line chart — the hero visual.
2. **Live signals** — what is going well / badly right now (buying signal, pain identified, budget discussed,
   decision maker, timeline, objection open, competitor, buyer confusion, rep pitching, rep talking too much, …).
3. **Call stage** (opening, discovery, qualification, pitch/demo, objection handling, pricing, closing, next steps).
4. **Open objection type** (price, timing, authority, need, trust, competitor) — speculative, confidence-gated.
5. **Next best move** — a coaching card picked by Jev from a hand-written 15-move playbook, with the best-fitting
   pre-written phrasing highlighted. Jev picks; code shows the text. Gated on confidence ("listening…" otherwise).
6. **Talk ratio and pace**, computed in code from the transcript.
7. **Tailored phrasing** (optional) — when the move changes, a small fast LLM (Claude Haiku 4.5) rewrites the chosen
   playbook line for *this* conversation, off the critical path; Jev then judges the sentence and code drops it if it
   invents a fact. The generic line is always shown first.

Jev never generates text. All arithmetic (composite score, EMA, talk ratio, durations, fact persistence) lives in code.
The only generative call is the optional tailored phrasing, and it is gated, verified and never blocking.

## Quick start

```bash
git clone https://github.com/moritzkremb/jev-sales-copilot.git && cd jev-sales-copilot
cp .env.example .env            # put TYPESAFE_API_KEY=... in it (JEV_API_KEY also accepted)
                                # optional: ANTHROPIC_API_KEY=... enables tailored phrasing
./run.sh                        # → http://localhost:8000
```

`run.sh` resolves the keys from the environment, then `.env`, then an optional shared secrets file (`COPILOT_SECRETS_FILE`,
lines `JEV_API_KEY=…` / `ANTHROPIC_API_KEY=…`).
It never prints a key. Keys stay server-side; the browser only talks to the local WebSocket.

Other entry points:

```bash
./run.sh cli --transcript good              # headless replay of data/calls/good.json, prints the timeline
./run.sh cli --transcript my_call.txt       # plain text: one "rep: …" / "prospect: …" per line, optional "[secs]" prefix
./run.sh cli --all --out eval/timelines.json
./run.sh test                               # unit + server + real-API integration + e2e (needs the key)
COPILOT_SKIP_LIVE=1 ./run.sh test           # offline: unit + server tests only
```

Requirements: Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/). No Node build step; the UI is one HTML file and loads Chart.js from a CDN.

## Modes

| Mode | How | Notes |
| --- | --- | --- |
| **Replay** (primary, used by tests and the demo) | pick one of the three hand-written calls, set speed (0.5×–20×), Play / Pause / Stop | utterances stream over a WebSocket with their real gaps (capped at 6 s) so the chart animates |
| **Video** (the tangible demo) | pick a real recorded call, **Load video**, then Play — the YouTube player is embedded and the copilot fires each utterance the moment it finishes being spoken | you see exactly what the rep would have seen in real time. Scrub freely: seeking back resets the session and fast-forwards the copilot to the new position; seeking forward catches up |
| **Live mic** | Chrome only — Web Speech API (`webkitSpeechRecognition`, continuous, interim results) | there is no speaker diarization in the browser: toggle **Rep / Prospect** (or press `S`) before each turn. Each final recognition result is sent as one utterance |
| **Type** | type an utterance, pick the speaker, Enter | the speaker alternates automatically after each send |

The three replay calls (`data/calls/*.json`, 38–42 utterances, `rep`/`prospect`, timestamps in seconds):

- `good` — Summit HVAC: discovery-led, pain quantified, budget + decision process surface, closes with a verbal yes (ends ≈ 90 %).
- `bad` — Bluewater Plumbing: feature dump, unhandled price objection, competitor never addressed, prospect disengages (ends ≈ 6 %).
- `mixed` — Ridgeline Electric: hostile start and price pushback (dips to ≈ 16 %), rep pivots to discovery, quantifies pain, reframes price, agrees a pilot (ends ≈ 80 %).

Plus one **real recorded call** for Video mode:

- `cloudtalk` — [CloudTalk's published discovery + BANT call](https://www.youtube.com/watch?v=uovWCGCl2-s) (9 min, 48 utterances): Nick qualifies Bradford,
  an SDR manager dialing from mobile phones and logging into HubSpot by hand. Transcribed locally with faster-whisper (`small.en`), speakers
  labeled by hand, `t`/`t_end` aligned to the YouTube timeline; the JSON carries `"video": {"youtube_id": …}`. In a real run the copilot
  goes 26 % → 80 %, locks pain, budget, decision maker, timeline and next step, and dips when Bradford says he is not the decision maker.

To add your own recording: `yt-dlp -f ba -o call.webm <url>`, `ffmpeg -i call.webm -ar 16000 -ac 1 call.wav`, transcribe (faster-whisper works
without torch: `uv run --with faster-whisper python -c '…'`), group segments into `rep`/`prospect` turns with `t` and `t_end`, add `video.youtube_id`,
drop the file in `data/calls/`.

## Architecture

```
static/index.html          single-page UI (Chart.js), WebSocket client, Web Speech API, YouTube IFrame sync for Video mode
copilot/server.py          FastAPI: /, /api/*, /ws (one CallSession per tab; replay task, live utterances, weight changes)
copilot/engine.py          CallSession: state window, feature extraction, composite + EMA, facts, objection state,
                           coaching gate, talk ratio / pace, sensitivity, recompute-without-Jev
copilot/jev_client.py      async wrapper over typesafe-sdk; logs model / usage / latency / request id; cost accounting
copilot/personalize.py     optional tailored phrasing: trigger rule, prompt, Anthropic HTTP client, output cleaning, Jev verification
copilot/constants.py       THE reviewable module: model pin, all 37 questions, weights, stage priors, thresholds, LLM settings
data/playbook.json         15 coaching moves: what / not_for (→ Choice option descriptions) + 3 phrasings each
data/calls/*.json          the three hand-written replay transcripts + cloudtalk.json (real recording, video-aligned)
copilot/cli.py             headless replay → terminal timeline / eval JSON
copilot/replay.py          transcript loading (JSON or text)
tests/                     unit (mock Jev), server (mock Jev), integration (real API), e2e (real API)
eval/timelines.json        probability per utterance for the three calls, from a real run
eval/integration_report.json  per-case observed values from the labeled-utterance test
```

### One request per utterance (speculative fan-out)

State sent to Jev (kept small on purpose — a rolling window):

```json
{
  "call_facts": {"duration_min": 5.7, "talk_ratio_rep": 0.54, "stage_history": ["opening","discovery"],
                 "known_facts": ["pain_identified","budget_discussed"], "objection": {"open": true, "type": "price"}},
  "recent_transcript": [{"t": "05:12", "speaker": "rep", "text": "…"}, "… last 12 utterances …"],
  "latest_utterance": {"t": "05:45", "speaker": "prospect", "text": "…"}
}
```

Questions (all in `copilot/constants.py`, every one written out in full and referencing `` `latest_utterance.text` `` or
`` `recent_transcript` ``):

- **Nouls (16)** — prospect-turn: `buying_signal`, `commitment`, `next_step_agreed`, `prospect_objecting`, `prospect_accepts`,
  `prospect_disengaging`, `buyer_confused`, `prospect_asked_price`; rep-turn: `rep_discovery_question`, `rep_pitching`;
  either: `next_step_proposed`, `competitor_mentioned`; window facts: `pain_identified`, `budget_discussed`,
  `decision_maker_identified`, `timeline_known`. Each has `{true, false}` criteria with examples.
- **Scores (3)** — `rapport`, `engagement`, `urgency`; 3 levels each, described as concrete situations with `signals`.
- **Choices (3 + 15)** — `stage` (8 options, `{what, not_for}`), `objection_type` (6 + `none`), `next_move` (15 playbook
  moves, `{what, not_for}` from the playbook), and one speculative `phrasing::<move>` Choice per move over its 3 lines.

Speaker-specific Nouls are asked on every turn and **masked in code** when the speaker does not match (Jev is literal;
conditionals belong in code). ~7,500 input tokens per request, ≈ $0.0003.

### Closing probability = code, not prompt

```
inst = STAGE_PRIOR[stage] + Σ weight_i · x_i
    noul / fact / code signals: x ∈ [0,1], contribution = w·x
    scores: x = score/(levels−1), contribution = w·(x−0.5)·2   (middle level is neutral)
p_t  = 0.40 · inst + 0.60 · p_{t−1}                           (EMA; clamped to [0.03, 0.95])
```

All weights, priors and the EMA α are in `SIGNAL_WEIGHTS` / `STAGE_PRIOR` / `EMA_ALPHA` and are shown in the UI as an
editable table. **Applying new weights recomputes the whole timeline from the stored raw answers with zero new Jev
calls** — "change a coefficient, not a prompt". The **sensitivity panel** lists the three contributions that moved most on the last utterance.

Durable facts (`pain_identified`, `budget_discussed`, `decision_maker_identified`, `timeline_known`, `next_step_agreed`)
persist in code once a Noul ≥ 0.70 (🔒 in the UI) and are fed back to Jev via `call_facts.known_facts`, so
"ask about budget" stops being suggested once budget is known.

### Confidence gating

| Decision | Gate |
| --- | --- |
| show coaching card | `next_move.confidence ≥ 0.35`, else "listening…" (leaning move shown greyed) |
| highlight a phrasing | `phrasing::<move>.confidence ≥ 0.30` |
| open objection | `prospect_objecting ≥ 0.60` on a prospect turn; type from `objection_type` if `confidence ≥ 0.40` |
| clear objection | prospect turn with `prospect_accepts` / `buying_signal` / `commitment` / `next_step_agreed` ≥ 0.60 **and** `prospect_disengaging < 0.60`; or 8 utterances without re-raising |
| persist a fact | Noul ≥ 0.70 |
| light a signal | Noul ≥ 0.60 |

### Tailored phrasing: Jev decides, a small LLM writes one sentence, Jev checks it

The playbook lines are generic by construction ("how many *jobs* slip" on a call about *dials*). With `ANTHROPIC_API_KEY`
set, the server adds one sentence written for this exact moment. The division of labour keeps the "nothing a human
didn't vet reaches the rep" property as far as possible:

| Step | Who | Detail |
| --- | --- | --- |
| which move | Jev | `next_move` Choice, gated on confidence ≥ 0.35, exactly as before |
| whether to call the LLM | code | only when the un-gated move **changed**, a **new fact** locked in, or the same move has held for 8 utterances; cooldown 2 utterances; three moves whose generic line is already fine are skipped (`PERSONALIZE_SKIP_MOVES`) |
| write the sentence | Claude Haiku 4.5 | last 6 utterances + known facts + the move's `what` and 3 generic lines as style anchors → one sentence, ≤ 32 words (55 for summaries), no invented numbers/names/features; `temperature 0.4`, ~900 input / ~30 output tokens, ≈ $0.0009 |
| validate | code | one line, quotes stripped, length-capped, discarded if the model hit `max_tokens` |
| verify | Jev | two Nouls on `{candidate_line, move, recent_transcript, known_facts}`: `invents_fact` (drop if ≥ 0.5) and `on_move` (drop if < 0.5) |
| stale check | code | dropped if the call has moved to a different move (or "listening…") since the request started |
| show | UI | `say (tailored): …` appears above the generic lines, with model, latency, Jev scores and trigger; dropped candidates are shown struck through so you can see the guardrail work |

Latency is off the critical path: Jev's ~350 ms answer updates everything immediately with the generic line; the
tailored line lands ~1 s later via a separate `phrasing` WebSocket message. In a real run of the CloudTalk call the LLM
was called 7 times over 48 utterances (≈ $0.006), showed 2 lines, and dropped 5: one for inventing a feature nobody had
mentioned (`invents_fact 0.80`), two summaries for length (since fixed with a per-move cap), and two as stale because at
50× replay speed the move changed before the LLM returned; at real speed that is rare.

Without the key the app behaves exactly as before. `tests/test_personalize.py` covers the trigger rule, output cleaning
and the server round-trip with a mock LLM (no network).

## Tests and evaluation

```bash
./run.sh test                    # everything (needs API key; ~1 min, ≈ $0.05 of Jev usage)
uv run pytest tests/test_engine_unit.py tests/test_server.py -q     # offline
```

- `tests/test_engine_unit.py` (25) — composite arithmetic and centering, clamping, EMA, talk ratio, pace window, fact
  persistence + feedback into state, speaker masking, objection lifecycle (open / not-cleared-by-disengaging / clear /
  expire), coaching gate, talk-ratio penalty, recompute-without-Jev, sensitivity, state window shape, error resilience,
  transcript loading, question completeness.
- `tests/test_server.py` (3) — HTTP endpoints; WebSocket replay (hello → reset → updates), pause, `set_weights` →
  `recomputed` with no extra Jev calls; live utterances; reset; unknown message → error.
- `tests/test_personalize.py` (11) — trigger rule (move change / new fact / refresh / cooldown / gate / skip list), output
  cleaning, prompt contents, `phrasing` round-trip with a mock LLM and one call per move change, silent when no key.
- `tests/test_integration_jev.py` (real API) — 19 hand-labeled utterances (clear price/timing/authority objection,
  buying signal, commitment, feature dump, budget, decision maker, next-step proposed/agreed, discovery question,
  disengaging, confusion, competitor, timeline, pain, acceptance, neutral price question, warm opening), 56 threshold
  checks in total. Requires ≥ 90 % pass; writes `eval/integration_report.json`. **Last run: 56/56 = 100 %.**
- `tests/test_e2e_replay.py` (real API) — replays all three calls, writes `eval/timelines.json`, asserts good ends ≥ 0.70
  and > bad + 0.30, bad ends ≤ 0.40 with a price objection flagged and the "stop talking" coaching fired, mixed dips
  ≤ 0.30 in its first half then ends ≥ 0.65; checks model pin, zero errors, latency p95 and cost per call.

Last real run (see `eval/timelines.json`):

| call | final p | min | facts persisted | Jev latency mean / p95 | tokens | cost |
| --- | --- | --- | --- | --- | --- | --- |
| good | 0.91 | 0.26 | pain, budget, decision maker, timeline, next step | 345 ms / 522 ms | 330,109 | $0.0139 |
| bad | 0.07 | 0.07 | competitor, budget (no next step ever agreed) | 391 ms / 532 ms | 298,774 | $0.0125 |
| mixed | 0.80 | 0.15 | pain, budget, decision maker, timeline, next step | 382 ms / 525 ms | 330,793 | $0.0139 |

Latency is measured client-side from macOS including TLS and network; the first request of a process is ~0.9 s (connection setup).

## Limitations

- **No diarization** in live-mic mode: the Web Speech API returns one stream; you mark who is speaking. Chrome only.
- Jev sees a **rolling 12-utterance window** plus a compact fact list, not the whole call. That keeps it fast and literal;
  facts detected earlier persist in code, not in the model's context.
- `objection_type` is asked speculatively on every turn and will name a type even for a neutral price question — that is
  why an objection only opens when the separate `prospect_objecting` Noul fires.
- Weights and thresholds were tuned on the three hand-written calls and 19 labeled utterances; treat them as starting
  points for real call data. Composite probability is a coaching signal, not a calibrated forecast.
- The phrasing Choices add ~1,900 tokens and ~20 ms per request; drop them from `QUESTIONS` if cost matters more than the "say this" line.
- The tailored line is the one place a generative model can be wrong. Jev's `invents_fact` check is conservative (it dropped
  a line for mentioning "automatic note summaries" that the prospect had in fact asked about), so expect some good lines to be
  discarded. It is also the one external dependency beyond Jev; without the key the feature is simply off.
- Model is pinned to `jev-1.13.0`; re-run the integration test when moving to a newer version.
