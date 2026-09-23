"""FastAPI app: static UI + one WebSocket per browser tab (replay or live utterances).

    uv run uvicorn copilot.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from . import constants as C
from .engine import CallSession
from .jev_client import JevJudge, resolve_api_key
from .personalize import AnthropicPersonalizer, PersonalizeState, Personalizer, resolve_llm_key, should_personalize, verify_with_jev
from .replay import list_calls, load_call

log = logging.getLogger("copilot.server")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

MAX_REAL_GAP_S = 6.0  # never wait longer than this between utterances, even at 1x
MIN_GAP_S = 0.15


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.judge = JevJudge() if resolve_api_key() else None
    if app.state.judge is None:
        log.error("No API key found (TYPESAFE_API_KEY / JEV_API_KEY). The UI will load but Jev calls will fail.")
    app.state.personalizer = AnthropicPersonalizer() if resolve_llm_key() else None
    if app.state.personalizer is None:
        log.info("No ANTHROPIC_API_KEY: tailored phrasing disabled, generic playbook lines only.")
    yield
    for name in ("judge", "personalizer"):
        obj = getattr(app.state, name, None)
        if obj is not None and hasattr(obj, "aclose"):
            await obj.aclose()


app = FastAPI(title="Sales Copilot (Jev)", lifespan=lifespan)


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------
@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "model": C.MODEL, "has_api_key": bool(resolve_api_key()), "questions": len(C.QUESTIONS), "llm": C.LLM_MODEL if resolve_llm_key() else None}


@app.get("/api/calls")
async def calls() -> list[dict[str, Any]]:
    return list_calls()


@app.get("/api/calls/{call_id}")
async def call(call_id: str) -> JSONResponse:
    try:
        return JSONResponse(load_call(call_id))
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=404)


@app.get("/api/playbook")
async def playbook() -> list[dict[str, Any]]:
    return C.PLAYBOOK


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return {
        "model": C.MODEL,
        "weights": [{"signal": k, **v} for k, v in C.SIGNAL_WEIGHTS.items()],
        "stage_prior": C.STAGE_PRIOR,
        "ema_alpha": C.EMA_ALPHA,
        "thresholds": {
            "fact_persist": C.FACT_PERSIST_THRESHOLD,
            "objection_open": C.OBJECTION_OPEN_THRESHOLD,
            "next_move_min_confidence": C.NEXT_MOVE_MIN_CONFIDENCE,
            "signal_on": C.SIGNAL_ON_THRESHOLD,
        },
        "questions": {k: v for k, v in C.QUESTIONS.items()},
    }


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
class Connection:
    def __init__(self, ws: WebSocket, judge: JevJudge | None, personalizer: Personalizer | None = None):
        self.ws = ws
        self.session = CallSession(judge) if judge else None
        self.personalizer = personalizer
        self.pstate = PersonalizeState()
        self.ptask: asyncio.Task | None = None
        self.llm_stats = {"requests": 0, "shown": 0, "rejected": 0, "errors": 0, "cum_cost_usd": 0.0, "last_ms": 0.0}
        self.replay_task: asyncio.Task | None = None
        self.speed = 1.0
        self.paused = asyncio.Event()
        self.paused.set()  # set == running
        self.send_lock = asyncio.Lock()

    async def send(self, payload: dict[str, Any]) -> None:
        async with self.send_lock:
            await self.ws.send_text(json.dumps(payload))

    async def stop_replay(self) -> None:
        if self.replay_task and not self.replay_task.done():
            self.replay_task.cancel()
            try:
                await self.replay_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self.replay_task = None
        self.paused.set()

    def reset_personalization(self) -> None:
        if self.ptask and not self.ptask.done():
            self.ptask.cancel()
        self.ptask = None
        self.pstate = PersonalizeState()

    # -- tailored phrasing: decide in code, generate in the background, verify with Jev, push if still current --
    def maybe_personalize(self, snap: dict[str, Any]) -> None:
        if self.personalizer is None or self.session is None:
            return
        reason = should_personalize(snap.get("coaching") or {}, snap.get("facts") or {}, snap["index"], self.pstate)
        if not reason:
            return
        if self.ptask and not self.ptask.done():
            self.ptask.cancel()  # a newer moment supersedes the in-flight request
        move = C.PLAYBOOK_BY_ID[snap["coaching"]["move_id"]]
        self.pstate.last_move_id, self.pstate.last_index, self.pstate.last_facts = move["id"], snap["index"], frozenset(snap["facts"])
        transcript = [u.to_state() for u in self.session.utterances[-C.PERSONALIZE_CONTEXT_UTTERANCES :]]
        facts = sorted(snap["facts"])
        self.ptask = asyncio.create_task(self._personalize(snap["index"], move, transcript, facts, snap.get("objection"), reason))

    async def _personalize(self, index: int, move: dict[str, Any], transcript: list[dict[str, Any]], facts: list[str], objection: dict[str, Any] | None, reason: str) -> None:
        assert self.session is not None and self.personalizer is not None
        self.llm_stats["requests"] += 1
        try:
            res = await self.personalizer(move, transcript, facts, objection)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            self.llm_stats["errors"] += 1
            log.warning("personalize failed: %s", e)
            return
        self.llm_stats["cum_cost_usd"] += res.cost_usd
        self.llm_stats["last_ms"] = round(res.latency_ms)
        if res.error:
            self.llm_stats["errors"] += 1
        verified: dict[str, float] = {}
        rejected = res.rejected_reason
        if res.text and C.PERSONALIZE_VERIFY_WITH_JEV:
            ok, verified, why = await verify_with_jev(self.session.judge, res.text, move, transcript, facts)
            if not ok:
                rejected = why
        # stale-protection: the call has moved on to a different move since we started
        latest = self.session.steps[-1] if self.session.steps else None
        if latest is None or latest.coaching.get("move_id") != move["id"] or latest.coaching.get("gated"):
            rejected = rejected or "stale"
        if rejected:
            self.llm_stats["rejected"] += 1
        else:
            self.llm_stats["shown"] += 1
        await self.send(
            {
                "type": "phrasing",
                "index": index,
                "move_id": move["id"],
                "text": None if rejected else res.text,
                "rejected": rejected,
                "reason": reason,
                "llm": {"model": res.model, "latency_ms": round(res.latency_ms), "input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "cost_usd": round(res.cost_usd, 6), "error": res.error, **self.llm_stats},
                "verification": verified,
                "candidate": res.text or (res.extra.get("raw") or None),
            }
        )

    async def run_replay(self, call: dict[str, Any]) -> None:
        assert self.session is not None
        utterances = call["utterances"]
        prev_t = None
        try:
            for u in utterances:
                if prev_t is not None:
                    gap = min(MAX_REAL_GAP_S, max(0.0, float(u["t"]) - prev_t))
                    await asyncio.sleep(max(MIN_GAP_S, gap / max(0.1, self.speed)))
                prev_t = float(u["t"])
                await self.paused.wait()
                snap = await self.session.ingest(u["speaker"], u["text"], float(u["t"]))
                snap["replay"] = {"call": call["id"], "index": snap["index"], "total": len(utterances)}
                await self.send(snap)
                self.maybe_personalize(snap)
            await self.send({"type": "replay_done", "call": call["id"], "final_probability": self.session.probability})
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.exception("replay failed")
            await self.send({"type": "error", "message": f"replay failed: {e}"})


@app.websocket("/ws")
async def websocket(ws: WebSocket) -> None:
    await ws.accept()
    conn = Connection(ws, ws.app.state.judge, getattr(ws.app.state, "personalizer", None))
    await conn.send(
        {
            "type": "hello",
            "model": C.MODEL,
            "has_api_key": conn.session is not None,
            "llm": C.LLM_MODEL if conn.personalizer is not None else None,
            "personalize_skip_moves": sorted(C.PERSONALIZE_SKIP_MOVES),
            "calls": list_calls(),
            "playbook": C.PLAYBOOK,
            "weights": conn.session.weights_table() if conn.session else [],
            "stage_prior": C.STAGE_PRIOR,
            "ema_alpha": C.EMA_ALPHA,
            "thresholds": {"next_move_min_confidence": C.NEXT_MOVE_MIN_CONFIDENCE, "objection_open": C.OBJECTION_OPEN_THRESHOLD, "signal_on": C.SIGNAL_ON_THRESHOLD},
        }
    )
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await conn.send({"type": "error", "message": "invalid json"})
                continue
            await handle_message(conn, msg)
    except WebSocketDisconnect:
        pass
    finally:
        await conn.stop_replay()
        conn.reset_personalization()


async def handle_message(conn: Connection, msg: dict[str, Any]) -> None:
    kind = msg.get("type")
    if conn.session is None and kind not in ("ping",):
        await conn.send({"type": "error", "message": "server has no API key (set TYPESAFE_API_KEY and restart)"})
        return
    session = conn.session
    assert session is not None

    if kind == "start_replay":
        await conn.stop_replay()
        try:
            call = load_call(str(msg.get("call", "good")))
        except FileNotFoundError as e:
            await conn.send({"type": "error", "message": str(e)})
            return
        session.reset()
        conn.reset_personalization()
        conn.speed = float(msg.get("speed", 1.0))
        await conn.send({"type": "reset", "call": call, "weights": session.weights_table()})
        conn.replay_task = asyncio.create_task(conn.run_replay(call))
    elif kind == "pause":
        conn.paused.clear()
        await conn.send({"type": "paused"})
    elif kind == "resume":
        conn.paused.set()
        await conn.send({"type": "resumed"})
    elif kind == "set_speed":
        conn.speed = max(0.25, min(50.0, float(msg.get("speed", 1.0))))
    elif kind == "stop":
        await conn.stop_replay()
        await conn.send({"type": "stopped"})
    elif kind == "reset":
        await conn.stop_replay()
        session.reset()
        conn.reset_personalization()
        await conn.send({"type": "reset", "call": None, "weights": session.weights_table()})
    elif kind == "utterance":
        text = str(msg.get("text", "")).strip()
        speaker = "prospect" if str(msg.get("speaker", "rep")).lower().startswith("p") else "rep"
        if not text:
            return
        snap = await session.ingest(speaker, text, msg.get("t"))
        await conn.send(snap)
        conn.maybe_personalize(snap)
    elif kind == "set_weights":
        weights = msg.get("weights") or {}
        session.set_weights({k: float(v) for k, v in weights.items()})
        if "stage_prior" in msg and isinstance(msg["stage_prior"], dict):
            for k, v in msg["stage_prior"].items():
                if k in session.stage_prior:
                    session.stage_prior[k] = float(v)
        session.recompute()
        snap = session.snapshot()
        snap["type"] = "recomputed"
        await conn.send(snap)
    elif kind == "ping":
        await conn.send({"type": "pong"})
    else:
        await conn.send({"type": "error", "message": f"unknown message type {kind!r}"})
