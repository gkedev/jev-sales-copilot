"""HTTP + WebSocket protocol tests with a scripted judge (no network)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from copilot import constants as C
from copilot.server import app


@pytest.fixture
def client(mock_judge_factory, helpers):
    n = helpers["noul"]
    # enough scripted answers for a full replay: first prospect turn carries a buying signal
    script = [{}, {"buying_signal": n(0.95)}] + [{} for _ in range(120)]
    with TestClient(app) as c:
        app.state.judge = mock_judge_factory(script)  # replace whatever lifespan created
        yield c


def test_http_endpoints(client):
    assert client.get("/").status_code == 200
    assert "Sales Copilot" in client.get("/").text
    h = client.get("/api/health").json()
    assert h["ok"] and h["model"] == C.MODEL
    calls = client.get("/api/calls").json()
    assert {c["id"] for c in calls} >= {"good", "bad", "mixed"}
    assert client.get("/api/calls/good").json()["id"] == "good"
    assert client.get("/api/calls/nope").status_code == 404
    assert len(client.get("/api/playbook").json()) == len(C.PLAYBOOK)
    cfg = client.get("/api/config").json()
    assert cfg["model"] == C.MODEL and "next_move" in cfg["questions"]


def _recv(ws, wanted: str, limit: int = 200):
    for _ in range(limit):
        msg = json.loads(ws.receive_text())
        if msg["type"] == wanted:
            return msg
        if msg["type"] == "error":
            raise AssertionError(msg)
    raise AssertionError(f"never received {wanted}")


def test_websocket_replay_and_weights(client):
    with client.websocket_connect("/ws") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello" and hello["has_api_key"] is True
        assert [c["id"] for c in hello["calls"]][:3] == ["good", "bad", "mixed"]
        assert len(hello["weights"]) == len(C.SIGNAL_WEIGHTS)

        ws.send_text(json.dumps({"type": "start_replay", "call": "good", "speed": 50}))
        reset = _recv(ws, "reset")
        assert reset["call"]["id"] == "good"
        first = _recv(ws, "update")
        assert first["index"] == 0 and first["replay"]["total"] == len(reset["call"]["utterances"])
        second = _recv(ws, "update")
        assert second["index"] == 1 and second["speaker"] == "prospect"
        assert second["signals"]["buying_signal"]["value"] == 0.95
        assert second["probability"] > first["probability"]
        assert second["jev"]["requests"] == 2
        assert set(second) >= {"stage", "coaching", "objection", "talk", "sensitivity", "timeline", "weights", "jev_last"}

        ws.send_text(json.dumps({"type": "pause"}))
        _recv(ws, "paused")
        ws.send_text(json.dumps({"type": "set_weights", "weights": {"buying_signal": 0.6}}))
        rec = _recv(ws, "recomputed")
        assert rec["timeline"][1]["probability"] > second["timeline"][1]["probability"]
        assert app.state.judge.calls == rec["jev"]["requests"]  # no new Jev calls for recompute

        ws.send_text(json.dumps({"type": "stop"}))
        _recv(ws, "stopped")


def test_websocket_live_utterances(client):
    with client.websocket_connect("/ws") as ws:
        _recv(ws, "hello")
        ws.send_text(json.dumps({"type": "utterance", "speaker": "rep", "text": "Walk me through your day."}))
        u1 = _recv(ws, "update")
        assert u1["index"] == 0 and u1["speaker"] == "rep" and u1["t"] == 0.0
        ws.send_text(json.dumps({"type": "utterance", "speaker": "prospect", "text": "It is chaos.", "t": 7.5}))
        u2 = _recv(ws, "update")
        assert u2["index"] == 1 and u2["speaker"] == "prospect" and u2["t"] == 7.5
        assert u2["talk"]["rep_words"] == 5 and u2["talk"]["prospect_words"] == 3
        ws.send_text(json.dumps({"type": "reset"}))
        r = _recv(ws, "reset")
        assert r["call"] is None
        ws.send_text(json.dumps({"type": "bogus"}))
        err = json.loads(ws.receive_text())
        assert err["type"] == "error"
