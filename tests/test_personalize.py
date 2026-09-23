"""Tailored phrasing: trigger rule, output cleaning, and the server round-trip with a mock LLM (no network)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from copilot import constants as C
from copilot.personalize import PersonalizeState, PhrasingResult, build_prompt, clean_line, should_personalize
from copilot.server import app


def _coach(move_id: str, gated: bool = False) -> dict:
    return {"gated": gated, "move_id": move_id, "confidence": 0.8}


# ---------------------------------------------------------------- trigger rule


def test_trigger_only_when_move_changes_or_fact_locks():
    st = PersonalizeState()
    assert should_personalize(_coach("quantify_the_pain"), {}, 3, st) == "move changed"
    st.last_move_id, st.last_index, st.last_facts = "quantify_the_pain", 3, frozenset()
    # same move, next utterances: nothing
    assert should_personalize(_coach("quantify_the_pain"), {}, 4, st) is None
    assert should_personalize(_coach("quantify_the_pain"), {}, 6, st) is None
    # a new durable fact re-triggers (after the cooldown)
    assert should_personalize(_coach("quantify_the_pain"), {"pain_identified": {"since": 5}}, 6, st) == "new fact"
    # long stretch on the same move allows one refresh
    assert should_personalize(_coach("quantify_the_pain"), {}, 3 + C.PERSONALIZE_REFRESH_UTTERANCES, st) == "refresh"


def test_trigger_respects_gate_cooldown_and_skip_list():
    st = PersonalizeState(last_move_id="ask_about_budget", last_index=10)
    assert should_personalize(_coach("propose_next_step", gated=True), {}, 20, st) is None
    assert should_personalize({}, {}, 20, st) is None
    assert should_personalize(_coach("propose_next_step"), {}, 10 + C.PERSONALIZE_COOLDOWN_UTTERANCES - 1, st) is None
    assert should_personalize(_coach("propose_next_step"), {}, 10 + C.PERSONALIZE_COOLDOWN_UTTERANCES, st) == "move changed"
    for skipped in C.PERSONALIZE_SKIP_MOVES:
        assert should_personalize(_coach(skipped), {}, 50, st) is None
    assert should_personalize(_coach("not_a_move"), {}, 50, st) is None


# ---------------------------------------------------------------- output cleaning


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"How many of those 50 dials actually turn into conversations each week?"', "How many of those 50 dials actually turn into conversations each week?"),
        ("Rep: Roughly how many hours a week does the manual logging eat up", "Roughly how many hours a week does the manual logging eat up."),
        ("What would it take to decide before January\nSecond line ignored", "What would it take to decide before January?"),
        ("", None),
        ("ok", None),
        (" ".join(["word"] * (C.PERSONALIZE_MAX_WORDS + 10)), None),
    ],
)
def test_clean_line(raw, expected):
    assert clean_line(raw) == expected


def test_prompt_contains_move_transcript_and_facts():
    move = C.PLAYBOOK_BY_ID["quantify_the_pain"]
    p = build_prompt(move, [{"speaker": "prospect", "text": "We do about 50 dials a day."}], ["pain_identified"], {"open": True, "type": "price"})
    assert move["title"] in p and "50 dials" in p and "pain_identified" in p and "Open objection: price" in p
    for line in move["phrasings"]:
        assert line in p


# ---------------------------------------------------------------- server round-trip


class MockPersonalizer:
    def __init__(self, text: str | None = "How many of those daily dials actually turn into conversations?"):
        self.text = text
        self.calls = 0

    async def __call__(self, move, transcript, facts, objection) -> PhrasingResult:
        self.calls += 1
        self.moves = getattr(self, "moves", []) + [move["id"]]
        return PhrasingResult(text=self.text, model="mock-llm", latency_ms=5.0, input_tokens=400, output_tokens=20, cost_usd=0.0005)


def _recv(ws, wanted: str, limit: int = 300):
    for _ in range(limit):
        msg = json.loads(ws.receive_text())
        if msg["type"] == wanted:
            return msg
        if msg["type"] == "error":
            raise AssertionError(msg)
    raise AssertionError(f"never received {wanted}")


@pytest.fixture
def client_with_llm(mock_judge_factory, helpers, monkeypatch):
    ch = helpers["choice"]
    moves = [m["id"] for m in C.PLAYBOOK]
    # utterance 0: default move (dig_into_the_problem, skipped); utterance 1: quantify_the_pain -> triggers a rewrite
    script = [{}, {"next_move": ch("quantify_the_pain", 0.8, moves)}] + [{"next_move": ch("quantify_the_pain", 0.8, moves)} for _ in range(50)]
    monkeypatch.setattr(C, "PERSONALIZE_VERIFY_WITH_JEV", False)
    with TestClient(app) as c:
        app.state.judge = mock_judge_factory(script)
        app.state.personalizer = MockPersonalizer()
        yield c


def test_phrasing_message_after_move_change(client_with_llm):
    with client_with_llm.websocket_connect("/ws") as ws:
        hello = _recv(ws, "hello")
        assert hello["llm"] == C.LLM_MODEL
        ws.send_text(json.dumps({"type": "utterance", "speaker": "rep", "text": "Walk me through your day."}))
        u0 = _recv(ws, "update")
        assert u0["coaching"]["move_id"] == "dig_into_the_problem"
        ws.send_text(json.dumps({"type": "utterance", "speaker": "prospect", "text": "We do 50 dials a day and log them by hand."}))
        u1 = _recv(ws, "update")
        assert u1["coaching"]["move_id"] == "quantify_the_pain"
        ph = _recv(ws, "phrasing")
        assert ph["index"] == 1 and ph["move_id"] == "quantify_the_pain" and ph["rejected"] is None
        assert ph["text"].startswith("How many of those daily dials")
        assert ph["llm"]["model"] == "mock-llm" and ph["llm"]["shown"] == 1
        # same move on the next utterance: no second LLM call
        ws.send_text(json.dumps({"type": "utterance", "speaker": "rep", "text": "Got it."}))
        _recv(ws, "update")
        assert app.state.personalizer.calls == 1
        assert app.state.personalizer.moves == ["quantify_the_pain"]


def test_no_llm_when_key_missing(mock_judge_factory):
    with TestClient(app) as c:
        app.state.judge = mock_judge_factory([{} for _ in range(10)])
        app.state.personalizer = None
        with c.websocket_connect("/ws") as ws:
            hello = _recv(ws, "hello")
            assert hello["llm"] is None
            ws.send_text(json.dumps({"type": "utterance", "speaker": "rep", "text": "Hi there, thanks for the time."}))
            _recv(ws, "update")
            ws.send_text(json.dumps({"type": "ping"}))
            assert _recv(ws, "pong")  # nothing else (no phrasing) arrived in between
