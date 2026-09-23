"""End-to-end with the real API: replay the three hand-written calls headlessly.

Writes eval/timelines.json (probability per utterance, latency, cost) and asserts the shapes:
good ends high, bad ends low, mixed dips then recovers. Prints latency mean/p95 and cost per call.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from copilot import constants as C
from copilot.cli import run_call
from copilot.jev_client import JevJudge
from copilot.replay import load_call

from conftest import requires_api_key

pytestmark = requires_api_key()

TIMELINES_PATH = Path(__file__).resolve().parent.parent / "eval" / "timelines.json"


@pytest.fixture(scope="module")
def timelines():
    async def _run():
        judge = JevJudge()
        try:
            return {cid: await run_call(load_call(cid), judge, quiet=True) for cid in ("good", "bad", "mixed")}
        finally:
            await judge.aclose()

    results = asyncio.run(_run())
    TIMELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    TIMELINES_PATH.write_text(json.dumps({"model": C.MODEL, "calls": list(results.values())}, indent=2), encoding="utf-8")
    return results


def _probs(r):
    return [row["probability"] for row in r["timeline"]]


def test_no_errors_and_model_pinned(timelines):
    for cid, r in timelines.items():
        assert r["errors"] == 0, cid
        assert r["model"] == C.MODEL, cid
        assert r["requests"] == len(r["timeline"])


def test_good_call_ends_high_and_above_bad(timelines):
    good, bad = timelines["good"], timelines["bad"]
    print(f"\nfinal probabilities: good={good['final_probability']:.2f} bad={bad['final_probability']:.2f} mixed={timelines['mixed']['final_probability']:.2f}")
    assert good["final_probability"] >= 0.70
    assert bad["final_probability"] <= 0.40
    assert good["final_probability"] > bad["final_probability"] + 0.30
    # the good call trends up: the last quarter averages above the first quarter
    p = _probs(good)
    q = len(p) // 4
    assert sum(p[-q:]) / q > sum(p[:q]) / q + 0.3
    assert {"next_step_agreed", "pain_identified", "decision_maker_identified"} <= set(good["facts"])


def test_bad_call_dips_and_flags_objection(timelines):
    bad = timelines["bad"]
    p = _probs(bad)
    assert min(p) <= 0.25
    assert "next_step_agreed" not in bad["facts"]
    assert "competitor_mentioned" in bad["facts"]
    # a price objection is opened at some point
    assert any(row["objection"]["open"] and row["objection"]["type"] == "price" for row in bad["timeline"])
    # the feature dump is caught: the rep monologue turns trigger the "stop talking" coaching at least once
    assert any(row["next_move"] == "stop_talking_ask_open_question" and not row["gated"] for row in bad["timeline"])


def test_mixed_call_dips_then_recovers(timelines):
    mixed = timelines["mixed"]
    p = _probs(mixed)
    half = len(p) // 2
    dip = min(p[:half])
    assert dip <= 0.30, f"expected an early dip, min of first half was {dip:.2f}"
    assert mixed["final_probability"] >= 0.65
    assert mixed["final_probability"] - dip >= 0.35
    assert "next_step_agreed" in mixed["facts"]


def test_latency_and_cost(timelines):
    for cid, r in timelines.items():
        print(f"{cid:<6} latency mean={r['latency_mean_ms']:.0f}ms p95={r['latency_p95_ms']:.0f}ms  tokens={r['total_input_tokens']}  cost=${r['total_cost_usd']:.4f} ({r['utterances']} utterances)")
        assert r["latency_p95_ms"] < 2000  # network included; Jev itself is ~100-300 ms
        assert r["total_cost_usd"] < 0.05  # cents per call
        assert r["total_cost_usd"] == pytest.approx(r["total_input_tokens"] / 1e6 * C.PRICE_PER_M_INPUT_TOKENS_USD, rel=1e-3)


def test_coaching_is_gated_sometimes_but_mostly_available(timelines):
    rows = [row for r in timelines.values() for row in r["timeline"]]
    shown = sum(1 for row in rows if not row["gated"])
    assert 0.5 <= shown / len(rows) <= 1.0, f"coaching shown on {shown}/{len(rows)} utterances"
