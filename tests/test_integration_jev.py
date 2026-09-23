"""Real-API integration: hand-labeled utterances must land on the expected side of the thresholds.

One Jev request per case (all ~35 questions, speculative fan-out). Reports the pass rate and
writes eval/integration_report.json. Requires TYPESAFE_API_KEY (or JEV_API_KEY).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from copilot import constants as C
from copilot.engine import CallSession, Utterance
from copilot.jev_client import JevJudge

from conftest import requires_api_key

pytestmark = requires_api_key()

REPORT_PATH = Path(__file__).resolve().parent.parent / "eval" / "integration_report.json"
MIN_PASS_RATE = 0.90

# A short, neutral context so `recent_transcript` is realistic. Each case appends its own turns.
BASE_CONTEXT = [
    ("rep", "Hi Dan, Maya from Relay. Thanks for making time today."),
    ("prospect", "Sure, happy to chat, I have about half an hour."),
    ("rep", "Great. Before I say anything about Relay, can you tell me a bit about how scheduling works for your team today?"),
    ("prospect", "We have eighteen techs. Our dispatcher builds the board in a spreadsheet every evening and then re-shuffles all day when jobs run long."),
]

Check = tuple[str, Callable[[dict[str, Any], dict[str, Any]], bool]]


def noul_ge(name: str, thr: float) -> Check:
    return (f"{name} >= {thr}", lambda a, s: a[name]["noul"] >= thr)


def noul_lt(name: str, thr: float) -> Check:
    return (f"{name} < {thr}", lambda a, s: a[name]["noul"] < thr)


def choice_is(name: str, *options: str) -> Check:
    return (f"{name} in {options}", lambda a, s: a[name]["choice"] in options)


def score_ge(name: str, thr: float) -> Check:
    return (f"{name}.score >= {thr}", lambda a, s: a[name]["score"] >= thr)


def score_le(name: str, thr: float) -> Check:
    return (f"{name}.score <= {thr}", lambda a, s: a[name]["score"] <= thr)


T_ON = C.SIGNAL_ON_THRESHOLD  # 0.6
T_FACT = C.FACT_PERSIST_THRESHOLD  # 0.7
T_OFF = 0.40

# (case id, extra context turns, latest utterance, checks)
CASES: list[tuple[str, list[tuple[str, str]], tuple[str, str], list[Check]]] = [
    (
        "price_objection",
        [("prospect", "What does something like this cost?"), ("rep", "For eighteen techs it is eleven hundred a month on the annual plan.")],
        ("prospect", "Eleven hundred a month? That's way over what we'd ever spend on scheduling software. That's a lot of money for a calendar."),
        [noul_ge("prospect_objecting", T_ON), choice_is("objection_type", "price"), choice_is("stage", "objection_handling", "pricing"), noul_lt("buying_signal", T_OFF)],
    ),
    (
        "buying_signal",
        [("rep", "When a job runs long, Relay re-sequences the rest of the day and texts each customer a new window automatically.")],
        ("prospect", "Okay, that's exactly the part that's killing us. How would we get started with something like this?"),
        [noul_ge("buying_signal", T_ON), noul_lt("prospect_objecting", T_OFF), noul_lt("prospect_disengaging", T_OFF)],
    ),
    (
        "commitment",
        [("rep", "So if Greg is comfortable on Thursday, is there anything else that would stop us starting the pilot the following Monday?")],
        ("prospect", "No. Honestly, let's do it. Send the agreement over and we'll sign this week."),
        [noul_ge("commitment", T_ON), noul_ge("buying_signal", T_ON), choice_is("stage", "closing", "next_steps"), noul_lt("prospect_objecting", T_OFF)],
    ),
    (
        "rep_feature_dump",
        [],
        ("rep", "So Relay is a field service platform. We automate the daily board, route techs efficiently, re-dispatch in real time, keep customers updated by SMS, integrate with every accounting package, and our mobile app has GPS tracking, digital forms, photo capture and signatures. The reporting suite alone pays for itself."),
        [noul_ge("rep_pitching", T_ON), noul_lt("rep_discovery_question", T_OFF), choice_is("stage", "pitch_demo"), choice_is("next_move", "stop_talking_ask_open_question")],
    ),
    (
        "budget_mention",
        [("rep", "Before I give you a number, what are you spending today on scheduling, including your dispatcher's time?")],
        ("prospect", "The add-on we use is about two hundred a month, and I've got discretion up to about a thousand a month for something that actually fixes this."),
        [noul_ge("budget_discussed", T_FACT), noul_lt("prospect_objecting", T_OFF)],
    ),
    (
        "decision_maker",
        [("rep", "How does a decision like this usually get made at Summit?")],
        ("prospect", "I can sign off on this myself up to a thousand a month. Above that, Greg the owner and I decide together."),
        [noul_ge("decision_maker_identified", T_FACT), choice_is("stage", "qualification")],
    ),
    (
        "next_step_proposed_by_rep",
        [("prospect", "That would give Greg what he needs, yes.")],
        ("rep", "Then how about I set up a thirty-minute demo with you, Greg and Carla on Thursday at ten, using your real job list?"),
        [noul_ge("next_step_proposed", T_ON), choice_is("stage", "next_steps"), noul_lt("rep_pitching", T_OFF)],
    ),
    (
        "next_step_agreed_by_prospect",
        [("rep", "How about a thirty-minute demo with Carla on Thursday at ten, using your actual job list?")],
        ("prospect", "Thursday at ten works. I'll make sure Carla joins."),
        [noul_ge("next_step_agreed", T_ON), noul_lt("prospect_disengaging", T_OFF), choice_is("stage", "next_steps")],
    ),
    (
        "rep_discovery_question",
        [],
        ("rep", "What happens to the customers further down the route when a job runs late?"),
        [noul_ge("rep_discovery_question", T_ON), noul_lt("rep_pitching", T_OFF), choice_is("stage", "discovery")],
    ),
    (
        "prospect_disengaging",
        [("rep", "I'd love to show you the dispatcher dashboard live, it really comes alive with the map view.")],
        ("prospect", "Yeah, just send me some info and I'll take a look when I get a chance. I need to run."),
        [noul_ge("prospect_disengaging", T_ON), noul_lt("next_step_agreed", T_OFF), noul_lt("buying_signal", T_OFF)],
    ),
    (
        "buyer_confused",
        [("rep", "We sync two-way with your accounting so invoices flow automatically, which closes the loop from dispatch to cash collection.")],
        ("prospect", "Wait, what do you mean by closes the loop? I'm not sure I follow."),
        [noul_ge("buyer_confused", T_ON), choice_is("next_move", "clarify_simply")],
    ),
    (
        "competitor_mentioned",
        [("rep", "For nine techs you'd be on our Growth plan at eight hundred and fifty a month.")],
        ("prospect", "We looked at DispatchPro last year and they were about half that price."),
        [noul_ge("competitor_mentioned", T_ON), noul_ge("prospect_objecting", T_ON), choice_is("objection_type", "competitor", "price")],
    ),
    (
        "timeline_known",
        [("rep", "What made you take this call now?")],
        ("prospect", "Summer. We add six seasonal techs in June and Carla can't do another summer on the spreadsheet, so we need something in place before then."),
        [noul_ge("timeline_known", T_FACT), score_ge("urgency", 1.0), noul_ge("pain_identified", T_FACT)],
    ),
    (
        "pain_identified",
        [("rep", "What's the most painful part of that for you?")],
        ("prospect", "Same-day changes. Carla is on the phone nonstop re-shuffling and last month we double-booked two techs to the same address twice. Customers call us angry."),
        [noul_ge("pain_identified", T_FACT), choice_is("next_move", "quantify_the_pain", "dig_into_the_problem", "summarize_and_check_understanding"), choice_is("stage", "discovery")],
    ),
    (
        "timing_objection",
        [("rep", "Would it make sense to start a pilot next month?")],
        ("prospect", "Honestly, not right now. We're slammed until September, can we revisit this after the busy season?"),
        [noul_ge("prospect_objecting", T_ON), choice_is("objection_type", "timing"), choice_is("next_move", "handle_timing_objection", "create_urgency_honestly")],
    ),
    (
        "authority_objection",
        [("rep", "Shall I send the agreement over so you can start Monday?")],
        ("prospect", "I can't approve this myself. It would have to go through our owner and he's out for two weeks."),
        [choice_is("objection_type", "authority"), noul_ge("decision_maker_identified", T_FACT), noul_ge("prospect_objecting", T_ON)],
    ),
    (
        "prospect_accepts",
        [("prospect", "Eleven hundred is a lot."), ("rep", "Fair. Compare it to the roughly five thousand a month you described in slipped jobs and Carla's hours.")],
        ("prospect", "Yeah, okay, that's a fair way to look at it. That makes sense."),
        [noul_ge("prospect_accepts", T_ON), noul_lt("prospect_objecting", T_OFF)],
    ),
    (
        "price_asked_neutral",
        [("rep", "Each branch gets its own board and dispatcher view, with shared techs if you want.")],
        ("prospect", "Okay, this is interesting. What does something like this cost?"),
        [noul_ge("prospect_asked_price", T_ON), choice_is("stage", "pricing"), noul_lt("prospect_objecting", T_OFF), choice_is("next_move", "state_price_with_context", "ask_about_budget")],
    ),
    (
        "warm_opening",
        [],
        ("rep", "Ha, I know Tuesdays are hectic for dispatch, thanks for squeezing me in. How has your week been?"),
        [noul_lt("rep_pitching", T_OFF), noul_lt("next_step_proposed", T_OFF)],
    ),
]


async def _run_cases() -> tuple[list[dict[str, Any]], int, int]:
    judge = JevJudge()
    results: list[dict[str, Any]] = []
    passed = total = 0
    try:
        for case_id, extra, (speaker, text), checks in CASES:
            session = CallSession(judge)
            # build the window in code without calling Jev for the context turns
            t = 0.0
            for spk, txt in BASE_CONTEXT + extra:
                session.utterances.append(Utterance(spk, txt, t, len(session.utterances)))
                t += 10
            latest = Utterance(speaker, text, t, len(session.utterances))
            session.utterances.append(latest)
            state = session.build_state(latest)
            result = await judge(state, C.QUESTIONS)
            assert not result.error, f"{case_id}: {result.error}"
            a = result.answers
            case_checks = []
            for label, fn in checks:
                ok = bool(fn(a, state))
                passed += ok
                total += 1
                case_checks.append({"check": label, "ok": ok})
            results.append(
                {
                    "case": case_id,
                    "latest": f"{speaker}: {text}",
                    "checks": case_checks,
                    "latency_ms": round(result.latency_ms),
                    "input_tokens": result.input_tokens,
                    "observed": {
                        **{k: round(v["noul"], 2) for k, v in a.items() if v["type"] == "noul"},
                        "stage": f"{a['stage']['choice']} ({a['stage']['confidence']:.2f})",
                        "objection_type": f"{a['objection_type']['choice']} ({a['objection_type']['confidence']:.2f})",
                        "next_move": f"{a['next_move']['choice']} ({a['next_move']['confidence']:.2f})",
                        "urgency": round(a["urgency"]["score"], 2),
                        "engagement": round(a["engagement"]["score"], 2),
                    },
                }
            )
    finally:
        await judge.aclose()
    return results, passed, total


def test_labeled_utterances_pass_rate():
    results, passed, total = asyncio.run(_run_cases())
    rate = passed / total
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps({"model": C.MODEL, "passed": passed, "total": total, "pass_rate": round(rate, 4), "cases": results}, indent=2),
        encoding="utf-8",
    )
    failures = [(r["case"], c["check"]) for r in results for c in r["checks"] if not c["ok"]]
    print(f"\nintegration pass rate: {passed}/{total} = {rate:.1%}  (model {C.MODEL})")
    for case, check in failures:
        obs = next(r for r in results if r["case"] == case)["observed"]
        print(f"  FAIL {case}: {check}   observed={ {k: obs[k] for k in obs if k in check or k in ('stage','objection_type','next_move')} }")
    assert rate >= MIN_PASS_RATE, f"pass rate {rate:.1%} < {MIN_PASS_RATE:.0%}: {failures}"
