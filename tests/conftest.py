from __future__ import annotations

import os
from typing import Any, Mapping

import pytest

from copilot import constants as C
from copilot.jev_client import JudgeResult, resolve_api_key


def _noul(p: float) -> dict[str, Any]:
    return {"type": "noul", "noul": p}


def _choice(choice: str, confidence: float = 0.9, options: list[str] | None = None) -> dict[str, Any]:
    options = options or [choice]
    rest = (1 - 0.9) / max(1, len(options) - 1)
    probs = {o: (0.9 if o == choice else rest) for o in options}
    return {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probs}


def _score(level: float, levels: int = 3, confidence: float = 0.8) -> dict[str, Any]:
    probs = {i: 0.0 for i in range(levels)}
    lo = int(level)
    if lo >= levels - 1:
        probs[levels - 1] = 1.0
    else:
        probs[lo] = 1 - (level - lo)
        probs[lo + 1] = level - lo
    return {"type": "score", "score": level, "confidence": confidence, "probabilities": probs, "levels": levels}


def neutral_answers() -> dict[str, dict[str, Any]]:
    """A full answer set where nothing is happening (all nouls low, scores mid, stage=discovery)."""
    a: dict[str, dict[str, Any]] = {}
    for qid, q in C.QUESTIONS.items():
        if q["type"] == "noul":
            a[qid] = _noul(0.05)
        elif q["type"] == "score":
            a[qid] = _score(1.0, len(q["criteria"]))
        elif qid == "stage":
            a[qid] = _choice("discovery", 0.9, list(C.STAGES))
        elif qid == "objection_type":
            a[qid] = _choice("none", 0.9, list(C.OBJECTION_TYPES))
        elif qid == "next_move":
            a[qid] = _choice("dig_into_the_problem", 0.7, [m["id"] for m in C.PLAYBOOK])
        elif qid.startswith("phrasing::"):
            a[qid] = _choice("p0", 0.6, ["p0", "p1", "p2"])
    return a


class MockJudge:
    """Scripted judge: pops one override dict per call (merged over neutral answers); counts calls."""

    def __init__(self, script: list[dict[str, Any]] | None = None):
        self.script = list(script or [])
        self.calls = 0
        self.states: list[Any] = []

    async def __call__(self, state: Any, questions: Mapping[str, Any]) -> JudgeResult:
        self.calls += 1
        self.states.append(state)
        answers = neutral_answers()
        if self.script:
            answers.update(self.script.pop(0))
        return JudgeResult(answers=answers, model="mock", input_tokens=1000, output_tokens=10, latency_ms=1.0, cost_usd=0.000042)


@pytest.fixture
def mock_judge_factory():
    return MockJudge


@pytest.fixture
def helpers():
    return {"noul": _noul, "choice": _choice, "score": _score}


def requires_api_key():
    return pytest.mark.skipif(
        not resolve_api_key() or os.environ.get("COPILOT_SKIP_LIVE") == "1",
        reason="needs TYPESAFE_API_KEY (or JEV_API_KEY) and network",
    )
