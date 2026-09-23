"""Call engine: turns a stream of utterances + Jev answers into the live copilot state.

Everything numeric lives here (talk ratio, pace, EMA, composite, gating, fact persistence,
sensitivity). Jev only answers the atomic questions in `constants.QUESTIONS`.
"""

from __future__ import annotations

import copy
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import constants as C
from .jev_client import Judge, JudgeResult

# ---------------------------------------------------------------------------
# Pure helpers (unit-tested directly)
# ---------------------------------------------------------------------------


def word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def talk_ratio(utterances: Iterable["Utterance"]) -> tuple[float, int, int]:
    """Rep share of words over the given utterances -> (ratio, rep_words, prospect_words)."""
    rep = prospect = 0
    for u in utterances:
        n = word_count(u.text)
        if u.speaker == "rep":
            rep += n
        else:
            prospect += n
    total = rep + prospect
    return (rep / total if total else 0.5), rep, prospect


def pace_wpm(utterances: list["Utterance"], speaker: str, now_t: float, window_s: float = C.PACE_WINDOW_S) -> float:
    """Words per minute spoken by `speaker` in the trailing window ending at `now_t`."""
    start = now_t - window_s
    words = sum(word_count(u.text) for u in utterances if u.speaker == speaker and u.t >= start)
    elapsed = min(window_s, max(now_t - (utterances[0].t if utterances else now_t), 10.0))
    return words * 60.0 / elapsed if elapsed > 0 else 0.0


def ema(previous: float, instant: float, alpha: float = C.EMA_ALPHA) -> float:
    return alpha * instant + (1.0 - alpha) * previous


def normalize_score(answer: dict[str, Any]) -> float:
    """Score answer -> [0,1] using score/(levels-1)."""
    levels = answer.get("levels") or (max(answer["probabilities"]) + 1 if answer.get("probabilities") else 2)
    return clamp(answer["score"] / max(1, levels - 1))


def composite(
    features: dict[str, float],
    stage: str,
    weights: dict[str, dict[str, Any]] | None = None,
    stage_prior: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Instantaneous closing probability + per-signal contributions (all in probability points)."""
    weights = weights or C.SIGNAL_WEIGHTS
    stage_prior = stage_prior or C.STAGE_PRIOR
    contributions: dict[str, float] = {"stage": stage_prior.get(stage, C.INITIAL_PROBABILITY)}
    for name, spec in weights.items():
        x = features.get(name)
        if x is None:
            continue
        w = float(spec["weight"])
        if spec["kind"] == "score":
            contributions[name] = w * (x - 0.5) * 2.0
        else:
            contributions[name] = w * x
    inst = clamp(sum(contributions.values()), C.P_MIN, C.P_MAX)
    return inst, contributions


def sensitivity(prev: dict[str, float], curr: dict[str, float], top_n: int = C.SENSITIVITY_TOP_N) -> list[dict[str, Any]]:
    """Top signals by absolute change in contribution between two utterances."""
    deltas = []
    for name in set(prev) | set(curr):
        d = curr.get(name, 0.0) - prev.get(name, 0.0)
        if abs(d) >= 0.0005:
            label = "Call stage" if name == "stage" else C.SIGNAL_WEIGHTS.get(name, {}).get("label", name)
            deltas.append({"signal": name, "label": label, "delta": round(d, 4)})
    deltas.sort(key=lambda d: (-abs(d["delta"]), -d["delta"], d["signal"]))  # deterministic on ties (set order is randomized)
    return deltas[:top_n]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def format_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class Utterance:
    speaker: str  # "rep" | "prospect"
    text: str
    t: float  # seconds since call start
    index: int = 0

    def to_state(self) -> dict[str, Any]:
        return {"t": format_ts(self.t), "speaker": self.speaker, "text": self.text}


@dataclass
class Objection:
    open: bool = False
    type: str | None = None
    confidence: float = 0.0
    since: int | None = None
    probability: float = 0.0


@dataclass
class Step:
    """Everything recorded for one utterance (raw answers kept so weights can be re-applied without Jev)."""

    utterance: Utterance
    answers: dict[str, dict[str, Any]]
    features: dict[str, float]
    stage: str
    stage_confidence: float
    instant: float
    probability: float
    contributions: dict[str, float]
    objection: Objection
    facts: dict[str, int]
    jev: dict[str, Any]
    talk: dict[str, Any]
    coaching: dict[str, Any]
    sensitivity: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class CallSession:
    def __init__(self, judge: Judge, weights: dict[str, dict[str, Any]] | None = None, questions: dict[str, Any] | None = None):
        self.judge = judge
        self.questions = questions or C.QUESTIONS
        self.weights: dict[str, dict[str, Any]] = copy.deepcopy(weights or C.SIGNAL_WEIGHTS)
        self.stage_prior: dict[str, float] = dict(C.STAGE_PRIOR)
        self.reset()

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self.utterances: list[Utterance] = []
        self.steps: list[Step] = []
        self.probability = C.INITIAL_PROBABILITY
        self.facts: dict[str, int] = {}  # fact name -> utterance index when it became durable
        self.objection = Objection()
        self.stage_history: list[str] = []
        self.latencies_ms: list[float] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.requests = 0
        self.errors = 0
        self.last_model: str | None = None

    # -- state for Jev --------------------------------------------------------
    def build_state(self, latest: Utterance) -> dict[str, Any]:
        window = self.utterances[-C.RECENT_WINDOW :]
        ratio, _, _ = talk_ratio(self.utterances)
        return {
            "call_facts": {
                "duration_min": round(latest.t / 60.0, 1),
                "talk_ratio_rep": round(ratio, 2),
                "stage_history": _compress(self.stage_history)[-6:],
                "known_facts": sorted(self.facts),
                "objection": (
                    {"open": True, "type": self.objection.type or "unknown"} if self.objection.open else {"open": False}
                ),
            },
            "recent_transcript": [u.to_state() for u in window],
            "latest_utterance": latest.to_state(),
        }

    # -- main entry -----------------------------------------------------------
    async def ingest(self, speaker: str, text: str, t: float | None = None) -> dict[str, Any]:
        if t is None:
            t = (self.utterances[-1].t + max(2.0, word_count(text) / 2.5)) if self.utterances else 0.0
        u = Utterance(speaker=speaker, text=text.strip(), t=float(t), index=len(self.utterances))
        self.utterances.append(u)
        state = self.build_state(u)
        result = await self.judge(state, self.questions)
        self._account(result)
        step = self._apply(u, result.answers, result_meta(result))
        self.steps.append(step)
        return self.snapshot(step)

    def _account(self, result: JudgeResult) -> None:
        self.requests += 1
        if result.error:
            self.errors += 1
            return
        self.latencies_ms.append(result.latency_ms)
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        self.total_cost_usd += result.cost_usd
        self.last_model = result.model

    # -- feature extraction (masking, persistence, code signals) --------------
    def extract_features(self, u: Utterance, answers: dict[str, dict[str, Any]], facts: dict[str, int], objection_open: bool) -> dict[str, float]:
        f: dict[str, float] = {}
        for name, spec in self.weights.items():
            kind = spec["kind"]
            if kind in ("noul", "fact"):
                a = answers.get(name)
                x = float(a["noul"]) if a and a.get("type") == "noul" else 0.0
                if name in C.PROSPECT_ONLY_SIGNALS and u.speaker != "prospect":
                    x = 0.0
                if name in C.REP_ONLY_SIGNALS and u.speaker != "rep":
                    x = 0.0
                if kind == "fact" and name in facts:
                    x = 1.0
                f[name] = x
            elif kind == "score":
                a = answers.get(name)
                f[name] = normalize_score(a) if a and a.get("type") == "score" else 0.5
        f["objection_open"] = 1.0 if objection_open else 0.0
        f["rep_talking_too_much"] = self._rep_talking_too_much(u)
        return f

    def _rep_talking_too_much(self, u: Utterance) -> float:
        if len(self.utterances) < C.TALK_RATIO_MIN_UTTERANCES:
            ratio_part = 0.0
        else:
            ratio, _, _ = talk_ratio(self.utterances)
            ratio_part = clamp((ratio - C.TALK_RATIO_WARN) / (C.TALK_RATIO_MAX - C.TALK_RATIO_WARN))
        monologue = u.speaker == "rep" and word_count(u.text) > C.REP_MONOLOGUE_WORDS
        return max(ratio_part, 0.6 if monologue else 0.0)

    def _update_facts(self, u: Utterance, answers: dict[str, dict[str, Any]]) -> None:
        for name, spec in self.weights.items():
            if spec["kind"] != "fact" or name in self.facts:
                continue
            a = answers.get(name)
            if not a or a.get("type") != "noul":
                continue
            if name in C.PROSPECT_ONLY_SIGNALS and u.speaker != "prospect":
                continue
            if a["noul"] >= C.FACT_PERSIST_THRESHOLD:
                self.facts[name] = u.index
        # competitor is context (not weighted as a fact) but worth remembering for Jev
        comp = answers.get("competitor_mentioned")
        if comp and comp.get("noul", 0) >= C.FACT_PERSIST_THRESHOLD and "competitor_mentioned" not in self.facts:
            self.facts["competitor_mentioned"] = u.index

    def _update_objection(self, u: Utterance, answers: dict[str, dict[str, Any]]) -> None:
        def p(name: str) -> float:
            a = answers.get(name)
            return float(a["noul"]) if a and a.get("type") == "noul" else 0.0

        obj_type = answers.get("objection_type")
        if u.speaker == "prospect":
            objecting = p("prospect_objecting")
            if objecting >= C.OBJECTION_OPEN_THRESHOLD:
                if not self.objection.open:
                    self.objection = Objection(open=True, since=u.index)
                self.objection.probability = objecting
                if obj_type and obj_type["choice"] != "none" and obj_type["confidence"] >= C.OBJECTION_TYPE_MIN_CONFIDENCE:
                    self.objection.type = obj_type["choice"]
                    self.objection.confidence = obj_type["confidence"]
                elif not self.objection.type:
                    self.objection.type = "unknown"
                return
            clearing = max(p("prospect_accepts"), p("buying_signal"), p("commitment"), p("next_step_agreed"))
            # "Sure, okay, I have to run" is not acceptance: a disengaging turn never clears an objection
            if self.objection.open and clearing >= C.OBJECTION_CLEAR_THRESHOLD and p("prospect_disengaging") < C.OBJECTION_CLEAR_THRESHOLD:
                self.objection = Objection()
                return
        if self.objection.open and self.objection.since is not None and u.index - self.objection.since > C.OBJECTION_MAX_AGE_UTTERANCES:
            self.objection = Objection()

    def _coaching(self, u: Utterance, answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
        nm = answers.get("next_move")
        if not nm or nm.get("type") != "choice":
            return {"gated": True, "reason": "no answer"}
        move = C.PLAYBOOK_BY_ID.get(nm["choice"])
        gated = nm["confidence"] < C.NEXT_MOVE_MIN_CONFIDENCE or move is None
        best = None
        if move:
            ph = answers.get(f"phrasing::{move['id']}")
            if ph and ph.get("type") == "choice" and ph["confidence"] >= C.PHRASING_MIN_CONFIDENCE:
                idx = int(ph["choice"][1:]) if ph["choice"][1:].isdigit() else 0
                best = move["phrasings"][idx] if idx < len(move["phrasings"]) else None
        return {
            "gated": gated,
            "move_id": move["id"] if move else nm["choice"],
            "title": move["title"] if move else nm["choice"],
            "what": move["what"] if move else "",
            "phrasings": move["phrasings"] if move else [],
            "best_phrasing": best,
            "confidence": round(nm["confidence"], 3),
            "probabilities": {k: round(v, 3) for k, v in sorted(nm["probabilities"].items(), key=lambda kv: -kv[1])[:4]},
        }

    def _apply(self, u: Utterance, answers: dict[str, dict[str, Any]], jev_meta: dict[str, Any]) -> Step:
        # stage
        st = answers.get("stage")
        if st and st.get("type") == "choice":
            stage, stage_conf = st["choice"], st["confidence"]
        else:
            stage, stage_conf = (self.stage_history[-1] if self.stage_history else "opening"), 0.0
        self.stage_history.append(stage)
        # durable facts and objection state first (they feed the features)
        self._update_facts(u, answers)
        self._update_objection(u, answers)
        features = self.extract_features(u, answers, self.facts, self.objection.open)
        inst, contributions = composite(features, stage, self.weights, self.stage_prior)
        self.probability = ema(self.probability, inst)
        prev_contrib = self.steps[-1].contributions if self.steps else {}
        ratio, rep_w, pro_w = talk_ratio(self.utterances)
        talk = {
            "ratio_rep": round(ratio, 3),
            "rep_words": rep_w,
            "prospect_words": pro_w,
            "pace_wpm_rep": round(pace_wpm(self.utterances, "rep", u.t)),
            "pace_wpm_prospect": round(pace_wpm(self.utterances, "prospect", u.t)),
            "rep_monologue": u.speaker == "rep" and word_count(u.text) > C.REP_MONOLOGUE_WORDS,
        }
        return Step(
            utterance=u,
            answers=answers,
            features=features,
            stage=stage,
            stage_confidence=stage_conf,
            instant=inst,
            probability=self.probability,
            contributions=contributions,
            objection=copy.copy(self.objection),
            facts=dict(self.facts),
            jev=jev_meta,
            talk=talk,
            coaching=self._coaching(u, answers),
            sensitivity=sensitivity(prev_contrib, contributions),
        )

    # -- weights can change without re-asking Jev -----------------------------
    def set_weights(self, updates: dict[str, float]) -> None:
        for name, w in updates.items():
            if name in self.weights:
                self.weights[name]["weight"] = float(w)

    def recompute(self) -> list[float]:
        """Re-run composite+EMA over stored raw answers with the current weights (no Jev calls)."""
        p = C.INITIAL_PROBABILITY
        prev_contrib: dict[str, float] = {}
        for step in self.steps:
            # features depend on facts/objection state as they were at that step; re-derive from stored data
            features = dict(step.features)
            inst, contributions = composite(features, step.stage, self.weights, self.stage_prior)
            p = ema(p, inst)
            step.instant, step.probability, step.contributions = inst, p, contributions
            step.sensitivity = sensitivity(prev_contrib, contributions)
            prev_contrib = contributions
        self.probability = p
        return [s.probability for s in self.steps]

    # -- views ----------------------------------------------------------------
    def latency_stats(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {"mean_ms": 0.0, "p95_ms": 0.0, "last_ms": 0.0}
        return {
            "mean_ms": round(statistics.fmean(self.latencies_ms), 1),
            "p95_ms": round(percentile(self.latencies_ms, 0.95), 1),
            "last_ms": round(self.latencies_ms[-1], 1),
        }

    def weights_table(self) -> list[dict[str, Any]]:
        return [
            {"signal": name, "label": spec["label"], "kind": spec["kind"], "weight": spec["weight"]}
            for name, spec in self.weights.items()
        ]

    def timeline(self) -> list[dict[str, Any]]:
        return [
            {"index": s.utterance.index, "t": s.utterance.t, "speaker": s.utterance.speaker, "probability": round(s.probability, 4), "instant": round(s.instant, 4), "stage": s.stage}
            for s in self.steps
        ]

    def snapshot(self, step: Step | None = None) -> dict[str, Any]:
        step = step or (self.steps[-1] if self.steps else None)
        base: dict[str, Any] = {
            "type": "update",
            "probability": round(self.probability, 4),
            "timeline": self.timeline(),
            "facts": {k: {"since": v} for k, v in self.facts.items()},
            "objection": {
                "open": self.objection.open,
                "type": self.objection.type,
                "confidence": round(self.objection.confidence, 3),
                "since": self.objection.since,
            },
            "stage_history": self.stage_history,
            "weights": self.weights_table(),
            "stage_prior": self.stage_prior,
            "jev": {
                "model": self.last_model,
                "requests": self.requests,
                "errors": self.errors,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "cum_cost_usd": round(self.total_cost_usd, 6),
                **self.latency_stats(),
            },
        }
        if step is None:
            return base
        signals = {}
        for name, spec in self.weights.items():
            x = step.features.get(name, 0.0)
            raw = step.answers.get(name)
            signals[name] = {
                "label": spec["label"],
                "kind": spec["kind"],
                "value": round(x, 3),
                "on": (x >= C.SIGNAL_ON_THRESHOLD) if spec["kind"] != "score" else None,
                "level": (round(raw["score"]) if raw and raw.get("type") == "score" else None),
                "persisted": name in step.facts,
                "contribution": round(step.contributions.get(name, 0.0), 4),
            }
        base.update(
            {
                "index": step.utterance.index,
                "t": step.utterance.t,
                "speaker": step.utterance.speaker,
                "text": step.utterance.text,
                "instant": round(step.instant, 4),
                "stage": {"choice": step.stage, "confidence": round(step.stage_confidence, 3)},
                "signals": signals,
                "coaching": step.coaching,
                "talk": step.talk,
                "sensitivity": step.sensitivity,
                "contributions": {k: round(v, 4) for k, v in step.contributions.items()},
                "jev_last": step.jev,
            }
        )
        return base


def result_meta(result: JudgeResult) -> dict[str, Any]:
    return {
        "model": result.model,
        "latency_ms": round(result.latency_ms, 1),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": round(result.cost_usd, 6),
        "request_id": result.request_id,
        "error": result.error,
    }


def _compress(seq: list[str]) -> list[str]:
    """Collapse consecutive duplicates: [a,a,b,b,a] -> [a,b,a]."""
    out: list[str] = []
    for s in seq:
        if not out or out[-1] != s:
            out.append(s)
    return out
