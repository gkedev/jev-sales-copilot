"""Thin async wrapper around the TypeSafe SDK: one speculative fan-out request per utterance.

Returns plain dicts so the engine (and tests, via a mock) never depend on SDK types.
Never logs the API key. Logs model, usage, latency and request id per call.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from . import constants as C

log = logging.getLogger("copilot.jev")


def resolve_api_key() -> str | None:
    """TYPESAFE_API_KEY with fallback to JEV_API_KEY (never printed)."""
    for name in C.API_KEY_ENV_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


@dataclass
class JudgeResult:
    answers: dict[str, dict[str, Any]]
    model: str = C.MODEL
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    request_id: str | None = None
    cost_usd: float = 0.0
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# The engine only needs something with this signature; tests inject a mock.
Judge = Callable[[Any, Mapping[str, Any]], Awaitable[JudgeResult]]


def normalize_answers(raw: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """SDK answer objects -> plain dicts. Score probabilities keyed by int level."""
    out: dict[str, dict[str, Any]] = {}
    for qid, a in raw.items():
        if isinstance(a, Mapping):
            kind = a.get("type")
        elif hasattr(a, "noul"):
            kind = "noul"
        elif hasattr(a, "choice"):
            kind = "choice"
        elif hasattr(a, "score"):
            kind = "score"
        else:
            kind = None
        if kind == "noul":
            out[qid] = {"type": "noul", "noul": float(a.noul if hasattr(a, "noul") else a["noul"])}
        elif kind == "choice":
            probs = a.probabilities if hasattr(a, "probabilities") else a["probabilities"]
            out[qid] = {
                "type": "choice",
                "choice": str(a.choice if hasattr(a, "choice") else a["choice"]),
                "confidence": float(a.confidence if hasattr(a, "confidence") else a["confidence"]),
                "probabilities": {str(k): float(v) for k, v in probs.items()},
            }
        elif kind == "score":
            probs = a.probabilities if hasattr(a, "probabilities") else a["probabilities"]
            probs_int = {int(k): float(v) for k, v in probs.items()}
            out[qid] = {
                "type": "score",
                "score": float(a.score if hasattr(a, "score") else a["score"]),
                "confidence": float(a.confidence if hasattr(a, "confidence") else a["confidence"]),
                "probabilities": probs_int,
                "levels": max(probs_int) + 1 if probs_int else 0,
            }
    return out


def cost_for_tokens(input_tokens: int) -> float:
    return input_tokens / 1_000_000 * C.PRICE_PER_M_INPUT_TOKENS_USD


class JevJudge:
    """Async callable: (state, questions) -> JudgeResult. Reuses one HTTP client."""

    def __init__(self, api_key: str | None = None, model: str = C.MODEL, timeout: float = C.REQUEST_TIMEOUT_S):
        key = api_key or resolve_api_key()
        if not key:
            raise RuntimeError("No API key: set TYPESAFE_API_KEY (or JEV_API_KEY). See .env.example / run.sh.")
        self.model = model
        self._client = AsyncTypeSafeClient(
            api_key=key,
            model=model,
            timeout=timeout,
            retry=RetryPolicy(max_retries=2, backoff_initial=0.2, backoff_max=1.0, timeout=timeout * 2),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __call__(self, state: Any, questions: Mapping[str, Any]) -> JudgeResult:
        t0 = time.perf_counter()
        try:
            resp = await self._client.system_one(state, dict(questions))
        except Exception as exc:  # noqa: BLE001 - surface every failure to the UI, keep the loop alive
            latency = (time.perf_counter() - t0) * 1000
            log.warning("jev request failed after %.0f ms: %s", latency, type(exc).__name__)
            return JudgeResult(answers={}, latency_ms=latency, error=f"{type(exc).__name__}: {exc}")
        latency = (time.perf_counter() - t0) * 1000
        usage = resp.usage
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        result = JudgeResult(
            answers=normalize_answers(resp.answers),
            model=resp.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency,
            request_id=resp.request_id,
            cost_usd=cost_for_tokens(in_tok),
        )
        log.info(
            "jev model=%s latency=%.0fms in_tokens=%d out_tokens=%d cost=$%.6f request_id=%s",
            resp.model, latency, in_tok, out_tok, result.cost_usd, resp.request_id,
        )
        return result
