"""Headless replay: print the closing-probability timeline for a transcript, optionally save JSON.

    uv run python -m copilot.cli --transcript data/calls/good.json
    uv run python -m copilot.cli --all --out eval/timelines.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from . import constants as C
from .engine import CallSession
from .jev_client import JevJudge
from .replay import list_calls, load_call


async def run_call(call: dict[str, Any], judge: JevJudge, quiet: bool = False) -> dict[str, Any]:
    session = CallSession(judge)
    rows = []
    if not quiet:
        print(f"\n== {call.get('title', call['id'])} ({len(call['utterances'])} utterances) ==")
        print(f"{'#':>3} {'t':>6} {'spk':<9} {'p':>5} {'inst':>5} {'stage':<18} {'move':<32} {'ms':>5}  text")
    for u in call["utterances"]:
        snap = await session.ingest(u["speaker"], u["text"], u["t"])
        coaching = snap["coaching"]
        move = "listening..." if coaching.get("gated") else coaching.get("title", "")
        rows.append(
            {
                "index": snap["index"],
                "t": snap["t"],
                "speaker": snap["speaker"],
                "text": snap["text"],
                "probability": snap["probability"],
                "instant": snap["instant"],
                "stage": snap["stage"]["choice"],
                "stage_confidence": snap["stage"]["confidence"],
                "objection": snap["objection"],
                "next_move": coaching.get("move_id"),
                "next_move_confidence": coaching.get("confidence"),
                "gated": coaching.get("gated"),
                "facts": sorted(snap["facts"]),
                "signals": {k: v["value"] for k, v in snap["signals"].items()},
                "sensitivity": snap["sensitivity"],
                "latency_ms": snap["jev_last"]["latency_ms"],
                "input_tokens": snap["jev_last"]["input_tokens"],
                "cost_usd": snap["jev_last"]["cost_usd"],
                "error": snap["jev_last"]["error"],
            }
        )
        if not quiet:
            bar = "#" * int(snap["probability"] * 20)
            print(
                f"{snap['index']:>3} {snap['t']:>6.0f} {snap['speaker']:<9} {snap['probability']*100:>4.0f}% {snap['instant']*100:>4.0f}% "
                f"{snap['stage']['choice']:<18} {move:<32} {snap['jev_last']['latency_ms']:>5.0f}  {bar:<20} {snap['text'][:60]}"
            )
    stats = session.latency_stats()
    summary = {
        "id": call["id"],
        "title": call.get("title", call["id"]),
        "expected": call.get("expected", ""),
        "final_probability": session.probability,
        "min_probability": min(r["probability"] for r in rows) if rows else None,
        "max_probability": max(r["probability"] for r in rows) if rows else None,
        "utterances": len(rows),
        "facts": sorted(session.facts),
        "model": session.last_model,
        "requests": session.requests,
        "errors": session.errors,
        "latency_mean_ms": stats["mean_ms"],
        "latency_p95_ms": stats["p95_ms"],
        "total_input_tokens": session.total_input_tokens,
        "total_output_tokens": session.total_output_tokens,
        "total_cost_usd": round(session.total_cost_usd, 6),
        "timeline": rows,
    }
    if not quiet:
        print(
            f"-- final p={session.probability*100:.0f}%  facts={sorted(session.facts)}  model={session.last_model}  "
            f"latency mean={stats['mean_ms']:.0f}ms p95={stats['p95_ms']:.0f}ms  tokens={session.total_input_tokens}  cost=${session.total_cost_usd:.4f}  errors={session.errors}"
        )
    return summary


async def main_async(args: argparse.Namespace) -> int:
    judge = JevJudge()
    try:
        calls = [load_call(c["id"]) for c in list_calls()] if args.all else [load_call(args.transcript)]
        results = [await run_call(c, judge, quiet=args.quiet) for c in calls]
    finally:
        await judge.aclose()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"model": C.MODEL, "calls": results}, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay a sales call through the copilot and print the probability timeline.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--transcript", help="call id (good/bad/mixed), .json path, or .txt path with 'rep:'/'prospect:' lines")
    g.add_argument("--all", action="store_true", help="run every call in data/calls")
    ap.add_argument("--out", help="write timelines JSON here (e.g. eval/timelines.json)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
