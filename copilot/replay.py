"""Load hand-written call transcripts (JSON or plain text) for replay and evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

CALLS_DIR = Path(__file__).resolve().parent.parent / "data" / "calls"

_LINE_RE = re.compile(r"^\s*(?:\[?(\d+(?:\.\d+)?)\]?\s*)?(rep|prospect)\s*:\s*(.+?)\s*$", re.IGNORECASE)


PREFERRED_ORDER = {"good": 0, "bad": 1, "mixed": 2}


def list_calls(calls_dir: Path = CALLS_DIR) -> list[dict[str, Any]]:
    out = []
    for p in sorted(calls_dir.glob("*.json"), key=lambda p: (PREFERRED_ORDER.get(p.stem, 99), p.stem)):
        call = load_call(p)
        out.append(
            {
                "id": call["id"],
                "title": call.get("title", call["id"]),
                "description": call.get("description", ""),
                "expected": call.get("expected", ""),
                "utterances": len(call["utterances"]),
                "duration_s": call["utterances"][-1]["t"] if call["utterances"] else 0,
                "video": call.get("video"),  # {"youtube_id": ...} when the call is a real recording
            }
        )
    return out


def load_call(path_or_id: str | Path, calls_dir: Path = CALLS_DIR) -> dict[str, Any]:
    """Load a call by id (`good`), by JSON path, or by plain-text path (`speaker: text` per line)."""
    # Resolve calls_dir once for security checks
    resolved_calls_dir = calls_dir.resolve()
    
    # Handle Path objects (from internal callers like list_calls)
    if isinstance(path_or_id, Path):
        # If it's already a Path, resolve and validate it's within calls_dir
        try:
            resolved_path = path_or_id.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise FileNotFoundError(f"Cannot resolve call path '{path_or_id}': {e}") from e
        
        # Security: Verify the resolved path is within calls_dir
        try:
            resolved_path.relative_to(resolved_calls_dir)
        except ValueError:
            raise ValueError(f"Path '{path_or_id}' is outside calls directory") from None
        
        path = resolved_path
    else:
        # Handle string identifiers (from user input)
        path_str = str(path_or_id)
        
        # Security: Reject absolute paths and path traversal attempts
        # Only allow simple identifiers or relative paths within calls_dir
        if path_str.startswith("/") or "\\" in path_str or ".." in path_str:
            raise ValueError(f"Invalid call identifier: '{path_or_id}' (absolute paths and traversal not allowed)")
        
        # Try as identifier first (e.g., "good" -> "good.json")
        candidate = calls_dir / f"{path_str}.json"
        if candidate.exists():
            path = candidate
        else:
            # Try as relative path within calls_dir (e.g., "subdir/call.json")
            candidate = calls_dir / path_str
            if candidate.exists():
                path = candidate
            else:
                raise FileNotFoundError(f"No call '{path_or_id}' (looked in {calls_dir})")
        
        # Security: Resolve path and verify it's within calls_dir
        try:
            resolved_path = path.resolve(strict=True)
        except (OSError, RuntimeError) as e:
            raise FileNotFoundError(f"Cannot resolve call path '{path_or_id}': {e}") from e
        
        # Security: Check that resolved path is under calls_dir (prevents symlink escapes)
        try:
            resolved_path.relative_to(resolved_calls_dir)
        except ValueError:
            raise ValueError(f"Path '{path_or_id}' resolves outside calls directory") from None
        
        path = resolved_path
    
    # Security: Verify it's a regular file (not a special file like /dev/zero)
    if not path.is_file():
        raise ValueError(f"Path '{path_or_id}' is not a regular file")
    
    # Security: Check file size to prevent DoS (10MB limit)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File '{path_or_id}' is too large ({file_size} bytes, max {MAX_FILE_SIZE})")
    
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            call = json.load(f)
        call.setdefault("id", path.stem)
        for i, u in enumerate(call["utterances"]):
            u.setdefault("t", float(i * 8))
            u["speaker"] = u["speaker"].lower()
        return call
    return parse_text_transcript(path.read_text(encoding="utf-8"), call_id=path.stem)


def parse_text_transcript(text: str, call_id: str = "transcript") -> dict[str, Any]:
    """`rep: ...` / `prospect: ...` per line; optional leading `[seconds]`. Missing times are spaced by word count."""
    utterances: list[dict[str, Any]] = []
    t = 0.0
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        ts, speaker, body = m.groups()
        if ts is not None:
            t = float(ts)
        utterances.append({"t": t, "speaker": speaker.lower(), "text": body})
        t += max(3.0, len(body.split()) / 2.5)
    return {"id": call_id, "title": call_id, "description": "", "expected": "", "utterances": utterances}
