"""Resuming runs after a Colab disconnect.

Scripts append one JSON row per finished unit of work (an eval condition, a
training checkpoint's metrics) to a JSON-lines file on Google Drive. On
restart they read it back and skip every row whose fingerprint already
exists, so re-running the same notebook cell picks up where it left off.

A fingerprint hashes every setting that determines a row's result, so a
changed config redoes the work instead of reusing stale rows. Settings that
don't change results (batch size of an eval pass, logging cadence) should be
left out by the caller.

Plain Python only, so it's unit-testable without torch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """Every complete row of a JSON-lines file ([] if it doesn't exist).

    A disconnect mid-write can leave a partial last line. That line is
    dropped *and cut from the file*, so the next append starts on a fresh
    line instead of gluing a new row onto the fragment. A malformed line
    anywhere else is a real problem, not a disconnect, and raises.
    """
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise ValueError(f"{path}: malformed line {i + 1} (not a truncated last line)") from None
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines[:i]) + ("\n" if i else ""))
    return rows


def append_jsonl_row(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def fingerprint(settings: dict[str, Any]) -> str:
    """Short, order-independent hash of a JSON-serializable settings dict."""
    blob = json.dumps(settings, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def completed_fingerprints(rows: list[dict[str, Any]]) -> set[str]:
    return {row["fingerprint"] for row in rows if "fingerprint" in row}
