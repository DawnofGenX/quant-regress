"""Eval-set loading for quant-regress.

Format: JSONL, one case per line, each ``{"id", "prompt", "expected"}``.
Comparison is exact-after-normalisation by design: a fuzzy scorer would hide
the marginal-decision damage this tool exists to detect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


class EvalSetError(ValueError):
    """Raised when an eval set is malformed. Always names the offending row."""


def normalise(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace."""
    return " ".join(str(text).split()).strip().lower()


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    expected: str


class EvalSet:
    def __init__(self, cases: Iterable[Case]):
        self.cases = list(cases)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    @classmethod
    def load(cls, path: str | Path) -> "EvalSet":
        p = Path(path)
        if not p.exists():
            raise EvalSetError(f"eval set not found: {p}")
        # A directory passes `exists()`, and `read_text()` on one raises
        # IsADirectoryError -- an uncaught OSError that escaped the CLI's error
        # handling and surfaced as exit 1, the code that means "accuracy
        # regressed". A configuration mistake impersonating a real regression
        # trains people to ignore the one exit code that matters, so check it
        # here where the message can be actionable.
        if p.is_dir():
            raise EvalSetError(
                f"eval set path is a directory, not a JSONL file: {p}. "
                f"Pass the path to the file itself."
            )
        if not p.is_file():
            raise EvalSetError(f"eval set is not a regular file: {p}")
        cases: list[Case] = []
        seen: set[str] = set()
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise EvalSetError(
                f"eval set is not valid UTF-8: {p}. "
                f"Fix the file encoding or re-save it as UTF-8."
            ) from exc
        for lineno, raw in enumerate(text.splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalSetError(f"line {lineno}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise EvalSetError(
                    f"line {lineno}: each row must be a JSON object (dict), "
                    f"got {type(row).__name__}. Example: "
                    f'{{"id": "1", "prompt": "...", "expected": "..."}}'
                )
            missing = [k for k in ("id", "prompt", "expected") if k not in row]
            if missing:
                raise EvalSetError(f"line {lineno}: missing {', '.join(missing)}")
            cid = str(row["id"])
            if cid in seen:
                raise EvalSetError(f"line {lineno}: duplicate id {cid!r}")
            seen.add(cid)
            cases.append(
                Case(id=cid, prompt=str(row["prompt"]), expected=normalise(row["expected"]))
            )
        if not cases:
            raise EvalSetError(f"eval set is empty: {p}")
        return cls(cases)
