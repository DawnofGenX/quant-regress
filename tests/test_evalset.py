"""The eval set must load deterministically and reject malformed input."""
from __future__ import annotations

import json

import pytest

from quant_regress.evalset import EvalSet, EvalSetError


def _write(tmp_path, rows):
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_loads_rows_in_file_order(tmp_path):
    rows = [
        {"id": "a", "prompt": "x?", "expected": "yes"},
        {"id": "b", "prompt": "y?", "expected": "no"},
    ]
    es = EvalSet.load(_write(tmp_path, rows))
    assert [c.id for c in es.cases] == ["a", "b"]
    assert len(es) == 2


def test_normalises_expected_for_exact_comparison(tmp_path):
    rows = [{"id": "a", "prompt": "x?", "expected": "  YES  "}]
    es = EvalSet.load(_write(tmp_path, rows))
    assert es.cases[0].expected == "yes"


def test_rejects_row_missing_prompt(tmp_path):
    rows = [{"id": "a", "expected": "yes"}]
    with pytest.raises(EvalSetError, match="prompt"):
        EvalSet.load(_write(tmp_path, rows))


def test_rejects_duplicate_ids(tmp_path):
    rows = [
        {"id": "a", "prompt": "x?", "expected": "yes"},
        {"id": "a", "prompt": "y?", "expected": "no"},
    ]
    with pytest.raises(EvalSetError, match="duplicate"):
        EvalSet.load(_write(tmp_path, rows))


def test_rejects_empty_file(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(EvalSetError, match="empty"):
        EvalSet.load(p)


def test_rejects_a_directory_with_an_actionable_message(tmp_path):
    """A directory passes exists(); read_text() then raises IsADirectoryError.

    That OSError escaped the CLI's EvalSetError handler and surfaced as exit 1 --
    the code that means "accuracy regressed". A configuration mistake must not
    impersonate a real regression, so this must be rejected where the message
    can name the mistake.
    """
    d = tmp_path / "evals"
    d.mkdir()
    with pytest.raises(EvalSetError) as exc:
        EvalSet.load(d)
    msg = str(exc.value)
    assert "directory" in msg
    assert str(d) in msg, "the message must name the offending path"


def test_rejects_non_utf8_file_with_actionable_error(tmp_path):
    """A non-UTF-8 file must raise EvalSetError, not UnicodeDecodeError."""
    p = tmp_path / "bad.jsonl"
    p.write_bytes(b"\xff\xfe\x00\x01\n")
    with pytest.raises(EvalSetError, match="utf-8|decode|encoding"):
        EvalSet.load(p)


def test_rejects_non_dict_json_row(tmp_path):
    """A JSONL row that is a list/string/number must raise EvalSetError, not TypeError."""
    p = tmp_path / "bad.jsonl"
    p.write_text('["not", "a", "dict"]\n', encoding="utf-8")
    with pytest.raises(EvalSetError, match="dict|object|row"):
        EvalSet.load(p)


def test_rejects_a_non_regular_file(tmp_path):
    """A fifo exists() but read_text() on it blocks waiting for a writer.

    So this must be rejected BEFORE any read is attempted — otherwise a stray
    path (a named pipe, a device) hangs the gate forever instead of failing it.
    The check is on the path type, not on the read succeeding.
    """
    import os

    fifo = tmp_path / "pipe"
    try:
        os.mkfifo(fifo)
    except (AttributeError, OSError, NotImplementedError):
        pytest.skip("cannot create a fifo on this platform")

    # Bound the call: a regression that reaches read_text() would block here.
    # Run in a thread so a hang fails the test instead of wedging the suite.
    import threading

    box: dict[str, object] = {}

    def run():
        try:
            EvalSet.load(fifo)
            box["result"] = "loaded"
        except EvalSetError as exc:
            box["result"] = f"EvalSetError: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            box["result"] = f"{type(exc).__name__}: {exc}"

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), (
        "EvalSet.load blocked on a fifo — the path-type check must run before "
        "read_text(), which waits forever for a writer"
    )
    assert str(box.get("result", "")).startswith("EvalSetError"), box.get("result")
