"""Every headline number in README.md must be re-derivable by a committed script.

This repo makes claims in other people's CI. A stale or fabricated number here
propagates into other people's build failures, so the claims are treated as
code: each one names the script that regenerates it, and this test fails if a
claim has no regenerating script.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
SCRIPTS = ROOT / "scripts"

CLAIM_RE = re.compile(r"<!--\s*claim:(?P<id>[a-z0-9-]+)\s*-->")
PERCENT_RE = re.compile(r"\b\d{1,3}(?:\.\d+)?%")


@pytest.mark.skipif(not README.exists(), reason="README not written yet")
def test_every_percent_in_readme_is_tagged_as_a_claim():
    """Untagged percentages are the ones that can silently rot."""
    text = README.read_text(encoding="utf-8")
    claimed = {m.group("id") for m in CLAIM_RE.finditer(text)}
    for line_no, line in enumerate(text.splitlines(), 1):
        if PERCENT_RE.search(line) and "claim:" not in line:
            # Badge lines (shields.io) are generated, not claims.
            if "img.shields.io" in line:
                continue
            pytest.fail(
                f"README.md:{line_no} has an untagged percentage: {line.strip()!r}. "
                "Add <!-- claim:ID --> and a script in scripts/ that regenerates it."
            )
    assert claimed or not PERCENT_RE.search(text)


@pytest.mark.skipif(not README.exists(), reason="README not written yet")
def test_every_claim_has_a_regenerating_script():
    text = README.read_text(encoding="utf-8")
    for m in CLAIM_RE.finditer(text):
        script = SCRIPTS / f"regen_{m.group('id').replace('-', '_')}.py"
        assert script.exists(), (
            f"README claim {m.group('id')!r} has no regenerating script at "
            f"{script.relative_to(ROOT)}"
        )


@pytest.mark.skipif(not README.exists(), reason="README not written yet")
def test_readme_states_cpu_only_and_threshold_semantics():
    text = README.read_text(encoding="utf-8").lower()
    assert "cpu" in text, "README must state the CPU-only constraint"
    assert "point" in text, "README must state the threshold unit"
