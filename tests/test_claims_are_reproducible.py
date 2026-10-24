"""Every headline number in README.md must be re-derivable by a committed script.

This repo makes claims in other people's CI. A stale or fabricated number here
propagates into other people's build failures, so the claims are treated as
code: each one names the script that regenerates it, and this test fails if a
claim has no regenerating script.
"""
from __future__ import annotations

import re
import subprocess
import sys
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


# ---------------------------------------------------------------------------
# The load-bearing half: run the script and compare the numbers.
#
# Until these tests existed, "every figure comes from that script and the test
# fails if a number stops matching" was a promise in the README that nothing
# enforced: the checks above only prove a file with the right name exists. A
# number could be edited to anything and the suite stayed green.
# ---------------------------------------------------------------------------

TOKEN_RES = (
    re.compile(r"\b\d{1,3}(?:\.\d+)?%"),        # 100.0%
    re.compile(r"\b\d+/\d+\b"),                  # 20/20
    re.compile(r"[+-]\d+\.\d+ pts"),             # +100.0 pts
    re.compile(r"\b(?:PASS|FAIL)\b"),            # verdicts
)


def _claim_ids() -> list[str]:
    return sorted({m.group("id") for m in CLAIM_RE.finditer(README.read_text(encoding="utf-8"))})


def _script_for(claim_id: str) -> Path:
    return SCRIPTS / f"regen_{claim_id.replace('-', '_')}.py"


def _run_script(script: Path) -> str:
    """Run a regenerating script and return its stdout.

    The scripts are self-contained and CPU-only by design, so a test can afford to
    execute them. If one ever becomes slow enough to be impractical here, the right
    fix is to make the script cheaper, not to stop checking the number.
    """
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"{script.relative_to(ROOT)} failed ({proc.returncode}):\n{proc.stderr[-1500:]}"
    )
    return proc.stdout


def _tokens(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in TOKEN_RES:
        found.update(m.group(0) for m in pattern.finditer(text))
    return found


@pytest.mark.skipif(not README.exists(), reason="README not written yet")
def test_every_tagged_number_is_produced_by_its_regenerating_script():
    """Every number sitting on a claim-marked README line must appear in the script's output.

    This is the check that makes the README's promise true. Mutating a figure to a
    number the script never prints now fails the build.
    """
    text = README.read_text(encoding="utf-8")
    claim_ids = _claim_ids()
    assert claim_ids, "README has no claim markers, so this test would be vacuous"

    for claim_id in claim_ids:
        script = _script_for(claim_id)
        printed = _tokens(_run_script(script))

        marker = f"claim:{claim_id}"
        marked_lines = [line for line in text.splitlines() if marker in line]
        checked = 0
        for line in marked_lines:
            visible = CLAIM_RE.sub("", line)
            missing = _tokens(visible) - printed
            if missing:
                pytest.fail(
                    f"README claims {sorted(missing)} under {marker!r}, but "
                    f"{script.relative_to(ROOT)} never prints them. Either the README "
                    f"figure drifted from the script, or the script stopped producing it.\n"
                    f"  README line: {line.strip()!r}\n"
                    f"  script output: {sorted(printed)!r}"
                )
            checked += 1
        assert checked, f"claim {claim_id!r} is marked but has no line to check"


VOLATILE_KEYS = {"seconds"}  # wall-clock, not a measurement


def _canonical(blob: bytes) -> object:
    """JSON with wall-clock fields removed, so two runs are comparable."""
    import json

    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in VOLATILE_KEYS}
        if isinstance(node, list):
            return [strip(x) for x in node]
        return node

    return strip(json.loads(blob))


def _committed_blob(path: Path) -> bytes | None:
    """The file as committed at HEAD, so a dirty tree cannot mask a stale artifact."""
    proc = subprocess.run(
        ["git", "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"],
        cwd=str(ROOT), capture_output=True,
    )
    return proc.stdout if proc.returncode == 0 else None


@pytest.mark.skipif(not README.exists(), reason="README not written yet")
def test_committed_results_are_exactly_what_a_fresh_run_produces():
    """A committed results/*.json must match regenerating it from the current code.

    Otherwise the auditable artifact in the repo describes a past machine, not the
    code in front of you — which is exactly the state this test was written to catch:
    ``results/int8_collapse.json`` had drifted behind the harness (missing
    ``misclassified`` entries) while CI's regen step overwrote it without diffing.
    The working tree is restored afterwards, so running the suite never dirties a repo.
    """
    results_dir = ROOT / "results"
    if not results_dir.is_dir():
        pytest.skip("no results/ directory")

    before = {p: p.read_bytes() for p in results_dir.glob("*.json")}
    assert before, "results/ has no committed JSON to verify"

    drifted: list[str] = []
    try:
        for claim_id in _claim_ids():
            _run_script(_script_for(claim_id))
        for path in before:
            fresh = _canonical(path.read_bytes())
            reference = _committed_blob(path)
            stale = _canonical(reference if reference is not None else before[path])
            if fresh != stale:
                drifted.append(str(path.relative_to(ROOT)))
    finally:
        for path, blob in before.items():
            path.write_bytes(blob)

    assert not drifted, (
        "regenerating produced different results than committed: " + ", ".join(drifted)
        + " — re-run the script and commit it, so the artifact matches the code"
    )

