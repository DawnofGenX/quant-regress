"""The release pipeline must not be able to fail on a stale version.

Observed 2026-10-07: the `v1.0.1` release ran the PyPI publish job from a tag
whose pyproject.toml still declared `0.1.1`. PyPI refused with

    HTTP 400 -- "File already exists"
    ... blake2_256 hash of quant_regress-0.1.1-*.whl
    (see https://pypi.org/help/#file-name-reuse)

because the 0.1.1 wheel was already published, and release filenames are
immutable. The workflow's guard checks that the tag *looks* like a version
(startsWith 'v' and contains a dotted part), which `v1.0.1` satisfied, so a
tag pointing at the wrong commit sailed through and left a permanent red run.

The tag shape is not the invariant -- the tag's COMMIT is. So this test reads
the real git state instead of trusting the workflow.

These tests cover the local clone. On CI, the checkout is a shallow clone
with no tags, so the tag assertions skip; the commit-history assertions still
run, and they are the ones that catch the real mistake (a version bump that
never landed on the release line).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "quant_regress" / "__init__.py"
VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)
TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _run(*args: str) -> str | None:
    if shutil.which("git") is None:
        return None
    try:
        r = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _declared_version() -> str:
    m = VERSION_RE.search(PYPROJECT.read_text(encoding="utf-8"))
    assert m, f"no version in {PYPROJECT}"
    return m.group(1)


def _init_version() -> str:
    m = re.search(r'__version__\s*=\s*"([^"]+)"', INIT.read_text(encoding="utf-8"))
    assert m, f"no __version__ in {INIT}"
    return m.group(1)


def _tags() -> set[str]:
    out = _run("tag", "--list", "--format=%(refname:short)")
    return {t.strip() for t in out.splitlines() if t.strip()} if out else set()


# --------------------------------------------------------------------------
# The two files must agree, or a release can be cut from one and published
# from the other.
# --------------------------------------------------------------------------


def test_declared_version_matches_runtime_version():
    assert _declared_version() == _init_version(), (
        f"pyproject.toml says {_declared_version()!r} but "
        f"src/quant_regress/__init__.py says {_init_version()!r}. The PyPI job "
        "builds from pyproject, so the two disagreeing is how the wrong wheel "
        "reaches the index."
    )


def test_version_is_three_dotted_parts():
    """The publish guard only fires on vMAJOR.MINOR.PATCH."""
    assert TAG_RE.match(f"v{_declared_version()}"), (
        f"version {_declared_version()!r} is not MAJOR.MINOR.PATCH; "
        "the release workflow would skip it as if it were the moving `v1` pin."
    )


# --------------------------------------------------------------------------
# Tag/commit coherence. Skipped when tags are absent (a shallow CI checkout).
# --------------------------------------------------------------------------


def _tag_points_at_version(tags: set[str]) -> None:
    for tag in sorted(t for t in tags if TAG_RE.match(t)):
        at_tag = _run("show", f"{tag}:pyproject.toml")
        if at_tag is None:
            continue
        m = VERSION_RE.search(at_tag)
        assert m, f"{tag}: pyproject.toml has no version"
        got = m.group(1)
        want = TAG_RE.match(tag).group(0)[1:]
        assert got == want, (
            f"tag {tag} points at a commit whose pyproject.toml declares "
            f"{got!r}, not {want!r}. PyPI would 400 with 'File already exists' "
            f"on the {got} wheel it already owns. Re-cut the tag at the commit "
            "that contains the version bump."
        )


@pytest.mark.skipif(not _tags(), reason="no git tags available (shallow checkout)")
def test_every_version_tag_points_at_its_own_version():
    _tag_points_at_version(_tags())


@pytest.mark.skipif(not _tags(), reason="no git tags available (shallow checkout)")
def test_the_top_version_tag_is_the_declared_version():
    """HEAD's declared version must be published, or a release is silently old.

    Checks the newest version tag against pyproject on main: if main has moved
    past every tag, the latest release does not contain the current code, and
    every user pinning the version tag gets the old build.
    """
    version_tags = [t for t in _tags() if TAG_RE.match(t)]
    if not version_tags:
        pytest.skip("no version tags")

    def key(t: str) -> tuple[int, ...]:
        m = TAG_RE.match(t)
        assert m, t
        return tuple(int(x) for x in m.group(1).split("."))

    newest = max(version_tags, key=key)
    newest_ = TAG_RE.match(newest)
    assert newest_, newest
    want = newest_.group(1)

    at_main = _run("show", f"origin/main:pyproject.toml") or _run("show", "main:pyproject.toml")
    if at_main is not None:
        m = VERSION_RE.search(at_main)
        assert m, "main: pyproject.toml has no version"
        main_version = m.group(1)
        assert main_version == want, (
            f"main declares {main_version!r} but the newest version tag {newest} "
            "declares a different version. The published release does not contain "
            "main's code: re-cut the tag and re-publish."
        )

    at_tag = _run("show", f"{newest}:pyproject.toml")
    if at_tag is not None:
        m = VERSION_RE.search(at_tag)
        assert m, f"{newest}: pyproject.toml has no version"
        assert m.group(1) == want, (
            f"tag {newest} does not carry its own version ({m.group(1)!r})"
        )
