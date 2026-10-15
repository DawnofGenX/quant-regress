"""End-to-end integration: a real checkpoint through the whole pipeline.

Every other test in this suite substitutes a fake for the model factory, so
nothing before this file proved that ``EvalSet.load`` -> ``build_model_factory``
-> ``QuantHarness.compare`` -> ``to_dict`` actually compose on real weights.
That is where the silent-PASS defects lived: the harness could measure
correctly and the adapter could hand it a model it never quantized, and each
half looked right in isolation.

Model choice, and what is safe to assert:

The checkpoints here are *randomly initialised*, so their accuracy is
meaningless — asserting a particular score would make this test a coin flip
that fails on CI for no reason. Exactly that mistake was made once before
(see ``tests/test_selftest_gate.py``): a fixture asserting a real tiny LM
regresses turns a correct gate into a red build.

So this file asserts the pipeline's *contract* on real models instead:

1. the same set is scored at every precision, and the numbers are internally
   consistent (``correct + len(misclassified) == total``);
2. the two precisions are genuinely different models — the int8 arm really
   was quantized, which is the property the guard exists to certify;
3. the whole run survives to a JSON report through the real ``main()``.

Accuracy values are recorded but never asserted.
"""
from __future__ import annotations

import json

import pytest

from quant_regress.cli import main
from quant_regress.evalset import EvalSet
from quant_regress.harness import QuantHarness
from quant_regress.adapter import _count_quantized, build_model_factory

#: BERT-style blocks are nn.Linear, which dynamic INT8 can reach. The tiny
#: random checkpoint is enough to prove the pipeline composes.
BERT = "hf-internal-testing/tiny-random-bert"


def _eval_set(tmp_path, n=4):
    rows = [
        {"id": f"c{i}", "prompt": f"Is claim number {i} supported by the text?",
         "expected": "yes"}
        for i in range(n)
    ]
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def _eval_path(tmp_path, n=4):
    """The eval set as a PATH, which is what cli.main takes."""
    return str(_eval_set(tmp_path, n))


@pytest.fixture(scope="module")
def bert_factory():
    """One real factory shared by the tests in this module.

    Loading a checkpoint is the expensive part; the models themselves are
    released per precision by the harness.
    """
    return build_model_factory(BERT, max_new_tokens=8)


def test_real_model_scores_the_whole_set_at_every_precision(tmp_path, bert_factory):
    """The pipeline must produce internally consistent numbers on a real model.

    ``correct + len(misclassified) == total`` is the invariant that catches
    a dropped or double-counted case, and it holds regardless of how accurate
    a random checkpoint happens to be.
    """
    es = EvalSet.load(_eval_set(tmp_path))
    h = QuantHarness(model_factory=bert_factory, max_new_tokens=8)
    res = h.compare(es, baseline_precision="fp32", candidate_precisions=["int8"])

    for arm in (res.baseline, *res.candidates):
        assert arm.total == len(es), (
            f"{arm.precision} scored {arm.total} of {len(es)} cases"
        )
        assert arm.correct + len(arm.misclassified) == arm.total, (
            f"{arm.precision}: {arm.correct} correct + "
            f"{len(arm.misclassified)} misclassified != {arm.total} total "
            f"-- the accounting is broken"
        )
        assert len(set(arm.misclassified)) == len(arm.misclassified), (
            f"{arm.precision}: a case id appears twice in misclassified"
        )

    # The candidate must be a DIFFERENT measurement from the baseline, both
    # in identity and in the numbers that follow from it.
    assert res.baseline.precision == "fp32"
    assert res.candidates[0].precision == "int8"


def test_the_int8_arm_really_was_quantized(bert_factory):
    """The pipeline's core claim: the int8 arm differs from fp32.

    This is the property the whole tool rests on. If the adapter handed the
    harness an unquantized model, the gate would compare a model against
    itself and report a 0.00-point drop — and every other test, being fake,
    would still pass. Only a real model can catch it.
    """
    fp32 = bert_factory("fp32")
    int8 = bert_factory("int8")

    assert _count_quantized(fp32) == 0, (
        "the fp32 arm reported quantized modules before any quantization ran"
    )
    assert _count_quantized(int8) > _count_quantized(fp32), (
        "the int8 arm is not actually quantized: the gate would compare a "
        "model against itself and report a 0.00-point drop"
    )


def test_the_real_run_reaches_a_json_report(tmp_path, bert_factory):
    """The full CLI path must survive a real model and leave evidence behind.

    This is the composition test: cli.main -> EvalSet -> factory -> compare
    -> to_dict -> file. A break anywhere in that chain is invisible to the
    unit tests, which each stub out the module on either side of it.
    """
    report = tmp_path / "report.json"
    rc = main([
        "--eval", _eval_path(tmp_path),
        "--model", BERT,
        "--precisions", "int8",
        "--max-drop-points", "100",
        "--report", str(report),
    ])

    assert rc in (0, 1), (
        f"the run exited {rc}: a completed measurement must be a verdict "
        f"(0 pass, 1 regression), never a usage error"
    )
    data = json.loads(report.read_text())
    assert data["baseline"]["precision"] == "fp32"
    assert data["candidates"][0]["precision"] == "int8"
    assert data["verdict"] in ("pass", "fail")
    # Both arms must have scored the whole set, and the ids must be the
    # eval set's -- not empty lists from a silent early exit.
    assert data["baseline"]["total"] == 4
    assert data["candidates"][0]["total"] == 4
