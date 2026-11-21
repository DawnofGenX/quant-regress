"""Independent lm-eval perplexity baseline for the INT8 collapse claim.

The repo's task-level accuracy curves show a large fp32 -> dynamic-INT8 drop
on arithmetic and sentiment.  A fair objection is: "maybe perplexity/KL would
have caught that too, so your task set is just unusually fragile."  This
script answers that objection with a real lm-eval run on a real corpus
(``wikitext``) using the same Qwen2.5-0.5B-Instruct checkpoint: first fp32,
then ``torch.quantize_dynamic(..., {torch.nn.Linear}, dtype=torch.qint8)``.

Both arms are written to results/ and the delta is judged by
``quant_regress.lm_eval_baseline.evaluate_delta``.  A delta >= 2.0 points
means the token-level proxy DID see the collapse: any README/CV claim that
"perplexity and KL still report the model as fine" must then be downgraded.
A smaller delta means perplexity missed it and the claim stands, with this
measurement as explicit evidence.

Run (CPU-only; the HF dataset is cached locally):
    HF_HUB_OFFLINE=1 .venv/bin/python scripts/regen_lm_eval_baseline.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import lm_eval  # noqa: E402
import torch  # noqa: E402
from torch.ao.quantization import quantize_dynamic  # noqa: E402
from lm_eval import simple_evaluate  # noqa: E402
from lm_eval.models.huggingface import HFLM  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from quant_regress.lm_eval_baseline import (  # noqa: E402
    DEFAULT_THRESHOLD,
    RESULT_SCHEMA,
    evaluate_delta,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TASK = "wikitext"


def perplexity_of(results: dict, task: str) -> float:
    """Pull the word-perplexity metric out of lm-eval's result dictionary."""
    metrics = results["results"][task]
    key = next(
        (name for name in metrics if name.startswith("word_perplexity,")), None
    )
    if key is None:
        raise KeyError(
            f"{task}: no word_perplexity metric in {sorted(metrics)}; "
            "this baseline needs a loglikelihood_rolling task"
        )
    return float(metrics[key])


def recorded_dtype(model) -> str:
    """Report the dtype actually used by the arm's Linear layers.

    A dynamic-quantized Linear stores packed int8 weights, so walking
    ``model.parameters()`` still reports float32 for the surviving embeddings.
    Check the module types instead; an arm labelled int8 must really contain
    quantized Linear layers.
    """
    if any(
        isinstance(module, torch.ao.nn.quantized.dynamic.Linear)
        for module in model.modules()
    ):
        return "torch.qint8"
    return str(next(model.parameters()).dtype)


def run_arm(model, tokenizer, task: str, limit: int, dtype: str) -> float:
    """Evaluate one already-built model object through HFLM on CPU."""
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        device="cpu",
        dtype=dtype,
        batch_size=1,
    )
    results = simple_evaluate(
        model=lm,
        tasks=[task],
        limit=limit,
        bootstrap_iters=0,
        log_samples=False,
    )
    return perplexity_of(results, task)


def write_result(path: Path, *, task: str, precision: str, model: str,
                 perplexity: float, num_samples: int, measured_dtype: str) -> None:
    payload = {
        "task": task,
        "precision": precision,
        "model": model,
        "perplexity": perplexity,
        "num_samples": num_samples,
        "measured_dtype": measured_dtype,
        "lm_eval_version": lm_eval.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    assert tuple(payload) == RESULT_SCHEMA, "result schema drift"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--limit", type=int, default=10,
                        help="documents per arm; keep small for CPU")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--no-write", action="store_true",
                        help="measure and print only (does not touch results/)")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit must be >= 1")

    print(f"model: {args.model} · task: {args.task} · docs/arm: {args.limit}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fp32 = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32
    ).to("cpu")
    fp32_dtype = recorded_dtype(fp32)
    print(f"fp32 arm: measured dtype {fp32_dtype}", flush=True)
    ppl_fp32 = run_arm(fp32, tokenizer, args.task, args.limit, "float32")
    print(f"fp32 perplexity: {ppl_fp32:.4f}", flush=True)

    int8 = quantize_dynamic(
        fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    int8_dtype = recorded_dtype(int8)
    print(f"int8 arm: measured dtype {int8_dtype}", flush=True)
    # HFLM calls tie_weights() in __init__, which tries to re-assign an
    # already-packed quantized weight.  Tying is already done in the fp32
    # checkpoint and dynamic quantization leaves embeddings intact, so a no-op
    # is the faithful behaviour here.
    int8.tie_weights = lambda: None
    ppl_int8 = run_arm(int8, tokenizer, args.task, args.limit, "auto")
    print(f"int8 perplexity: {ppl_int8:.4f}", flush=True)

    delta = ppl_int8 - ppl_fp32
    verdict = evaluate_delta(delta, args.threshold)
    print(
        f"delta: {delta:+.4f} points · threshold: {args.threshold:.1f} · "
        f"{'PASS (proxy misses)' if verdict['pass'] else 'FAIL (proxy detects)'}"
    )
    print(verdict["action"])

    if not args.no_write:
        common = dict(task=args.task, model=args.model,
                      num_samples=args.limit)
        write_result(
            args.out_dir / "lm_eval_fp32.json", precision="fp32",
            perplexity=ppl_fp32, measured_dtype=fp32_dtype, **common,
        )
        write_result(
            args.out_dir / "lm_eval_int8.json", precision="int8",
            perplexity=ppl_int8, measured_dtype=int8_dtype, **common,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
