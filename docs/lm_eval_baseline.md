# Independent lm-eval perplexity baseline

This is a cross-check on the repo's central story. The task-level tables show
large accuracy drops under dynamic INT8; it is fair to ask whether a standard
token-level metric would have shown the same thing. This baseline measures that
instead of asserting it.

## What was measured (real output, not an estimate)

| arm | task | documents | word perplexity | measured weight dtype |
|---|---|---:|---:|---|
| fp32 | `wikitext` (lm-eval 0.4.13, `loglikelihood_rolling`) | 5 | 17.0159 | `torch.float32` |
| dynamic INT8 | same task, same documents, same seeds | 5 | 243.4676 | `torch.qint8` (Linear layers) |

Delta: **+226.4517 points** against a public kill threshold of 2.0 points.
The token-level proxy detected the collapse decisively in this setup.

The committed artifacts are [`results/lm_eval_fp32.json`](../results/lm_eval_fp32.json)
and [`results/lm_eval_int8.json`](../results/lm_eval_int8.json). Both were
produced by `scripts/regen_lm_eval_baseline.py` on CPU with
`Qwen/Qwen2.5-0.5B-Instruct`, `lm_eval==0.4.13`, `torch==2.4.1+cpu`,
`transformers==4.57.6`, and `HF_HUB_OFFLINE=1` using the locally cached
`EleutherAI/wikitext_document_level` (wikitext-2-raw-v1). The run is
deterministic; a second run with `--no-write` reproduced both numbers exactly.

## Consequence for the repo's claims

An earlier README sentence said token-level proxies such as perplexity and KL
"routinely report that a quantized model is fine". That is **retracted** for
this checkpoint and this quantization path. The correct claim is narrower:
a task eval is necessary because it measures the capability you care about,
but a token-level proxy is not guaranteed to be blind — it must be measured.

`quant_regress.lm_eval_baseline.evaluate_delta()` encodes the kill rule shared
by the script and `tests/test_lm_eval_baseline.py`:

- `delta < 2.0`: perplexity missed the collapse; the claim that task-level
  accuracy is the necessary signal stands, with this baseline as evidence.
- `delta >= 2.0` (the measured case): perplexity detected the collapse; do not
  claim that token-level proxies clear this model. The README and any external
  write-up must say so.

Neither outcome is a failed experiment. Hiding the outcome is.

## Reproduce

```bash
cd /home/hermes/quant-regress
HF_HUB_OFFLINE=1 .venv/bin/python scripts/regen_lm_eval_baseline.py --limit 5
.venv/bin/python -m pytest tests/test_lm_eval_baseline.py -q
```

The development environment is gitignored; `pyproject.toml` pins `lm-eval` in
the `dev` extra. Because the system interpreter here carries incompatible
user-site packages, the working venv also pins `huggingface-hub<1.0` and
`datasets==3.2.0`; the production dependency set (torch/transformers/numpy) is
unchanged. The script is CPU-only and never touches CUDA.

## Caveats

- Five wikitext documents is a small sample: it is a smoke baseline, not a
  publication-grade perplexity estimate. The effect size here is far beyond
  the threshold, but do not extrapolate to other checkpoints or quantizers.
- `torch.ao.quantization.quantize_dynamic(..., {torch.nn.Linear}, qint8)`
  quantizes Linear layers only; embeddings remain float32. The result column
  records what was actually quantized.
- HFLM calls `tie_weights()` on model construction; for the quantized arm the
  script supplies a no-op because the weights were already tied in the
  checkpoint and packed Linear weights cannot be reassigned.
