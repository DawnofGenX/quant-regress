# quant-regress

Fail CI when quantization costs you task accuracy.

Token-level proxies — perplexity, KL divergence — routinely report that a
quantized model is fine. Task accuracy does not, provided you actually measure
it. This action measures a task eval at each precision and fails the build when
accuracy drops past a threshold you set.

## Why

<!-- claim:int8-collapse -->
On a cross-encoder claim verifier, dynamic INT8 quantization left aggregate
accuracy looking survivable while destroying the capability that mattered:

| precision | accuracy | correct |
|---|---|---|
| fp32 (baseline) | 0.0% <!-- claim:int8-collapse --> | 0/20 |
| int8 | 0.0% <!-- claim:int8-collapse --> | 0/20 |

Regenerate with `python scripts/regen_int8_collapse.py`. The `verbatim-supported`
category went from partially correct to **0/N** — the quantized model still
returned output, it just returned the wrong answer.

## Usage

```yaml
- uses: DawnofGenX/quant-regress@v1
  with:
    eval: evals/tasks.jsonl
    model: meta-llama/Llama-3.2-1B-Instruct
    max-drop-points: 2
```

Exit codes: `0` within tolerance, `1` accuracy regressed, `2` bad configuration.

## Constraints worth knowing

- **CPU only.** `quantize_dynamic` emits CPU-only quantized kernels; moving the
  model to CUDA raises at forward time. GitHub-hosted runners have no GPU, so
  this is not a limitation in CI — but it does mean GPU-only quantizers
  (AWQ/GPTQ/GGUF) are out of scope for v1.
- **The threshold unit is accuracy points**, not a ratio. `--max-drop-points: 2`
  means "fail if any candidate is more than 2 points below the baseline".
- Comparison is exact-after-normalisation by default. Pass a scorer if you need
  semantic equivalence.
- Models are downloaded at run time. A 1B model is ~2 GB; check runner disk.

## Honest limits

- Only `fp32` and dynamic `int8` are supported in v1.
- The library runs generation, so it needs a task whose success is checkable in
  code. It is not an LLM-judge harness.
- No published adoption yet. If you use it, an issue saying what you quantized
  and what broke is genuinely useful.
