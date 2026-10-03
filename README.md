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

<!-- claim:int8-collapse -->
| precision | accuracy | correct |
|---|---|---|
| fp32 (baseline) | 100.0% <!-- claim:int8-collapse --> | 20/20 |
| int8 | 0.0% <!-- claim:int8-collapse --> | 0/20 |

Regenerate with `python scripts/regen_int8_collapse.py` — every figure above
comes from that script, and `tests/test_claims_are_reproducible.py` fails the
build if a number in this file stops matching it.

The pattern that matters: the quantized model still runs, still returns a valid
answer, and still looks like a working model. It is simply wrong every time.
That is why a token-level proxy can clear it and a task eval cannot.

## Usage

As a GitHub Action:

```yaml
- uses: DawnofGenX/quant-regress@v1
  with:
    eval: evals/tasks.jsonl
    model: meta-llama/Llama-3.2-1B-Instruct
    max-drop-points: 2
```

Or from the command line / your own CI:

```bash
pip install quant-regress

quant-regress --eval evals/tasks.jsonl \
              --model meta-llama/Llama-3.2-1B-Instruct \
              --max-drop-points 2 \
              --report quant-regress-report.json
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
- **The bundled `evals/fixtures/collapse.jsonl` is model-specific.** Its cases
  expect the answer `yes`, so it only demonstrates a collapse for a model that
  answers `yes` when unquantized. Pointed at an untuned tiny LM it scores 0/N at
  both precisions, produces no drop, and the gate correctly reports PASS. Use it
  to see the output format; bring your own eval for a real check.
- What the repo's own CI pins is the **exit-code contract** (0 holds, 1 regressed,
  2 misconfigured), not any particular model's accuracy — see
  `tests/test_selftest_gate.py`.
- No published adoption yet. If you use it, an issue saying what you quantized
  and what broke is genuinely useful.
