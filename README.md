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
build if a number in this file stops matching it. The run's output is committed at
[`results/int8_collapse.json`](results/int8_collapse.json), so the table above can be
audited without running anything.

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
- **Not every architecture can be quantized by this method, and the action
  refuses rather than lying.** `quantize_dynamic` only replaces module types in
  torch's own mapping table, which covers `nn.Linear`. Architectures whose
  blocks use something else — GPT-2 and GPT-J use `transformers.Conv1D` — would
  have their transformer body left in fp32, so the "int8" arm would really be
  fp32 and the tool would report a **0.00-point drop and PASS** for a model it
  never quantized. That architecture now exits `2` (misconfigured) with an
  actionable message instead. Use an `nn.Linear`-based architecture:
  BERT, DeBERTa, Llama, Mistral.
- **Only the model's continuation is scored.** `generate()` returns the prompt
  followed by the new tokens, so the prediction is what the model added — not
  the question you asked it. Prompts should ask for a short, exact answer, and
  `expected` should be that answer. A chatty model will not exact-match: pass a
  custom scorer via `QuantHarness(..., scorer=fn)` when using the library. (There
  is no `--scorer` flag on the CLI yet; from the command line the default
  exact-after-normalisation comparison is what you get.)
- **The threshold unit is accuracy points**, not a ratio. `--max-drop-points: 2`
  means "fail if any candidate is more than 2 points below the baseline".
- Comparison is exact-after-normalisation by default. Pass a scorer if you need
  semantic equivalence.
- Models are downloaded at run time. A 1B model is ~2 GB; check runner disk.

## Honest limits

- Only `fp32` and dynamic `int8` are supported in v1.
- The library runs generation, so it needs a task whose success is checkable in
  code. It is not an LLM-judge harness.
- **The bundled `evals/fixtures/collapse.jsonl` is model-specific, and it does
  not exercise the generate path.** Its cases expect the answer `yes`, so it only
  demonstrates a collapse for a model that answers `yes` when unquantized.
  Pointed at an untuned tiny LM it scores 0/N at both precisions, produces no
  drop, and the gate correctly reports PASS. It is also driven by a stub model
  exposing `.answer()`, so it tells you nothing about real decoding. Use it to
  see the output format; bring your own eval for a real check.
- A model that is quiet at both precisions yields a 0.00-point drop and PASSes.
  The gate reports *change*, not *quality*: if your baseline model is already
  wrong, quant-regress will not tell you. Check the baseline accuracy in the
  report before trusting a PASS.
- What the repo's own CI pins is the **exit-code contract** (0 holds, 1 regressed,
  2 misconfigured), not any particular model's accuracy — see
  `tests/test_selftest_gate.py`.
- No published adoption yet. If you use it, an issue saying what you quantized
  and what broke is genuinely useful.
