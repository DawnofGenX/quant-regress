# quant-regress

Fail CI when quantization costs you task accuracy.

> **If you pinned `@v1` before 2026-10-07, re-sync.** That tag has moved to pick up
> five correctness fixes, including one where the gate reported PASS for a model it had
> never actually quantized. See [`CHANGELOG.md`](CHANGELOG.md).

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
    min-accuracy: 0.5      # optional; see "Change is not quality" below
```

`@v1` is a moving tag — it tracks the current release line, and has been re-pointed once
(to pick up the fixes in `CHANGELOG.md`). Pin a commit SHA if you need immutability.

Or from the command line / your own CI:

```bash
pip install quant-regress

quant-regress --eval evals/tasks.jsonl \
              --model meta-llama/Llama-3.2-1B-Instruct \
              --max-drop-points 2 \
              --min-accuracy 0.5 \
              --report quant-regress-report.json
```

Exit codes: `0` within tolerance, `1` accuracy regressed, `2` bad configuration.

**`--min-accuracy` is worth setting.** Without it the gate only measures *change*, so a
model that is wrong at every precision yields a zero-point drop and **passes** — a green build
for a model that cannot do the task at all. `--min-accuracy` is a fraction (`0.5` = half) and
fails the build when the *baseline* is that weak. It is a quality failure, so it exits `1`,
not `2`: the tool measured correctly and the answer is no. See
[change is not quality](#change-is-not-quality-read-this-before-trusting-a-pass).

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
- **The two gates use different units, deliberately.** `--max-drop-points` is in
  *percentage points*; `--min-accuracy` is a *fraction* — `0.5`, not `50`. A
  percentage-style value is rejected with an explanatory error rather than
  silently clamped, because a typo there would disable the floor instead of
  tightening it.
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
### Change is not quality — read this before trusting a PASS

The gate's default judgement is about *change*: it fails when quantization costs
accuracy. That means **a model that is equally bad at both precisions produces a
zero-point drop and passes.** Measured on a ten-case set: <!-- claim:quiet-model -->
a model answering nothing correctly at either precision reports
`worst drop: +0.00 pts -> PASS` and exit `0` <!-- claim:quiet-model -->, and the same
model with `--min-accuracy 0.5` reports `FAIL`. Regenerate with
`python scripts/regen_quiet_model.py`; the run's output is committed at
[`results/quiet_model.json`](results/quiet_model.json).

That is correct for the tool's purpose — quantization did not cause it — but it
means a green build is **not** evidence that the model works. Two options:

- Pass `--min-accuracy 0.5` (or your own floor) so the baseline is gated too. This
  turns the case above into `FAIL` / exit `1`, with the reason printed.
- Read the baseline accuracy in the report (`results/*.json`, or the
  `baseline accuracy:` line) before trusting a PASS. The number has always been
  recorded; nothing enforced it.

A floor breach is reported as a **quality** failure (exit `1`), not a configuration
error (exit `2`), because the tool measured correctly and the answer is simply no.
- What the repo's own CI pins is the **exit-code contract** (0 holds, 1 regressed,
  2 misconfigured), not any particular model's accuracy — see
  `tests/test_selftest_gate.py`.
- No published adoption yet. If you use it, an issue saying what you quantized
  and what broke is genuinely useful.
