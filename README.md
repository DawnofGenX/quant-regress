# quant-regress

Fail CI when quantization costs you task accuracy.


Token-level proxies — perplexity, KL divergence — routinely report that a
quantized model is fine. Task accuracy does not, provided you actually measure
it. This action measures a task eval at each precision and fails the build when
accuracy drops past a threshold you set.

> **If you pinned `@v1` before 2026-10-15, re-sync.** The tag has been re-pointed at least four
> times, most recently on 2026-10-15, each time to pick up correctness fixes — including one where
> the gate reported PASS for a model it had never actually quantized. For a gate you rely on, pin a
> SHA rather than a moving tag. See [`CHANGELOG.md`](CHANGELOG.md).

## Measured on a real checkpoint

The headline is a real measurement, not a stub. `scripts/regen_real_model.py`
loads **Qwen/Qwen2.5-0.5B-Instruct** and runs two 60-case eval sets through the
same `QuantHarness` and adapter path the GitHub Action uses — one where the task
is arithmetic word problems (numeral-scored), one where it is sentiment
classification (exact match plus declared aliases). Both sets are 60 cases, so
one case is worth 1.67 points and a 2-point tolerance is reachable; the scorer
is chosen per set, and every arm records the dtype it actually measured.

### Arithmetic: int8 destroys the capability

<!-- claim:real-model -->
| precision | accuracy | correct | drop | 95% CI (pts) | measured dtype | <!-- claim:real-model -->
|---|---|---|---|---|---|
| fp32 (baseline) | 33.3% <!-- claim:real-model --> | 20/60 <!-- claim:real-model --> | - | - | torch.float32 |
| int8 | 3.3% <!-- claim:real-model --> | 2/60 <!-- claim:real-model --> | +30.0 pts <!-- claim:real-model --> | 18.3 to 43.3 | torch.qint8 |
| bf16 | 33.3% <!-- claim:real-model --> | 20/60 <!-- claim:real-model --> | +0.0 pts | 0.0 to 0.0 | torch.bfloat16 |

worst drop: +30.0 pts -> FAIL <!-- claim:real-model -->

baseline 33.3% is above the --min-accuracy floor (0.3) <!-- claim:real-model -->

The interval is what makes 30.0 points mean something: a 95% confidence interval <!-- claim:real-model -->
of 18.3 to 43.3 points excludes zero, so this is a collapse rather than one
unlucky case. bf16's interval of 0.0 to 0.0 is the negative control — it matches
fp32 case for case.

Read this table for the delta, not for the model. A 0.5B checkpoint is only
partly competent at these word problems, and the tiers show how uneven that is:
`easy` cases (one step) are mostly solved, `medium` (three addends, a division
step) mostly are not, and the non-monotonic `hard` row is the reason the set is
worth reading per tier instead of in aggregate.

| arm | difficulty | correct |
|---|---|---|
| fp32 | easy | 12/20 <!-- claim:real-model --> |
| fp32 | medium | 2/20 <!-- claim:real-model --> |
| fp32 | hard | 6/20 <!-- claim:real-model --> |

`bf16` matches `fp32` case for case, which is the useful negative result: the
half-precision backend is not what breaks the model, and the `measured dtype`
column is what makes that claim checkable rather than asserted.

### Sentiment: a competent baseline, and the same collapse

<!-- claim:real-model -->
| precision | accuracy | correct | drop | 95% CI (pts) | measured dtype | <!-- claim:real-model -->
|---|---|---|---|---|---|
| fp32 (baseline) | 98.3% <!-- claim:real-model --> | 59/60 <!-- claim:real-model --> | - | - | torch.float32 |
| int8 | 3.3% <!-- claim:real-model --> | 2/60 <!-- claim:real-model --> | +95.0 pts <!-- claim:real-model --> | 88.3 to 100.0 | torch.qint8 |

worst drop: +95.0 pts -> FAIL <!-- claim:real-model -->

baseline 98.3% is above the --min-accuracy floor (0.5) <!-- claim:real-model -->

This is the table to quote. The same checkpoint classifies 59 of 60 reviews
correctly at fp32 and 2 of 60 at dynamic int8: the quantized model still runs,
still returns a label, and is wrong almost every time. A token-level proxy that
only watches the output distribution clears that; a task eval does not.

Regenerate everything with one command — `python scripts/regen_real_model.py`
with no flags prints both committed runs, and `--measure` re-runs the model on
CPU (about two minutes for both sets on 24 cores). The committed artifacts are
[`results/real_model_arith.json`](results/real_model_arith.json) and
[`results/real_model_classify.json`](results/real_model_classify.json), and
`tests/test_claims_are_reproducible.py` fails the build if a number in these
tables stops matching the script's output.

### The controlled reproduction (stub)

The original motivation was a real cross-encoder claim verifier in
[`citesure`](https://github.com/DawnofGenX/citesure), where dynamic INT8 left
aggregate accuracy looking survivable while destroying the capability that
mattered. This repo's adapter loads `AutoModelForCausalLM`, so it **cannot load
a cross-encoder at all** — the table below is therefore this repo's **controlled
reproduction** of that shape, not a run of that model: a stub whose only
behaviour is to separate at fp32 and collapse at int8, so the gate has something
deterministic to catch in its own CI. Read its figures as a test of the gate's
arithmetic, not as a model measurement.

<!-- claim:int8-collapse -->
| precision | accuracy | correct |
|---|---|---|
| fp32 (baseline) | 100.0% <!-- claim:int8-collapse --> | 20/20 |
| int8 | 0.0% <!-- claim:int8-collapse --> | 0/20 |

Regenerate with `python scripts/regen_int8_collapse.py`; the run's output is
committed at [`results/int8_collapse.json`](results/int8_collapse.json).

The pattern that matters in both tables: the quantized model still runs, still
returns a valid answer, and still looks like a working model. It is simply
wrong every time. That is why a token-level proxy can clear it and a task eval
cannot.

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

`@v1` is a moving tag — it tracks the current release line, and has been re-pointed at least four
times (to pick up the fixes in `CHANGELOG.md`). Pin a commit SHA if you need immutability.

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

`--dry-run` checks the whole configuration and exits **without downloading a
model** — a typo'd precision, a baseline equal to a candidate, or a broken eval
set costs milliseconds instead of a multi-GB download. `--version` prints the
installed version, for pinning a ref or filing a bug.

### Outputs

`--report` writes JSON containing every arm's accuracy, the drop in points, and the
`misclassified` case ids — so a red build names the cases that got worse instead of
only saying that something did. `--output-format junit --junit-path results.xml`
additionally writes JUnit XML with one `<testcase>` per precision, each failing arm
carrying a `<failure>` naming the drop and the cases. That is what Jenkins, GitLab
and the GitHub test-results pane render.

`--progress-every 25` prints a progress line to **stderr** every N cases. Stdout
stays a parseable table, so a workflow that greps the summary line is unaffected.

### Is the drop real? Confidence intervals

Every report carries a paired bootstrap interval on the drop, because a point
estimate cannot distinguish "one case flipped" from "the quantized model cannot
do the task". The text report prints one line per candidate, e.g.
`95% CI of the drop (int8): 18.3 to 43.3 points (n=60, 10000 resamples)` <!-- claim:real-model -->

The interval is in **points**, is paired on case identity (both arms answer the
same cases, so only the flips are resampled), and is seeded, so the same run
always prints the same bounds. Markdown gains a `95% CI (pts)` column, JUnit <!-- claim:real-model -->
gains a `drop_ci_95_<precision>` property, and the JSON report gains
`drop_ci_95` plus `drop_significant` per candidate.

`--require-significance` changes the verdict, not the numbers: a candidate whose
interval contains zero is reported as **PASS** with a note explaining that the
drop is inside the sampling noise. It is off by default — silently passing a
real regression is worse than flagging a noisy one — and it is the right flag
when a one-case delta should not block a merge. Note what it cannot do: on a
60-case set the interval is 10-20 points wide, so *no* tolerance below ~10
points is resolvable. Widen the eval set for a finer judgement.

### Step summaries

`--summary-file PATH` appends the markdown table to `PATH`; the literal value
`auto` means `$GITHUB_STEP_SUMMARY`, and an empty value skips. It appends rather
than overwrites, so several runs accumulate in one job summary, and a write
failure (read-only path, missing directory) warns on stderr while leaving the
exit code alone. The Action exposes the same knob as the `summary-file` input,
defaulting to `auto`.

### Matching a chatty model

`--scorer contains` matches when `expected` appears anywhere in the prediction, for
a model that answers in a sentence. The default is `exact` — normalised equality,
which is what makes a marginal-decision drop visible in the first place. Only those
two are exposed on the CLI; a scorer that needs real logic belongs in the library,
via `QuantHarness(..., scorer=fn)`. Deliberately not an arbitrary expression: this
tool runs on arbitrary PRs, and evaluating one would be code execution.

`--labels "no,yes"` names the classes for a logits-only model with no `id2label`.
`--system-prompt` prepends a prompt to every case. `--hf-token` (or `$HF_TOKEN`)
authenticates a gated model.

### Chat models

An instruction-tuned checkpoint is prompted through **its own chat template**, not
by pasting text. A base model's tokenizer has no template, so it keeps the plain
concatenation — `--system-prompt` behaves as before. This matters because a chat
model fed a hand-pasted prompt answers conversationally rather than with the
expected token, so the accuracy being measured would be the accuracy of a broken
prompt, and the marginal-decision damage this tool exists to detect would be
invisible.

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
  `expected` should be that answer. A chatty model will not exact-match: use
  `--scorer contains`, or pass a scorer via `QuantHarness(..., scorer=fn)` when
  using the library.
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

### Cost and caching

A run downloads the model once (HF cache) and then generates on every case at
every precision, on CPU, one case at a time. A 20-case, two-precision eval on a
1B model is minutes per invocation, and every PR re-pays it. Two levers:

- `--response-cache DIR` caches each prediction keyed by `(model, precision,
  prompt)`. A re-run of an unchanged eval against an unchanged model skips
  generation entirely and finishes in seconds; the cache is a single
  `responses.json` you can commit or share between runners. When every case is
  cached, the model is not loaded at all, so a fully cached run never touches
  the network or the weights.
- Batching across cases is not implemented yet: prompts go through the model's
  own chat template with variable length, and naive padding changes which tokens
  are scored. Until that is solved honestly, generation stays per-case.

If neither lever applies, budget the run accordingly and read the baseline
accuracy in the report before trusting a PASS.

## Honest limits

- Only `fp32`, `fp16`, `bf16` and dynamic `int8` are supported in v1. AWQ / GPTQ / GGUF / NF4
  need external tooling; the adapter is a registry (`src/quant_regress/adapter.py`,
  `BACKENDS`) so a new backend is a one-function addition, but each must ship
  with a committed measurement before it is advertised.
- The library runs generation, so it needs a task whose success is checkable in
  code. It is not an LLM-judge harness.
- **The bundled `evals/fixtures/collapse.jsonl` is model-specific, and it does
  not exercise the generate path.** Its cases expect the answer `yes`, so it only
  demonstrates a collapse for a model that answers `yes` when unquantized.
  Pointed at an untuned tiny LM it scores 0/N at both precisions, produces no
  drop, and the gate correctly reports PASS. It is also driven by a stub model
  exposing `.answer()`, so it tells you nothing about real decoding. Use it to
  see the output format; bring your own eval for a real check.
- **60 cases is 1.67 points per case.** That is the smallest drop the bundled
  sets can express; a `--max-drop-points` finer than 1.67 is unreachable and the
  CLI now warns (and exits `2` with `--strict-granularity`). Bigger sets buy
  resolution, not credibility — 600 cases resolve 0.17 points and cost ten times
  the CPU.
- **The measured confidence intervals are 12-25 points wide, not 2.** On the
  60-case runs above, the arithmetic int8 drop is 30.0 points with a 95% <!-- claim:real-model -->
  interval of 18.3 to 43.3, and the sentiment drop is 95.0 with 88.3 to 100.0.
  The Action's default `max-drop-points: 2` is therefore a *policy* threshold on
  the point estimate — it says "any drop blocks the merge", not "I can resolve a
  2-point effect". If you need to resolve a small delta, grow the set; no
  statistical test recovers resolution the sample size never had.
- **The `easy` / `medium` / `hard` tiers describe the problem, not a measured
  pass rate.** They are assigned by construction — one step, three addends or a
  division step, a distractor clause — so a weak checkpoint can score `hard`
  above `medium`. Read the per-tier table as "where did it break", not as a
  curriculum.
- **The published numbers are one checkpoint on one CPU box, greedy decoding.**
  They say what dynamic int8 did to `Qwen/Qwen2.5-0.5B-Instruct` on 120 cases;
  they do not rank quantization methods, and they are not a claim about larger
  checkpoints. What generalises is the gate: point it at your model, your task
  and your tolerance, and it answers for that combination.
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

More writing on this stack: https://dev.to/dawnofgenx
