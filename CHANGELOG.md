# Changelog

All notable changes to quant-regress. This project is pre-1.0 stable in API but the
`v1` GitHub Action tag is what users are told to pin, so **action-visible changes are
called out here as breaking**.

## 1.0.3 — pinned at `v1`

### Fixed — the measurement path could OOM, crash, or certify a fake quantization

- **Models are released as soon as their measurement finishes**, in a `finally` so
  a raising scorer cannot leak one. Two precisions means two models, and a 7B
  checkpoint held twice is ~28 GB on a runner with far less.
- **The fp32 original is dropped and a collection forced** right after
  `quantize_dynamic` returns, so peak memory is no longer 2x the model size.
- **A broken model exits 2 instead of tracebacking.** `_predict` failures are
  re-raised as `RegressError` (a `RuntimeError`), which the CLI already catches. An
  unhandled `TypeError` from a broken model previously exited **1** — the code
  meaning "accuracy regressed", i.e. a broken model impersonating a regression.
- **An unwritable `--report` path exits 2 with a message** instead of crashing
  after the measurement already produced a verdict.
- **The quantize guard fires on the swap count alone.** It previously required
  `and blocked:`, a second condition a small `nn.Linear` model can never satisfy —
  so an encoder with a handful of Linear layers passed with an "int8" arm that was
  fp32 in all but name, and the gate compared a model against itself.
- **The incomplete-run check now works.** `_measure` recorded `total=len(eval_set)`
  — the *declared* size — so the check in `compare()` compared a number against
  itself and could never fire. It records the count actually scored, so a set that
  yields fewer cases than it declares is refused rather than reported as a verdict.

### Added — the CLI surface

- `--scorer exact|contains`. A fixed pair, not an expression: this tool runs on
  arbitrary PRs, and evaluating one would be code execution.
- `--labels`, `--system-prompt`, `--hf-token` (falls back to `$HF_TOKEN`).
- `--output-format text|json|junit` with `--junit-path`. JUnit is what Jenkins,
  GitLab and the GitHub test-results pane read.
- `--progress-every N`, printed to **stderr** so stdout stays parseable.
- **Precisions are validated before any model is built**, so a typo like `int4`
  costs nothing to discover instead of a full download.
- **`to_dict()` carries `misclassified` ids for every arm** — a red build names the
  cases that got worse.

### Added — the Action

- The entrypoint binds **named** arguments (`--eval <path> …`) instead of
  `$1..$7` positionally. The positional list silently mis-bound the moment a flag
  was added: `--max-new-tokens` could arrive as `--min-accuracy`.
- New inputs: `baseline-precision`, `max-new-tokens`, `scorer`, `system-prompt`,
  `labels`, `output-format`, `junit-path`, `hf-token`. All default to the previous
  behaviour.

### Fixed in the test suite

- `test_action_contract.py` pins the action's argument translation, including that
  an empty `--min-accuracy` is dropped rather than forwarded as `""` (which would
  fail validation and disable the floor).
- Coverage added for four paths that were reachable but unguarded: the
  incomplete-run refusal, the empty-eval-set refusal, a custom scorer actually
  being called, and default-scorer normalisation. The first of those exposed the
  real `total=len(eval_set)` defect above.
- Every new guard was verified load-bearing by re-introducing the bug and
  confirming the specific test goes red, then green on restore.

## 1.0.2 — pinned at `v1`

### Fixed

- **A baseline equal to a candidate was silently accepted**, so the harness compared
  a model against itself, produced a 0.0-point drop, and reported PASS having measured
  no quantization at all. Now refused before any model loads.
- **The logits path ignored the prompt.** It passed `input_ids=[[1]]` regardless of
  the question, so every case got the same prediction and the comparison was
  meaningless; a tokenizer-less logits model was scored on that dummy input. The
  prompt is tokenized now, and such a model is refused with an actionable error.
- **A non-UTF-8 eval set crashed with `UnicodeDecodeError`** (exit `1` — the code
  meaning "accuracy regressed", so a bad file impersonated a real regression), and a
  JSON row that was not an object surfaced as a bare `KeyError`. Both raise
  `EvalSetError` naming the line and showing a valid example row.
- **The `v1.0.1` release failed to publish**: its tag pointed at a commit whose
  `pyproject.toml` still declared `0.1.1`, so PyPI rejected the duplicate wheel with
  `400 File already exists`. The tag has been re-cut at the version bump, and a
  release test now fails if a version tag ever points at a commit carrying a different
  version.

### Fixed in the test suite

- `test_logits_only_model_uses_labels_not_a_stringified_logit` exercised the dummy
  input the fix removes, so it could not fail on the bug it guards. It now supplies a
  tokenizer and keeps its real assertion — that `id2label` yields `"no"` rather than the
  stringified logit `"0.9"`. Every guard added here was verified load-bearing by
  re-introducing the bug and confirming the run goes red, then green on restore.

## Unreleased — pinned at `v1` as of 2026-10-07

The `v1` tag was moved from `53402e8` to `05b0135` so that the ref the README tells
users to pin carries the fixes below. **If you pinned `@v1` before this date, you were
running code with every one of these defects and need to re-sync your lockstep.**

### Fixed — the gate could PASS a real regression

- **The "int8" arm could be fp32 in the transformer body.** `quantize_dynamic` only
  replaces module types in torch's own mapping table, which covers `nn.Linear`.
  Architectures whose blocks use something else — GPT-2 and GPT-J use
  `transformers.Conv1D` — were left entirely unquantized apart from one out-of-block
  `lm_head`. Measured on `tiny-random-gpt2`: 20 `Conv1D` modules untouched, exactly
  1 module quantized. The tool then reported `drop_points: 0.0`, `verdict: pass`, for
  a model it never quantized. Now refused with an actionable message and exit `2`.
- **The prompt was scored as the model's answer.** `generate()` returns the input ids
  followed by the new ones, and the harness decoded the whole tensor, so the prediction
  began with the question asked. Under exact-match scoring the generate path was
  unsatisfiable: every case scored 0 at every precision, the drop was 0.00 points, and
  the gate reported PASS. Only the continuation is decoded now.
- **Non-finite and negative thresholds inverted or disabled the gate.** `nan` and
  `+inf` made every comparison false, so PASS always won; a negative threshold made the
  test "the quantized model must be at least as bad", so a collapse would pass and a
  healthy model would fail. Both are rejected. `0` is deliberately still allowed.
- **`--precisions ""` reported PASS having compared nothing.** With no candidates,
  `worst_drop_points` fell back to `default=0.0`. Now refused before any model loads.
- **`--eval` pointing at a directory exited `1`.** A directory passes `exists()`, then
  `read_text()` raised an uncaught `IsADirectoryError`, which surfaced as the exit code
  meaning "accuracy regressed" — a config mistake impersonating a real regression.
  Now exit `2`. This also fixes named pipes and devices, on which `read_text()` blocks
  forever.

### Added

- **`--min-accuracy`** — gate the baseline's own quality, as a fraction (`0.5`, not
  `50`). Without it a model that is wrong at every precision produces a 0.00-point
  drop and passes, so a green build was not evidence the model works. A floor breach is
  a *quality* failure (exit `1`), not a configuration error (exit `2`): the tool
  measured correctly and the answer is no. Opt-in; behaviour and exit codes are
  unchanged when it is not passed. Also exposed as an Action input.
- `results/int8_collapse.json` is committed, so the README's headline table is auditable
  without running anything.
- `scripts/regen_quiet_model.py` + `results/quiet_model.json` regenerate the
  "change is not quality" figures.

### Known limitations

- Only `fp32` and dynamic `int8` are supported. AWQ/GPTQ/GGUF/NF4 are out of scope.
- `AutoModelForCausalLM` only. Cross-encoders, classifiers and embedders cannot be
  loaded through the CLI — including the cross-encoder that motivated this tool.
- CPU only. No batching, no progress output, no `--scorer` flag on the CLI.
- The bundled `evals/fixtures/collapse.jsonl` is driven by a stub model and proves the
  gate fires on a controlled collapse; it is not evidence about any real checkpoint.

## 0.1.1 — 2026-10-04

- Fixed the documented `pip install` path; verified a clean-venv install.
- `fix(self-test)`: pin the exit-code contract instead of asserting a real model
  regresses.

## 0.1.0 — 2026-10-03

- First release: Action + CLI + library, exit codes 0/1/2, Trusted Publishing.