# Changelog

All notable changes to quant-regress. This project is pre-1.0 stable in API but the
`v1` GitHub Action tag is what users are told to pin, so **action-visible changes are
called out here as breaking**.

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