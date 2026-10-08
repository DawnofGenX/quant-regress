#!/bin/sh
set -eu

# The action passes named flags (--eval <path> ...), not positional args, so a
# new input is a one-line change here and in action.yml. This was formerly a
# positional list ($1..$7) that silently mis-bound whenever a flag was added.

EVAL=""
MODEL=""
PRECISIONS="int8"
BASELINE="fp32"
MAX_DROP="2"
MIN_ACCURACY=""
MAX_NEW_TOKENS="32"
SCORER="exact"
SYSTEM_PROMPT=""
LABELS=""
OUTPUT_FORMAT="text"
JUNIT_PATH=""
REPORT="quant-regress-report.json"
HF_TOKEN_INPUT=""
CACHE_DIR=""

while [ $# -gt 0 ]; do
  flag="$1"
  case "$flag" in
    --eval|--model|--precisions|--baseline-precision|--max-drop-points|--min-accuracy|--max-new-tokens|--scorer|--system-prompt|--labels|--output-format|--junit-path|--report|--hf-token|--cache-dir)
      shift
      [ $# -gt 0 ] || { echo "::error::missing value for $flag"; exit 2; }
      case "$flag" in
        --eval) EVAL="$1" ;;
        --model) MODEL="$1" ;;
        --precisions) PRECISIONS="$1" ;;
        --baseline-precision) BASELINE="$1" ;;
        --max-drop-points) MAX_DROP="$1" ;;
        --min-accuracy) MIN_ACCURACY="$1" ;;
        --max-new-tokens) MAX_NEW_TOKENS="$1" ;;
        --scorer) SCORER="$1" ;;
        --system-prompt) SYSTEM_PROMPT="$1" ;;
        --labels) LABELS="$1" ;;
        --output-format) OUTPUT_FORMAT="$1" ;;
        --junit-path) JUNIT_PATH="$1" ;;
        --report) REPORT="$1" ;;
        --hf-token) HF_TOKEN_INPUT="$1" ;;
        --cache-dir) CACHE_DIR="$1" ;;
      esac
      shift
      ;;
    *)
      echo "::error::unknown argument to quant-regress: $flag"
      exit 2
      ;;
  esac
done

if [ ! -f "$EVAL" ]; then
  echo "::error::eval set not found at $EVAL"
  exit 2
fi

if [ -z "$MODEL" ]; then
  echo "::error::model input is required"
  exit 2
fi

set -- --eval "$EVAL" --model "$MODEL" \
       --precisions "$PRECISIONS" \
       --baseline-precision "$BASELINE" \
       --max-drop-points "$MAX_DROP" \
       --max-new-tokens "$MAX_NEW_TOKENS" \
       --scorer "$SCORER" \
       --output-format "$OUTPUT_FORMAT" \
       --report "$REPORT"

# Only forward the floor when provided: an empty string would fail validation
# ("" is not a fraction), and the flag must stay opt-in.
if [ -n "$MIN_ACCURACY" ]; then
  set -- "$@" --min-accuracy "$MIN_ACCURACY"
fi

# Free-text inputs are dropped when empty rather than passed as "", which the
# CLI would otherwise read as an empty system prompt or an empty label list.
if [ -n "$SYSTEM_PROMPT" ]; then
  set -- "$@" --system-prompt "$SYSTEM_PROMPT"
fi

if [ -n "$LABELS" ]; then
  set -- "$@" --labels "$LABELS"
fi

if [ -n "$CACHE_DIR" ]; then
  set -- "$@" --cache-dir "$CACHE_DIR"
fi

# Forwarded only when set, for the same reason as the floor: an empty value
# must not become the literal JUnit filename (the CLI falls back to a
# alongside --report only when the flag is absent).
if [ -n "$JUNIT_PATH" ]; then
  set -- "$@" --junit-path "$JUNIT_PATH"
fi

# An explicit input wins; otherwise fall back to the HF_TOKEN repository secret
# the user configured. Never echoed.
HF_TOKEN_VALUE="${HF_TOKEN_INPUT:-${HF_TOKEN:-}}"
if [ -n "$HF_TOKEN_VALUE" ]; then
  set -- "$@" --hf-token "$HF_TOKEN_VALUE"
fi

# Disable the HF Xet backend: it needs extra native deps that are absent in the
# slim runtime image, and the failure is opaque.
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1

rc=0
python -m quant_regress.cli "$@" || rc=$?

# Always upload the report, including on failure.
if [ -f "$REPORT" ]; then
  echo "::group::quant-regress report"
  cat "$REPORT"
  echo "::endgroup::"
fi

exit $rc
