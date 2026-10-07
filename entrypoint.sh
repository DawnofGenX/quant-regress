#!/bin/sh
set -eu

EVAL="$1"; MODEL="$2"; PRECISIONS="$3"; MAX_DROP="$4"; REPORT="$5"; CACHE_DIR="${6:-}"
MIN_ACCURACY="${7:-}"

if [ ! -f "$EVAL" ]; then
  echo "::error::eval set not found at $EVAL"
  exit 2
fi

set -- --eval "$EVAL" --model "$MODEL" \
       --precisions "$PRECISIONS" \
       --max-drop-points "$MAX_DROP" \
       --report "$REPORT"

if [ -n "$CACHE_DIR" ]; then
  set -- "$@" --cache-dir "$CACHE_DIR"
fi

# Only forward the floor when provided: an empty string would fail validation
# ("" is not a fraction), and the flag must stay opt-in.
if [ -n "$MIN_ACCURACY" ]; then
  set -- "$@" --min-accuracy "$MIN_ACCURACY"
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
