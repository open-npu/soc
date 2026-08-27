#!/usr/bin/env bash
# Fresh CSIM golden → firmware → SoC RTL. Never reuse leftover npy/blob.
# Fails unless UART contains RESULT: PASS and no layer reports npu=0.
set -euo pipefail

MODEL="${1:?usage: $0 model_b_int16}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GEN="$ROOT/tools/gen/${MODEL}_golden.py"
LOG="$ROOT/soc/sim/sim_${MODEL}_accept.log"

if [[ ! -f "$GEN" ]]; then
  echo "ERROR: missing $GEN" >&2
  exit 2
fi

echo "=== 0/4 wipe cached converter/golden for $MODEL ==="
rm -rf "/tmp/${MODEL}_golden"
rm -rf "$ROOT/rtl/tb/golden/golden_dma_e2e/${MODEL}"

echo "=== 1/4 convert + CSIM dump → $MODEL golden ==="
python3 "$GEN"

echo "=== 2/4 pack firmware from that golden ==="
cd "$ROOT/soc/firmware"
make clean
make MODEL="$MODEL"

echo "=== 3/4 SoC Verilator vs this CSIM ==="
cd "$ROOT/soc/sim"
make run 2>&1 | tee "$LOG"

echo "=== 4/4 require RESULT: PASS and npu>0 ==="
if grep -q "RESULT: FAIL" "$LOG"; then
  echo "ERROR: $MODEL printed RESULT: FAIL" >&2
  exit 1
fi
if ! grep -q "RESULT: PASS" "$LOG"; then
  echo "ERROR: $MODEL did not print RESULT: PASS" >&2
  exit 1
fi
if grep -qE "PERF_L[0-9]+ npu=0( |$)" "$LOG"; then
  echo "ERROR: $MODEL has a layer with npu busy cycles=0" >&2
  exit 1
fi
echo "OK: $MODEL RESULT: PASS"
