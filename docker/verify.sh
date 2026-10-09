#!/usr/bin/env bash
# End-to-end check of the CPU testing ground. Runs inside the dev container:
#
#   docker compose -f docker/docker-compose.yml run --rm dev bash docker/verify.sh
#
# Every step must pass; the script prints a summary and exits non-zero on any failure.
set -uo pipefail
cd /workspace

OUT=$(mktemp -d /tmp/lpn_verify.XXXX)
export LPN_OUTPUT_ROOT="$OUT"
RESULTS=()
FAILED=0

step() {  # step "<name>" <command...>
  local name=$1; shift
  echo; echo "=== $name"
  local start=$SECONDS
  if "$@"; then
    RESULTS+=("PASS  $name ($((SECONDS - start))s)")
  else
    RESULTS+=("FAIL  $name ($((SECONDS - start))s)"); FAILED=1
  fi
}

environment() {
  python - <<'EOF'
import sys, torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
assert "+cpu" in torch.__version__ and not torch.cuda.is_available(), "expected the CPU-only torch build"
EOF
}

train_mini() {  # train_mini <config> -> runs it and checks final.pt exists
  local name; name=$(basename "$1" .yaml)
  python scripts/train.py --config "$1" --device cpu || return 1
  test -f "$OUT/runs/$name/final.pt" || { echo "missing $OUT/runs/$name/final.pt"; return 1; }
  python - "$OUT/runs/$name/metrics.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
done = [r for r in rows if r["kind"] == "done"][-1]
print("final:", {k: v for k, v in done.items() if k.startswith("seen/")})
EOF
}

rerun_is_noop() {
  python scripts/train.py --config src/configs/stage3_scratch_mini.yaml --device cpu 2>&1 | tee /dev/stderr \
    | grep -q "already complete"
}

sigterm_then_resume() {  # SIGTERM mid-run -> exit 75 + checkpoint; rerun finishes with no duplicate steps
  local run="$OUT/runs/interrupted"
  python scripts/train.py --config src/configs/stage3_scratch_mini.yaml --device cpu --name interrupted &
  local pid=$!
  for _ in $(seq 600); do [ -f "$run/checkpoint.pt" ] && break; sleep 0.5; done
  sleep 2
  kill -TERM "$pid"; wait "$pid"; local rc=$?
  echo "interrupted run exited with $rc (expect 75)"
  [ "$rc" -eq 75 ] || return 1
  test ! -f "$run/final.pt" || { echo "run finished before the SIGTERM landed; nothing was tested"; return 1; }
  python scripts/train.py --config src/configs/stage3_scratch_mini.yaml --device cpu --name interrupted \
    | grep -q "resumed from step" || return 1
  python - "$run/metrics.jsonl" <<'EOF'
import json, sys
steps = [json.loads(l)["step"] for l in open(sys.argv[1]) if '"train"' in l]
assert steps == sorted(set(steps)), f"duplicate/out-of-order steps: {steps}"
print("train rows:", len(steps), "steps", steps[0], "..", steps[-1], "(no duplicates)")
EOF
}

parity_needs_jax() {  # Colab-only: must refuse cleanly here, not crash
  local out; out=$(python colab/check_port_parity.py 2>&1); local rc=$?
  echo "$out" | tail -3
  [ "$rc" -ne 0 ] && echo "$out" | grep -q "run it on Colab"
}

step "environment: CPU-only torch"            environment
step "lint (ruff)"                             ruff check .
step "unit tests (pytest)"                     python -m pytest -q
step "image satisfies colab/requirements.txt"  python colab/vm_setup.py colab/requirements.txt
step "stage 1 mini (frozen + head)"            train_mini src/configs/stage1_frozen_mini.yaml
step "stage 2 mini (fine-tune)"                train_mini src/configs/stage2_finetune_mini.yaml
step "stage 3 mini (from scratch)"             train_mini src/configs/stage3_scratch_mini.yaml
step "rerun of a finished run is a no-op"      rerun_is_noop
step "SIGTERM -> checkpoint -> resume"         sigterm_then_resume
step "gpu_smoke runs on CPU"                   python colab/gpu_smoke.py --device cpu --steps 3 --batch-size 16
step "parity check refuses without JAX"        parity_needs_jax

echo; echo "=== summary (outputs in $OUT)"
printf '%s\n' "${RESULTS[@]}"
exit $FAILED
