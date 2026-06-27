#!/usr/bin/env bash
# Sequential 1000-step perf experiments. Each config isolates one lever
# cumulatively so gains can be attributed:
#   A baseline            -> current code (compile wrap = no-op)
#   B +compile(step)      -> the real torch.compile fix
#   C +amp(bf16)+tf32     -> mixed precision / tensor cores
#   D +workers+pin        -> overlapped data loading
set -u
source .venv/bin/activate
export WANDB_API_KEY=$(tr -d '[:space:]' < ~/.wandb_personal_key)
export SDL_VIDEODRIVER=dummy
# persistent inductor cache so later compile runs reuse artifacts
export TORCHINDUCTOR_CACHE_DIR=/workspace/pusht-slim/.inductor_cache

COMMON="--steps 1000 --eval-every 0 --save-every 0 --perf-log-every 100"

run () {
  name="$1"; shift
  echo "================ $name ================"
  python -u train.py $COMMON --run-name "$name" "$@" > "perf_${name}.log" 2>&1
  echo "exit=$? for $name"
}

run perf-A-baseline      --compile-mode wrap --amp none --no-tf32   --num-workers 0 --no-pin-memory
run perf-B-compile-step  --compile-mode step --amp none --no-tf32   --num-workers 0 --no-pin-memory
run perf-C-amp-tf32      --compile-mode step --amp bf16 --tf32      --num-workers 0 --no-pin-memory
run perf-D-workers-pin   --compile-mode step --amp bf16 --tf32      --num-workers 4 --pin-memory
echo "ALL DONE"
