#!/bin/bash
# DRIVER8 — C-series continuation at OBS_HORIZON=1 (user directive: obs1 only).
# All prior C-series results (incl. C1's SR 0.72) were obs2, so run the C1 recipe
# at obs1 first as the control, then the augmentation/scale/backbone variants.
# Each run redirects stdout straight to its own log (no tee pipeline: DRIVER7's tee
# died mid-C2 and the later per-run logs ended up 0 bytes).
cd /workspace/pusht-slim
source .venv/bin/activate
export WANDB_API_KEY=$(tr -d '[:space:]' < ~/.wandb_personal_key)
export SDL_VIDEODRIVER=dummy

COMMON="OBS_HORIZON=1 OBJECTIVE=flow TOTAL_STEPS=100000 EVAL_EVERY=25000 EVAL_EPISODES=100 VISION_NORM=group VISION_PRETRAINED=0"

run_one() {
  local name="$1"; shift
  echo ">>> DRIVER8 starting $name : $* $COMMON @ $(date -u)" >> run_logs/DRIVER8.log
  env $COMMON "$@" RUN_NAME="$name" python -u train.py > "run_logs/$name.log" 2>&1
  local code=$?
  echo "RUN_DONE exit=$code" >> "run_logs/$name.log"
  echo "RUN_DONE exit=$code" > "run_logs/$name.DONE"
  echo ">>> DRIVER8 finished $name exit=$code @ $(date -u)" >> run_logs/DRIVER8.log
}

# C5: obs1 control — C1's winning recipe (200M + rn18-GN-scratch), no aug
run_one C5-200m-rn18gn-obs1-100k HIDDEN_DIM=896 N_LAYERS=12 N_HEADS=8 VISION_BACKBONE=resnet18

# C6: C5 + color-jitter aug (obs1 rerun of the lost C2)
run_one C6-200m-rn18gn-aug-obs1-100k HIDDEN_DIM=896 N_LAYERS=12 N_HEADS=8 VISION_BACKBONE=resnet18 AUGMENT=1

# C7: ~300M transformer + rn18GN + aug (obs1 rerun of the lost C3)
run_one C7-300m-rn18gn-aug-obs1-100k HIDDEN_DIM=1024 N_LAYERS=14 N_HEADS=8 VISION_BACKBONE=resnet18 AUGMENT=1

# C8: 200M + resnet34-GN-scratch (bigger CNN tokenizer) + aug (obs1 rerun of the lost C4)
run_one C8-200m-rn34gn-aug-obs1-100k HIDDEN_DIM=896 N_LAYERS=12 N_HEADS=8 VISION_BACKBONE=resnet34 AUGMENT=1

echo "ALL_DONE @ $(date -u)" > run_logs/DRIVER8-ALL.DONE
