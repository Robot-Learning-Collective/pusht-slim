# pusht-slim

A minimal, single-file imitation-learning pipeline for the [PushT](https://github.com/huggingface/gym-pusht) task, in the spirit of [nanoGPT](https://github.com/karpathy/nanoGPT): dataset, model, training loop, and eval all live in `train.py` (~600 lines), configured by plain constants at the top of the file.

The policy is a ~200M-parameter [DiT](https://arxiv.org/abs/2212.09748) trained with [flow matching](https://arxiv.org/abs/2210.02747) on the [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) demonstrations. It reaches a success rate of ~0.6–0.7 after 100k steps (~3.5 h on an RTX 4090).

## How it works

- **Data** — the hub dataset is just an MP4 (every frame of every episode, concatenated) plus a parquet with per-frame state/action/episode metadata. `PushTDataset` decodes the video once (~7 s) and keeps everything in RAM (~0.7 GB as uint8), so `__getitem__` is pure tensor slicing. Data is served raw — the model owns all normalization. No lerobot dependency.
- **Observation encoding** — frames are cropped to 84×84 (random in train, center in eval) and passed through a pretrained DINOv2 ViT-S/14, fine-tuned end-to-end. Its 6×6 patch grid is pooled by SpatialSoftmax into 32 keypoints. The 2-DoF robot state goes through a small MLP.
- **Action generation** — a DiT with adaLN conditioning (image + state + flow time) predicts the velocity field of a flow from Gaussian noise to a 16-step action chunk. Inference is 10 Euler steps.
- **Control** — receding horizon: predict 16 actions, execute 8, replan.
- **Eval** — every 10k steps the policy is rolled out in `gym-pusht` for 50 episodes; success rate, average max reward, and 3 rollout videos are logged to wandb.

## Run

```bash
pip install -r requirements.txt
export WANDB_API_KEY=...   # metrics and eval videos go to wandb
python train.py
```

Set `WANDB_ENTITY` / `WANDB_PROJECT` at the top of `train.py` to your own wandb account. All hyperparameters are constants in the same block. Checkpoints land in `checkpoints_<run-name>/`.

For running on a cloud GPU box, see [runpod_setup.md](runpod_setup.md).

## State-only training

`train_state.py` trains a 418k-parameter flow policy from robot `(x, y)` and
T `(x, y, angle)`. It loads no images and renders no evaluation videos.
The default standard PushT loader recovers exact T poses from the matching
`lerobot/pusht_keypoints` dataset and verifies their geometry and row alignment.

```bash
pip install -r requirements-state.txt
python train_state.py --steps 60000 --output checkpoints_state_run1
```

The trainer selects CUDA when available (CUDA graphs and fused AdamW), otherwise
CPU. Use `--device cpu` or `--no-cuda-graphs` to select the eager path. GPU and
CPU arithmetic can produce slightly different policies. The graph warmup restores
model, EMA and optimizer state before the first training update.

Evaluation defaults to **50 fixed-seed episodes every 10,000 updates and at the
last update**, saving `best.pt` and `last.pt`. The best checkpoint is then tested
on 100 separate fixed-seed episodes (`--test-episodes`, `--test-seed`). Evaluation
uses batched CPU inference and the ordinary headless simulator.

For a single evaluation of the final model on 200 freshly randomized episodes:

```bash
python train_state.py --steps 60000 --eval-every 0 --eval-episodes 200 \
  --random-eval-seeds --output checkpoints_state_run2
```

`--eval-every 0` disables intermediate evaluation and best-checkpoint selection;
only `last.pt` is written and evaluated once. Without `--random-eval-seeds`, reset
and policy-noise seeds stay fixed. With it, each evaluation samples fresh seeds
independently of the training seed. `evaluations.jsonl` records every evaluation's
reset/noise seeds, initial states, episode lengths, successes and coverage. Pass
one saved record as `replay=` to `eval_state.evaluate` to reproduce those conditions.
Every rollout runs until success or the simulator's full episode time limit.

W&B defaults remain `robot_learning_collective/pushT-slim`; `--entity` and
`--project` override them. `--wandb-mode disabled` supports local checks. Results
and model/data configuration are also saved locally. The original visual trainer
and its defaults are unchanged.

For reconstructed mobile data, install the `pusht_mobile_env` Python package and
its Rust extension from the mobile simulator project, then run:

```bash
python train_state.py --env-id PushTMobile-v0 --dataset-id zaringleb/pusht_mobile \
  --state-dataset /path/to/reconstruction/states.parquet \
  --steps 60000 --eval-every 0 --eval-episodes 200 --random-eval-seeds \
  --output checkpoints_mobile_run1
```

The reconstruction directory must include `manifest.json` and `audit.json`.
The loader checks the state-file checksum and verifies agent coordinates/actions
against the pinned Hub dataset. An adjacent `source.parquet` avoids the Hub
download and is checked against the manifest checksum. Reconstructed T poses
are estimates. Data, checkpoints and compiled simulator wheels are not included
in this repository.

Learning rate uses warmup and cosine decay. `--decay-steps` optionally separates
the decay horizon from the training budget; omitting it preserves the usual
full-budget decay. `--duration-minutes` instead uses wall-clock decay and includes
periodic evaluations in its budget. No historical run logs are required.

```bash
python -m unittest test_state test_state_options -q
```
