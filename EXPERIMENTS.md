# PushT DiT flow-matching — experiment report

Goal: push success rate (SR) from the ~0.55 plateau toward the diffusion-policy
target (~0.80). Symptom to attack: EMA runs reach **avg_max_reward ≈ 0.98** but
SR stalls ≈ 0.5 — the policy gets ~97% of the way but fails the final settle.

Branch: `feat/zaringleb/claude-flow-experiments`. Each experiment is one commit +
one wandb run. Eval = 50 episodes (SR noise at 20 eps was ±0.11), EMA weights.

## Setup / what was already known (from prior runs, wandb `pushT-slim`)
- Baseline (obs=1, flow 10-step, resnet34, adaLN): SR 0.55, AMR 0.86 (`abl0-baseline`).
- +EMA+cosine LR: **AMR 0.98**, SR 0.5 (`ema-cosine_lr`) — best AMR seen.
- Already tried and *worse*: logit-normal t (0.15), beta t (0.35), DINOv3 backbone
  (0.16–0.42), encoder-LR 10% (0.44), 200k steps (0.55, no gain over 50k).

## Reference findings (lerobot + papers)
- lerobot Diffusion Policy (~80% ref): **n_obs_steps=2**, **DDPM epsilon, 100 train steps**,
  cosine beta, clip_sample, resnet18+GroupNorm, global cond vector.
- DiT-Block Policy (2410.10088): **global adaLN-Zero conditioning beats cross-attention
  and in-context tokens** (the latter ~16% worse / unstable). Current code already uses
  global adaLN. Also: putting compute in the CNN tokenizer beats scaling the transformer
  in low-data regimes → temper expectations for a bigger transformer.
- HRI FM PushT (2409.01083): obs=1, cross-attn, uniform t, **Euler 16 steps**, ~0.9.
- ManiFlow (2509.01819): cross-attn context + time-only adaLN; consistency-flow for
  few-step precision (the "less simple" upgrade — kept in back pocket).

## Plan
| # | Change (delta vs baseline) | Rationale |
|---|---|---|
| B0 | EMA + cosine LR (obs=1, flow) | strong control, reproduce AMR 0.98 |
| E1 | obs_horizon=2 | velocity/temporal context (lerobot default) |
| E2 | DDPM diffusion objective, 100 steps (obs=1) | lerobot's 80% objective |
| E3 | obs=2 + best objective | closest to lerobot DP recipe |
| E4 | bigger model (~200M) on best config | user request; scaling test |
| E5 | vision as in-context tokens | user hypothesis (papers predict worse) |
| E6 | flow inference-steps sweep (10/16/50/100) | cheap precision test |

## Results
| # | run name | wandb id | config | SR | AMR | notes |
|---|---|---|---|---|---|---|
| B0 | B0-baseline-ema-cosine | (running) | obs1, flow-10 | – | – | control |

_(updated as runs finish)_

## Conclusions
_(pending)_
