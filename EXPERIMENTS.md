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
| E6 | flow inference-steps + action-chunk sweep | cheap closed-loop/precision test |
| E7 | ResNet18 + GroupNorm, random init (orig. Diffusion Policy encoder) | vision-encoder is prime plateau suspect (user request) |
| E8 | relative/delta actions, per-timestep z-score norm (TRI) | final-precision lever (user request) |

## Results
SR is noisy even at 50 eps — B0's per-eval SR ranged 0.36–0.58 within one run. So I
report **best / last** SR across evals and lean on AMR (low variance) for ranking.

| # | run name | wandb id | config | best SR | last SR | best AMR | notes |
|---|---|---|---|---|---|---|---|
| B0 | B0-baseline-ema-cosine | 0c3czg0e | obs1, flow-10, 50k | 0.58 | 0.36 | 0.95 | control; SR 0.44/0.58/0.40/0.48/0.36 (bouncy) |
| E1 | E1-obs2-flow | 7pd6cicq | obs2, flow-10, 50k | 0.54 | 0.54 | 0.89 | SR 0.48/0.52/0.52/0.50/0.54 — **stable ~0.52**, mean > B0's ~0.45 but plateau holds |
| E2 | E2-obs1-diffusion | 6bz09nbu | obs1, DDPM-100, 50k | 0.54 | 0.54 | 0.96 | 25k 0.48/0.935 → 50k 0.54/0.960. **Highest AMR**; DDPM sampler works; still plateau |
| E3 | E3-obs2-diffusion | hvmqdz3y | obs2, DDPM-100, 50k | 0.45 | 0.45 | 0.93 | lerobot-recipe combo; **plateau holds** (40-ep final eval) |
| E5 | E5-vision-tokens | (wandb) | obs2, flow, tokens, 30k | 0.00 | 0.00 | 0.18 | **FAILS** (random); loss trains fine (0.41→0.095) but policy unusable — matches DiT-Block ablation: in-context ≈0%. Confirms global adaLN is right |
| E4 | E4-big200m-flow | 9tr0zfle | obs2, flow, 221M, 25k | 0.60 | 0.60 | 0.905 | 12.5k 0.52/0.878 → 25k **0.60**/0.905. **Best SR so far**; 200M lifted SR above the plateau even at only 25k steps (vs 50k elsewhere) |
| E7 | E7-resnet18-gn-scratch | (queued) | resnet18+GN+scratch, obs2, flow, 40k | – | – | – | orig-DP encoder (user); queued after E4 |
| E8 | E8-relative-actions | (queued) | relative/per-ts-norm, obs2, flow, 40k | – | – | – | TRI delta actions (user); queued after E7 |

**Time note:** 100-step DDPM eval is ~45 min/eval (batch-1 rollouts), so E2 took ~2h48m.
To fit the budget I gave E3 a single final eval and run E5/E4 on flow (10-step eval, ~10× faster)
at reduced steps (signal is clear by ~25-30k).

**Note (eval bug, fixed):** early E1 eval showed SR 0.0 with healthy training loss — the eval
adapter stacked obs at replan cadence (8 env steps apart) vs training's consecutive frames,
so obs=2 read an 8× bogus velocity. Fixed to update history every env step (commit 9222561).
obs=1 runs (B0, E2) were unaffected.

_(updated as runs finish)_

## Analysis — what the plateau actually is
gym_pusht: `reward = clip(coverage/0.95, 0, 1)` and `terminated = is_success = coverage > 0.95`
(pusht.py:257-258). So the episode **ends the moment coverage exceeds 95%**, and
`max_reward == 1.0 ⟺ success`. Therefore AMR ≈ 0.96 with SR ≈ 0.54 means: ~half the episodes
reach 1.0 (success), the other half **stall just below 95% coverage** and never close the last
few percent. This is a genuine **final-precision** gap (can't nudge coverage 0.90→0.95), not
drift away from a reached goal.

Evidence on *which* lever matters:
- Objective (flow vs DDPM), obs horizon (1 vs 2), and 200k-vs-50k steps all land at ~0.5 SR.
- **E2 (DDPM, 100 denoising steps) also plateaus at 0.54** → denoising/inference precision is
  NOT the bottleneck.
- Both known ~80% references differ from us most in the **vision encoder**: lerobot DP uses
  ResNet18 + **GroupNorm, trained from scratch**; we use ResNet34 + **BatchNorm + ImageNet
  pretrained**. BatchNorm's train/eval running-stat mismatch and ImageNet features on synthetic
  PushT frames are the prime remaining suspects for the missing final precision.

## Conclusions
_(pending E3/E4/E5/E6; preliminary: the ~0.5 SR plateau is a last-5%-coverage precision gap that
obs-horizon, objective, and denoising steps do not fix — vision encoder (GroupNorm+from-scratch)
and closed-loop cadence are the top untested levers.)_

## Recommended next experiments (if this direction continues)
1. **Vision encoder = ResNet18 + GroupNorm, trained from scratch** (lerobot DP's exact choice) —
   highest-value untested lever for the final-precision gap.
2. **Smaller action chunk at eval** (replan every 1–4 steps) — cheap closed-loop test (see E6).
3. **Temporal action ensembling** (ManiFlow tip) for smoother final settling.
