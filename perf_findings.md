# Training Speed: Findings & Optimizations

Branch: `perf/zaringleb/training-speed`. Target: `train.py` (PushT DiT, ResNet34
backbone, hidden=600 / 6 layers / 8 heads, batch size 64 — **batch size held
constant** per request).

All measurements are **real 1000-step training runs on the RTX 4090**, loss
checked against the golden reference
[run `ntq1qdmg`](https://wandb.ai/robot_learning_collective/pushT-slim/runs/ntq1qdmg)
(golden loss @ step: 200≈0.335, 400≈0.307, 600≈0.238, 800≈0.194, 1000≈0.280).

## TL;DR

**Steady-state throughput went 6.5 → 15.2 it/s (2.34×); time/step 154 → 65 ms.**
Loss is unchanged (stays in the golden noise band, bf16 included).

| Run | Change (cumulative)        | it/s | ms/step | compute ms | data ms | speedup |
|-----|----------------------------|-----:|--------:|-----------:|--------:|--------:|
| A   | baseline (current code)    |  6.5 |   154   |    147     |   5.9   | 1.00×   |
| B   | + `torch.compile` (real)   |  7.8 |   129   |    122     |   6.0   | 1.20×   |
| C   | + bf16 autocast + TF32     | 14.1 |    71   |     62     |   8.0   | 2.17×   |
| D   | + `num_workers`/pin/prefetch | 15.2 |  65   |     64     |   0.3   | 2.34×   |

Projected for the full 200k-step run (pure training compute):
**~8.5 h → ~3.7 h, saving ~4.9 h (~57%).** (Eval/checkpoint time unchanged.)

## The headline bug: `torch.compile` was a no-op

The old code did:

```python
policy = torch.compile(policy)
...
loss = policy.training_step(images, obs, actions)
```

`torch.compile(policy)` returns an `OptimizedModule` that only intercepts
`forward`/`__call__`. But training calls `policy.training_step(...)`, and inside
that method `self.forward(...)` resolves to the **original, uncompiled** module.
So compilation never touched the training graph — which is exactly why "adding
torch.compile gave no speedup."

Evidence: in run A (`--compile-mode wrap`, the old behaviour) the first-step cold
start was only **4.5 s** — pure CUDA/cuDNN init, no real compilation. When
compilation actually happens (runs B–D) the first step takes **300–640 s** of
inductor codegen.

Fix: compile the training-step callable so the fwd+bwd graph is compiled:

```python
train_step_fn = torch.compile(policy.training_step)   # compile-mode "step"
```

## Method & instrumentation

Added lightweight, always-on perf logging to `train.py` (console + W&B):

- **Cold start vs iterations** — a one-time line printed after step 1:
  `[perf] first step done: cold_start=… step1_wall=… data=… h2d=… compute=…`
  (`perf/cold_start_s`, `perf/first_step_s` in W&B). This isolates the
  compile/CUDA-init cost from steady-state throughput.
- **Per-window steady-state** (every `--perf-log-every`, default 100):
  `perf/steps_per_s`, `perf/ms_per_step`, and a breakdown into
  `perf/data_ms` (waiting on the dataloader), `perf/h2d_ms` (host→device copy),
  `perf/compute_ms` (fwd+bwd+optimizer, GPU-synced). The cold first step is
  excluded from the averages.

All optimizations are exposed as flags so each lever could be A/B-tested over a
real 1000-step run (see `run_perf_suite.sh`), and the winning combo is now the
default in `PerfConfig`.

## Hypotheses, tested one by one

Going in, compute was expected to dominate; the baseline breakdown confirmed it:
**compute = 147 ms of 154 ms/step (~96%)**, data only ~6 ms. So GPU math was the
bottleneck, not the dataloader (initially).

1. **`torch.compile` no-op (H1).** Confirmed and fixed (above).
   Result: 147 → 122 ms compute, **6.5 → 7.8 it/s (+20%)**.
2. **No mixed precision (H2) + TF32 disabled (H3).** fp32 leaves the 4090's
   tensor cores idle on the ResNet34 convs and attention/MLP matmuls. Enabled
   bf16 autocast (no GradScaler needed) + TF32.
   Result: 122 → 62 ms compute, **7.8 → 14.1 it/s (+81%)** — the biggest win.
   Loss unaffected (bf16 @200: 0.314, golden 0.335).
3. **Synchronous data loading (H4).** Once compute dropped to ~62 ms, the ~8 ms
   serial dataloader fetch (`num_workers=0`) became ~11% of the step. Added
   `num_workers=4`, `pin_memory=True`, `prefetch_factor=4`,
   `persistent_workers=True`, plus `non_blocking` H2D copies.
   Result: data 8.0 → 0.3 ms (fully overlapped), **14.1 → 15.2 it/s (+8%)**.

### Loss parity (sanity vs golden)

| step | golden | A | B | C (bf16) | D (bf16) |
|-----:|-------:|--:|--:|---------:|---------:|
| 200  | 0.335 | 0.289 | 0.274 | 0.314 | 0.382 |
| 400  | 0.307 | 0.303 | 0.291 | 0.274 | 0.219 |
| 600  | 0.238 | 0.197 | 0.227 | 0.274 | 0.267 |
| 800  | 0.194 | 0.225 | 0.194 | 0.219 | 0.214 |
| 1000 | 0.280 | 0.290 | 0.197 | 0.182 | 0.241 |

All within run-to-run noise; **bf16 shows no degradation**.

## Cold start (important caveat on this box)

`torch.compile` adds a large one-time first-step cost. On this filesystem the
inductor C++ codegen is unusually slow (mostly I/O wait): **~300–640 s** observed
(on a fast local SSD this is typically 30–60 s). It is amortized to nothing over
a 200k-step run (~3.7 h), but matters for short/iterative debugging.

Mitigations in place / recommended:
- A **persistent inductor cache** is used in `run_perf_suite.sh`
  (`TORCHINDUCTOR_CACHE_DIR=/workspace/pusht-slim/.inductor_cache`), so restarts
  with the same graph reuse artifacts (run D's cold start dropped to 304 s vs C's
  641 s). Export this in the real training launch too.
- For quick debugging, run with `--compile-mode none` to skip compilation.

## Projected full-run impact (200k steps)

| | it/s | training compute time |
|---|---:|---:|
| Baseline (current) | 6.5 | ~30,800 s (~8.5 h) |
| Optimized | 15.2 | ~13,200 s (~3.7 h) |

**~4.9 h saved (~57% less) per 200k-step run**, plus one ~5–10 min compile cold
start (this box) that is otherwise negligible. Eval and checkpointing are
unchanged.

## What changed in `train.py`

- Compile the **training step** (real fix) instead of wrapping the module
  (`--compile-mode step|wrap|none`, default `step`).
- **bf16 autocast** around fwd/bwd (`--amp bf16|fp16|none`, default `bf16`).
- **TF32** matmul/cuDNN enabled (`--tf32`, default on).
- DataLoader: `num_workers=4`, `pin_memory`, `prefetch_factor=4`,
  `persistent_workers`; `non_blocking` H2D copies;
  `optimizer.zero_grad(set_to_none=True)`.
- Perf instrumentation (cold-start line + per-window throughput/breakdown to
  console & W&B).
- `--steps` flag for short perf runs; `--eval-every 0 / --save-every 0` to disable
  during measurement.

Defaults now produce the optimized config, so `python train.py` Just Works; the
golden recipe is reproduced with `--compile-mode wrap --amp none --no-tf32
--num-workers 0`.

## Not done / further opportunities (no batch-size change)

- **`channels_last`** memory format for the conv backbone (flag already wired,
  untested here) — sometimes +5–15% with AMP.
- **`fused=True` AdamW** — small optimizer-step win.
- **fp16 + GradScaler** instead of bf16 — marginal vs bf16, more fragile; not
  worth it here.
- **SDPA / FlashAttention** in `AdaLNBlock` (replace `nn.MultiheadAttention`)
  — `torch.compile` + bf16 already fuse most of it; revisit if attention shows up.
- **CUDA graphs** (`torch.compile(mode="reduce-overhead")`) — can help at this
  small step time, but increases compile cost and rigidity.
- Batch size is the obvious remaining throughput lever (GPU is only ~7 GB / 24 GB)
  but is **intentionally left unchanged** per request.

---

# Eval-during-training: Findings & Optimizations

Same methodology: real runs that train 500 steps (eager, fast) then run the
in-training `evaluate()` (20 episodes, `n_steps=10`) via `--eval-every 500`.
Added an eval timing breakdown printed after each eval and logged to W&B
(`eval/time_total_s`, `eval/time_inference_s`, `eval/time_env_s`,
`eval/inference_ms_per_call`, `eval/env_ms_per_step`, …):

```
[eval] 20 eps, n_steps=10: total=… | inference=… (…ms x N) env=… (…ms x M) setup=… other=…
```

## TL;DR

**Kept one cheap change: a context cache in the denoising loop → eval
113.9 s → 74.0 s for 20 episodes (1.54×)**, numerically identical, ~zero added
complexity.

Two further ideas were measured but **deliberately not kept**: batched
parallel-env eval (would give 74.0 → 26.8 s) and bf16-in-eval (rejected, slower
at batch-1). Rationale: eval is a tiny slice of a ~3.7 h training run, so the few
minutes saved don't justify the code complexity (vector envs, subprocess
lifecycle, autoreset masking, extra flags). See "Measured but not kept" below.

| Run | eval total | per ep | inference (ms/call) | env | setup |
|-----|-----------:|-------:|--------------------:|----:|------:|
| A baseline                 | 113.9 s | 5.70 s | 78.4 s (103.1) | 19.0 s | 16.2 s |
| B + context cache (fp32)   |  74.0 s | 3.70 s | 38.5 s (50.7)  | 16.5 s | 18.6 s |
| C + bf16 autocast          |  88.5 s | 4.42 s | 52.0 s (68.4)  | 16.1 s | 20.0 s |

## Where eval time goes

Baseline breakdown: **inference 69%**, env-step 17%, env setup 14%. So eval is
**inference-bound**, not physics-bound. With 6000 env steps and chunk size 8 that's
760 replans, each running 10 denoising forwards = 7600 model forwards, all at
**batch size 1**.

## The fix: cache the observation context across the denoising loop

`inference()` ran the full `forward()` every denoising step, and `forward()`
re-encodes the image (ResNet34) + state every call — but that context depends only
on the observation, **not** on the noisy action `x` or the flow time `t`, so it's
identical across all 10 steps. Refactored the model into:

- `encode_context(images, obs)` — the expensive backbone, run **once** per replan;
- `forward_with_context(x_t, t, context)` — the cheap per-step part (timestep
  embed + DiT blocks), run 10×.

`forward()` and `training_step()` are unchanged in behaviour (numerically
identical), so **loss and success metrics are unaffected** — only redundant work
is removed. Inference dropped 103.1 → 50.7 ms/call (~2×; it's not 10× because the
6 DiT blocks are also a meaningful share of each forward, and only the backbone is
cached).

## Rejected: bf16 autocast in eval

Adding bf16 autocast to the denoising loop made inference **slower**
(50.7 → 68.4 ms/call). At batch size 1 the forwards are launch/latency-bound, not
throughput-bound, so the per-op cast overhead outweighs any tensor-core gain.
Eval now defaults to fp32 (`--eval-amp none`); the flag remains for experiments.

## Projected impact

For the default schedule (eval every 25k → 8 evals over 200k steps):
**~15.2 min → ~9.9 min of eval (~5.3 min saved)**. This is a worst-case estimate:
the runs use a barely-trained policy whose episodes run to truncation (max length);
a competent policy ends successful episodes early, so absolute eval time is lower
while the ~1.5× speedup holds.

## What changed (eval)

- `DiTPolicy.encode_context` / `forward_with_context`; `inference()` encodes the
  observation context once and reuses it across denoising steps (the only kept
  eval change — numerically identical, ~zero added complexity).
- A single lightweight timing line in `evaluate()` (`[eval] N eps … Xs`, also
  logged as `eval/time_total_s`).

## Projected impact (eval, full run)

8 evals over 200k steps (eval every 25k):

| Eval | per eval | 8 evals |
|------|---------:|--------:|
| original loop (pre-changes)    | 113.9 s | ~15.2 min |
| **context-cache loop (kept)**  |  74.0 s |  **~9.9 min** |

## Measured but not kept (low ROI vs complexity)

These were prototyped and benchmarked, then reverted to keep `evaluate()` simple.
Recorded here so the analysis isn't lost:

- **Batched parallel-env eval** (`AsyncVectorEnv`, one env per episode, batched
  policy forward): **74.0 → 26.8 s (2.76×; 4.25× vs the original 113.9 s)** —
  inference 38.5 → 3.8 s (batch-20 forwards), physics 16.5 → 3.7 s (parallel
  processes). `sync` variant: 34.9 s. After this, env *construction* (~19 s for 20
  envs) dominates, and reusing the vector env across evals could reach ~8 s/eval.
  Dropped because it adds vector-env handling, subprocess lifecycle, `NEXT_STEP`
  autoreset masking, and extra flags for only a few minutes saved per multi-hour
  run.
- **bf16 autocast in eval**: rejected — *slower* (50.7 → 68.4 ms/call). At batch-1
  the denoising forwards are launch/latency-bound, so cast overhead outweighs any
  tensor-core gain.

## Further opportunities (eval)

- **Fewer denoising steps** at eval (sweep `n_steps`) trades success rate for
  speed; worth a quality/speed Pareto check if eval cost ever matters.
- If eval frequency is increased a lot, revisit the batched parallel-env approach
  above (it's a clean ~4× when it's worth the complexity).
