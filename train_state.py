"""State-only flow matching for standard or mobile PushT.

Standard PushT uses lerobot/pusht plus exact poses recovered from its keypoint
companion. Mobile PushT uses an audited reconstructed state Parquet. No images or videos are loaded, rendered, or used by the policy.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import pandas as pd
import torch
from torch import nn
import wandb

from eval_state import evaluate


# gym-pusht uses a body origin at the top edge of the T, not its centroid.
KEYPOINTS_BODY = np.array([[-60, 0], [60, 0], [60, 30], [-60, 30],
                          [-15, 30], [15, 30], [15, 120], [-15, 120]], dtype=np.float64)


def poses_from_keypoints(keypoints):
    """Rigid 2D alignment of simulator coordinates; no visual estimation."""
    points = np.asarray(keypoints, dtype=np.float64).reshape(-1, 8, 2)
    center = points.mean(axis=1)
    centered = points - center[:, None]
    template_center = KEYPOINTS_BODY.mean(axis=0)
    template = KEYPOINTS_BODY - template_center
    dot = (centered * template).sum(axis=(1, 2))
    cross = (template[:, 0] * centered[:, :, 1]
             - template[:, 1] * centered[:, :, 0]).sum(axis=1)
    angle = np.arctan2(cross, dot)
    c, s = np.cos(angle), np.sin(angle)
    position = center - np.column_stack([
        c * template_center[0] - s * template_center[1],
        s * template_center[0] + c * template_center[1]])
    reconstructed = np.stack([
        c[:, None] * KEYPOINTS_BODY[:, 0] - s[:, None] * KEYPOINTS_BODY[:, 1],
        s[:, None] * KEYPOINTS_BODY[:, 0] + c[:, None] * KEYPOINTS_BODY[:, 1],
    ], axis=-1) + position[:, None]
    error = float(np.linalg.norm(reconstructed - points, axis=-1).max())
    if not np.isfinite(error) or error > 1e-3:
        raise ValueError(f"T keypoints do not match the simulator geometry: {error=}")
    return np.column_stack([position, angle]).astype(np.float32), error


def load_standard_data(dataset_id, revision, horizon, seed, keypoints_revision=None,
                       representation="pose"):
    api = HfApi(token=False)
    def read(repo, rev):
        rev = rev or api.dataset_info(repo).sha
        files = api.list_repo_files(repo, repo_type="dataset", revision=rev)
        tables = [pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset",
                    revision=rev, token=False)) for f in files
                  if f.startswith("data/") and f.endswith(".parquet")]
        table = pd.concat(tables).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        return table, rev
    table, revision = read(dataset_id, revision)
    companion_id = "lerobot/pusht_keypoints"
    companion, keypoints_revision = read(companion_id, keypoints_revision)
    for column in ["episode_index", "frame_index"]:
        np.testing.assert_array_equal(table[column], companion[column])
    for column in ["observation.state", "action"]:
        np.testing.assert_array_equal(np.stack(table[column]), np.stack(companion[column]))
    keypoints = np.stack(companion["observation.environment_state"])
    poses, error = poses_from_keypoints(keypoints)
    block = poses if representation == "pose" else keypoints
    states = np.concatenate([np.stack(table["observation.state"]), block], axis=-1).astype(np.float32)
    raw_actions = np.stack(table["action"]).astype(np.float32)
    episode_ids = table["episode_index"].to_numpy()
    if not np.isfinite(states).all() or not np.isfinite(raw_actions).all():
        raise ValueError("Nonfinite demonstration data")
    indices = np.arange(len(table))[:, None] + np.arange(horizon)[None]
    in_bounds = indices < len(table)
    indices = indices.clip(max=len(table) - 1)
    valid = in_bounds & (episode_ids[indices] == episode_ids[:, None])
    masks = np.repeat(valid[:, :, None], 2, axis=-1).astype(np.float32)
    actions = raw_actions[indices] * masks
    ids = np.unique(episode_ids)
    tensors = tuple(torch.from_numpy(a) for a in (states, actions, masks))
    train = tensors
    val = None
    metadata = dict(dataset_id=dataset_id, dataset_revision=revision,
                    keypoints_dataset_id=companion_id, keypoints_revision=keypoints_revision,
                    pose_max_keypoint_error_px=error, total_episodes=len(ids),
                    total_frames=len(states), train_frames=len(train[0]),
                    val_frames=0, val_episode_ids=[],
                    observation_fields=(["agent_x", "agent_y", "block_x", "block_y", "block_angle"]
                        if representation == "pose" else ["agent_x", "agent_y"] +
                        [f"keypoint_{i}_{axis}" for i in range(8) for axis in ("x", "y")]),
                    angle_units="radians", images=False, action_offset=0)
    return train, val, metadata


def load_reconstructed_data(path, dataset_id, revision, horizon):
    """Load all reconstructed episodes; evaluate on independent simulator resets.

The adjacent audit/manifest bind the state estimates to the pinned Hub data.
Verify the original robot positions/actions before constructing masked chunks.
Conservative pixel flags are diagnostics, not reasons to drop demonstrations.
"""
    manifest = json.loads((path.parent / "manifest.json").read_text())
    audit = json.loads((path.parent / "audit.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != audit["artifact_sha256"][path.name]:
        raise ValueError("Reconstructed state file does not match its audit checksum")
    if manifest["dataset_id"] != dataset_id or (revision and revision != manifest["revision"]):
        raise ValueError("Reconstructed states belong to a different dataset/revision")
    revision = manifest["revision"]
    table = pd.read_parquet(path).sort_values("index").reset_index(drop=True)
    source_path = path.parent / "source.parquet"
    if not source_path.exists():
        source_path = Path(hf_hub_download(dataset_id, "data/chunk-000/file-000.parquet",
                           repo_type="dataset", revision=revision, token=False))
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != manifest["parquet_sha256"]:
        raise ValueError("Source actions do not match the pinned dataset checksum")
    source = pd.read_parquet(source_path).sort_values("index").reset_index(drop=True)
    states = np.stack(table["observation.state"]).astype(np.float32)
    actions = np.stack(table["action"]).astype(np.float32)
    if states.shape != (len(source), 5) or actions.shape != (len(source), 2):
        raise ValueError("Expected five state inputs and two action coordinates")
    for column in ["index", "episode_index", "frame_index", "timestamp"]:
        np.testing.assert_array_equal(table[column], source[column])
    np.testing.assert_array_equal(states[:, :2], np.stack(source["observation.state"]))
    np.testing.assert_array_equal(actions, np.stack(source["action"]))
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite reconstructed data")
    if not np.all(np.abs(states[:, 4]) <= np.pi):
        raise ValueError("Expected block angles in [-pi, pi]")
    episode_ids = table.episode_index.to_numpy()
    indices = np.arange(len(table))[:, None] + np.arange(horizon)[None]
    in_bounds = indices < len(table)
    indices = indices.clip(max=len(table) - 1)
    valid = in_bounds & (episode_ids[indices] == episode_ids[:, None])
    masks = np.repeat(valid[:, :, None], 2, axis=-1).astype(np.float32)
    targets = actions[indices] * masks
    train = tuple(torch.from_numpy(a) for a in (states, targets, masks))
    metadata = dict(dataset_id=dataset_id, dataset_revision=revision,
        reconstructed_states_path=str(path.resolve()), reconstructed_states_sha256=digest,
        pose_source="inverse-rendering estimates", total_episodes=len(np.unique(episode_ids)),
        total_frames=len(table), train_frames=len(table), val_frames=0, val_episode_ids=[],
        validation_method="independent simulator resets",
        observation_fields=["agent_x", "agent_y", "block_x", "block_y", "block_angle"],
        angle_units="radians", images=False, action_offset=0)
    return train, None, metadata


class StateFlowPolicy(nn.Module):
    def __init__(self, hidden=256, layers=3, horizon=16, representation="pose"):
        super().__init__()
        self.horizon = horizon
        self.representation = representation
        self.register_buffer("time_freq", 2 * math.pi * 2 ** torch.arange(8).float())
        self.input = nn.Linear(horizon * 2 + (5 if representation == "pose" else 18) + 17, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                          nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(layers)
        ])
        self.output = nn.Linear(hidden, horizon * 2)

    def forward(self, noisy_actions, t, state):
        if self.representation == "pose":
            # Exactly five state features: four coordinates plus one wrapped angle.
            angle = torch.remainder(state[:, 4:] + math.pi, 2 * math.pi) - math.pi
            state = torch.cat([state[:, :4] / 256 - 1, angle / math.pi], dim=-1)
        else:
            state = state / 256 - 1
        phase = t[:, None] * self.time_freq
        # Include raw time so t=0 and t=1 remain distinct; integer-frequency
        # Fourier features alone are periodic over the entire flow interval.
        x = self.input(torch.cat([noisy_actions.flatten(1), state, t[:, None],
                                  phase.sin(), phase.cos()], dim=-1))
        for block in self.blocks:
            x = x + block(x)
        return self.output(x).reshape(-1, self.horizon, 2)

    @torch.no_grad()
    def inference(self, state, steps=10, generator=None):
        x = torch.randn(len(state), self.horizon, 2, generator=generator)
        for i in range(steps):
            t = torch.full((len(state),), i / steps)
            x = x + self(x, t, state) / steps
        return (x.clamp(-1, 1) + 1) * 256


def flow_loss(model, batch, generator=None):
    state, action, mask = batch
    target = action / 256 - 1
    noise = torch.randn(target.shape, generator=generator).to(target.device)
    t = torch.rand(len(target), generator=generator).to(target.device)
    noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * target
    error = (model(noisy, t, state) - (target - noise)).square()
    return (error * mask).sum() / mask.sum()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dataset", type=Path, help="Audited reconstructed mobile states.parquet")
    p.add_argument("--dataset-id", default="lerobot/pusht")
    p.add_argument("--env-id", choices=["gym_pusht/PushT-v0", "PushTMobile-v0"], default="gym_pusht/PushT-v0")
    p.add_argument("--revision")
    p.add_argument("--keypoints-revision")
    p.add_argument("--representation", choices=["pose", "keypoints"], default="pose")
    budget = p.add_mutually_exclusive_group()
    budget.add_argument("--steps", type=int)
    budget.add_argument("--duration-minutes", type=float,
                        help="Wall-clock training budget including periodic eval; final evaluation runs afterward")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--action-chunk", type=int, default=8)
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=10000, help="0 evaluates only at the end")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--test-seed", type=int, default=20000)
    p.add_argument("--test-episodes", type=int, default=100,
                   help="Final test of the selected best checkpoint in periodic mode")
    p.add_argument("--random-eval-seeds", action="store_true",
                   help="Sample fresh reset/noise seeds for every evaluation")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--decay-steps", type=int,
                   help="Optional cosine decay horizon; defaults to the training step budget")
    p.add_argument("--entity", default="robot_learning_collective")
    p.add_argument("--project", default="pushT-slim")
    p.add_argument("--name", default="pusht-state-cpu")
    p.add_argument("--group", default="pusht-state-representations-cpu")
    p.add_argument("--output", type=Path, default=Path("checkpoints_state"))
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    p.add_argument("--wandb-offline-fallback", action="store_true",
                   help="Preserve the training run offline if online W&B initialization fails")
    args = p.parse_args(argv)
    if args.steps is None and args.duration_minutes is None:
        args.steps = 10000
    if args.steps is not None and args.steps < 1:
        p.error("steps must be positive")
    if args.duration_minutes is not None and (not math.isfinite(args.duration_minutes) or args.duration_minutes <= 0):
        p.error("duration-minutes must be finite and positive")
    if not 1 <= args.action_chunk <= args.horizon:
        p.error("action-chunk must be between 1 and horizon")
    if min(args.batch_size, args.eval_episodes, args.test_episodes, args.threads,
           args.hidden, args.layers, args.denoise_steps) < 1:
        p.error("counts must be positive")
    if args.eval_every < 0:
        p.error("eval-every must be nonnegative")
    if args.decay_steps is not None and (args.decay_steps < 1 or args.duration_minutes is not None):
        p.error("decay-steps must be positive and requires a step budget")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.cuda_graphs = args.cuda_graphs and args.device == "cuda"
    if (args.output / "config.json").exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.env_id == "PushTMobile-v0":
        if args.representation != "pose":
            raise ValueError("mobile mode supports pose representation only")
        if args.dataset_id != "zaringleb/pusht_mobile":
            raise ValueError("mobile env requires --dataset-id zaringleb/pusht_mobile")
        if args.state_dataset is None:
            raise ValueError("mobile env requires --state-dataset")
        train, _, metadata = load_reconstructed_data(args.state_dataset, args.dataset_id,
                                                     args.revision, args.horizon)
    else:
        if args.state_dataset is not None:
            raise ValueError("--state-dataset requires --env-id PushTMobile-v0")
        if args.dataset_id != "lerobot/pusht":
            raise ValueError("standard env requires --dataset-id lerobot/pusht")
        train, _, metadata = load_standard_data(args.dataset_id, args.revision,
                    args.horizon, args.seed, args.keypoints_revision, args.representation)
    model = StateFlowPolicy(args.hidden, args.layers, args.horizon, args.representation).to(args.device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4,
        **({"capturable": True, "fused": True} if args.cuda_graphs else {}))
    graph_step = None
    if args.cuda_graphs:
        from cuda_step import GraphStep
        graph_step = GraphStep(model, ema, optimizer, args.batch_size)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(metadata, parameters=sum(p.numel() for p in model.parameters()),
                  device=args.device, ema_decay=0.995, model_version=2,
                  schedule_basis="wall_time" if args.duration_minutes else "steps")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "config.json").exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    try:
        run = wandb.init(entity=args.entity, project=args.project, name=args.name,
                         group=args.group, config=config, mode=args.wandb_mode)
    except wandb.errors.Error as error:
        if args.wandb_mode != "online" or not args.wandb_offline_fallback:
            raise
        print(f"Online W&B initialization failed: {error}. Continuing with offline logging.", flush=True)
        args.wandb_mode = "offline"
        config["wandb_mode"] = "offline"
        (args.output / "config.json").write_text(json.dumps(config, indent=2))
        run = wandb.init(entity=args.entity, project=args.project, name=args.name,
                         group=args.group, config=config, mode="offline")
    (args.output / "run.json").write_text(json.dumps(
        {"url": run.url, "id": run.id, "directory": run.dir, "mode": args.wandb_mode}, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    def evaluate_and_record(policy, episodes, seed, step, phase):
        metrics, details = evaluate(policy, episodes, args.action_chunk, args.denoise_steps,
            seed=seed, env_id=args.env_id, random_seeds=args.random_eval_seeds, return_details=True)
        with (args.output / "evaluations.jsonl").open("a") as log:
            log.write(json.dumps(dict(step=step, phase=phase, metrics=metrics, **details)) + "\n")
        return metrics

    best_score = (-float("inf"), -float("inf"))
    start = time.monotonic()
    duration_seconds = args.duration_minutes * 60 if args.duration_minutes else None
    window_loss = 0.
    # Separate streams keep training batches/noise identical across representations,
    # even though initializing differently sized input layers consumes different RNG.
    batch_generator = torch.Generator().manual_seed(args.seed + 1)
    loss_generator = torch.Generator().manual_seed(args.seed + 2)
    try:
        for step in itertools.count(1):
            indices = torch.randint(len(train[0]), (args.batch_size,), generator=batch_generator)
            batch = tuple(t[indices] for t in train)
            warmup = min(200, max(1, args.steps // 10)) if args.steps else 200
            progress = (min(1., (time.monotonic() - start) / duration_seconds)
                        if duration_seconds else min(1., max(0, step - warmup) / max(1, (args.decay_steps or args.steps) - warmup)))
            lr = args.lr * min(1., step / warmup) * (0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2)
            if graph_step is not None:
                loss_value, grad_value = graph_step.step(batch, loss_generator, lr)
            else:
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss = flow_loss(model, tuple(t.to(args.device) for t in batch), loss_generator)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}")
                loss.backward()
                grad = nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                with torch.no_grad():
                    for target, source in zip(ema.parameters(), model.parameters()):
                        target.lerp_(source, 0.005)
                loss_value, grad_value = loss.item(), grad.item()
            window_loss += loss_value
            final_step = (time.monotonic() - start >= duration_seconds
                          if duration_seconds else step >= args.steps)
            metrics = {}
            if step % 100 == 0 or final_step:
                metrics = {"train/loss": window_loss / (step % 100 or 100),
                           "train/lr": lr, "train/grad_norm": grad_value,
                           "time/steps_per_second": step / (time.monotonic() - start),
                           "time/elapsed_seconds": time.monotonic() - start}
                window_loss = 0.
            evaluate_now = args.eval_every > 0 and (step % args.eval_every == 0 or final_step)
            if evaluate_now:
                metrics.update(evaluate_and_record(ema, args.eval_episodes, 10000, step, "validation"))
            if evaluate_now or final_step:
                checkpoint = dict(step=step, model_state_dict=ema.state_dict(),
                                  raw_model_state_dict=model.state_dict(),
                                  optimizer_state_dict=optimizer.state_dict(), config=config,
                                  metrics=metrics, torch_rng_state=torch.get_rng_state())
                torch.save(checkpoint, args.output / "last.pt")
                score = ((metrics["eval/success_rate"], metrics["eval/mean_max_coverage"])
                         if evaluate_now else best_score)
                if evaluate_now and score > best_score:
                    best_score = score
                    torch.save(checkpoint, args.output / "best.pt")
                    run.summary["best_eval_success_rate"] = best_score[0]
                    run.summary["best_step"] = step
            if metrics:
                print(f"step {step}: {json.dumps(metrics)}", flush=True)
                run.log(metrics, step=step)
            if final_step:
                break
        run.summary["checkpoint_dir"] = str(args.output.resolve())
        run.summary["completed_steps"] = step
        # Test the best checkpoint in periodic mode, or the final checkpoint in end-only mode.
        selected = torch.load(args.output / ("best.pt" if args.eval_every else "last.pt"),
                              map_location=args.device, weights_only=False)
        ema.load_state_dict(selected["model_state_dict"])
        test_metrics = {key.replace("eval/", "test/"): value for key, value in
                        evaluate_and_record(ema, args.test_episodes if args.eval_every else args.eval_episodes,
                                            args.test_seed, selected["step"], "test").items()}
        test_metrics.update(selected_step=selected["step"], completed_steps=step, test_seed_start=None if args.random_eval_seeds else args.test_seed,
                            elapsed_seconds=time.monotonic() - start)
        run.log(test_metrics, step=step)
        run.summary.update(test_metrics)
        (args.output / "results.json").write_text(json.dumps(test_metrics, indent=2))
        print(f"FINAL: {json.dumps(test_metrics)}", flush=True)
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


if __name__ == "__main__":
    main()
