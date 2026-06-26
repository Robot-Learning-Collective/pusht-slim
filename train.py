import argparse
import math
import os
from contextlib import nullcontext
from datetime import datetime
import time

import wandb
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import gymnasium as gym
import gym_pusht  

ROBOT_DOF = 2
PREDICTION_HORIZON = 16
ACTION_CHUNK_SIZE = 8
N_DENOISING_STEPS = 10
DATASET_ID = "lerobot/pusht"
# DiT sized to ~47.5M (matching lerobot's DiT backbone): hidden 600, 6 layers, 8 heads
N_HEADS = 8
N_LAYERS = 6
HIDDEN_DIM = 600
TOTAL_STEPS = 200_000
LR = 1e-4
LOG_EVERY = 200
EVAL_EVERY = 25_000
EVAL_EPISODES = 20
SAVE_EVERY = 25_000
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE = 64
WANDB_PROJECT = "pushT-slim"
WANDB_ENTITY = "robot_learning_collective"

# PushT positions and actions are pixel coordinates in [0, 512]
# We normalize everything the model sees to [-1, 1]
# so it lives at the same scale as the Gaussian noise used by flow matching,
# then unnormalize the model's output before sending it back to the env.
COORD_MIN, COORD_MAX = 0.0, 512.0


from dataclasses import dataclass


@dataclass
class PerfConfig:
    """Toggles for performance experiments so optimizations can be A/B tested
    over real 1000-step runs without editing code."""
    compile_mode: str = "step"   # none | wrap | step
    amp: str = "bf16"            # none | bf16 | fp16
    tf32: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    channels_last: bool = False
    log_every: int = 100         # perf reporting cadence


def normalize(x):
    return 2.0 * (x - COORD_MIN) / (COORD_MAX - COORD_MIN) - 1.0


def unnormalize(x):
    return (x + 1.0) / 2.0 * (COORD_MAX - COORD_MIN) + COORD_MIN


# =============================================================================
# 1. Environment
# =============================================================================
class PushTAdapter:
    """Translates between the PushT env and the policy.

    Keeps all PushT/coordinate specifics (obs layout, normalization, tensor
    plumbing) in one place so the eval loop can stay env-agnostic: it only
    sees model-ready tensors going in and env-ready actions coming out.
    """

    def __init__(self, device):
        self.device = device

    def observe(self, obs):
        # env obs dict -> batched, normalized (images, states) on device
        image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float() / 255.0
        state = normalize(torch.from_numpy(obs["agent_pos"]).float())
        return image.unsqueeze(0).to(self.device), state.unsqueeze(0).to(self.device)

    def act(self, actions):
        # policy output in [-1, 1] -> env actions in [0, 512]
        return unnormalize(actions).squeeze(0).cpu().numpy()


def evaluate(policy, device, n_episodes, render, n_steps):
    render_mode = "human" if render else "rgb_array"
    t0 = time.perf_counter()
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode=render_mode
    )
    adapter = PushTAdapter(device)
    policy.eval()

    successes = 0
    max_rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        max_reward = -float("inf")

        info = {}
        while not done:
            images_in, states_in = adapter.observe(obs)

            # Predict a full action horizon, then map it back to env actions.
            actions = policy.inference(images_in, states_in, n_steps=n_steps)
            actions = adapter.act(actions)

            # Receding horizon: execute one chunk, then replan from the new state.
            for action in actions[:ACTION_CHUNK_SIZE]:
                obs, reward, terminated, truncated, info = env.step(action)
                max_reward = max(max_reward, float(reward))
                if render:
                    time.sleep(0.05)
                done = terminated or truncated
                if done:
                    break

        if info.get("is_success", False):
            successes += 1
        max_rewards.append(max_reward)

    env.close()
    policy.train()

    t_total = time.perf_counter() - t0
    print(
        f"[eval] {n_episodes} eps, n_steps={n_steps}: {t_total:.1f}s "
        f"({t_total / n_episodes:.2f}s/ep)",
        flush=True,
    )
    return {
        "success_rate": successes / n_episodes,
        "avg_max_reward": sum(max_rewards) / n_episodes,
        "eval/time_total_s": t_total,
    }


# =============================================================================
# 2. Dataset
# =============================================================================
class PushTDataset(Dataset):
    def __init__(self, dataset_id, prediction_horizon):
        self.dataset = LeRobotDataset(dataset_id)
        self.prediction_horizon = prediction_horizon
        self.valid_indices = self._determine_valid_indices()

        # decode every frame once and keep it in RAM; afterwards each __getitem__
        print("Caching dataset in memory...")
        self.cache = {}
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            self.cache[idx] = {
                "observation.image": item["observation.image"],
                "observation.state": item["observation.state"],
                "action": item["action"],
            }
        print("Done caching.")

    def _determine_valid_indices(self):
        valid_indices = []
        episode_indices = np.array(self.dataset.hf_dataset["episode_index"])

        for idx in range(len(self.dataset)):
            episode = episode_indices[idx]

            # Need enough future actions for one predicted horizon.
            future_idx = idx + self.prediction_horizon - 1
            if future_idx >= len(self.dataset):
                continue

            # and those future frames must be in the same episode
            future_episode = episode_indices[future_idx].item()
            if future_episode == episode:
                valid_indices.append(idx)

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]

        obs = self.cache[start_idx]["observation.state"]
        actions = torch.stack(
            [
                self.cache[i]["action"]
                for i in range(start_idx, start_idx + self.prediction_horizon)
            ]
        )
        images = self.cache[start_idx]["observation.image"]

        # normalize positions and actions to [-1, 1]; images are already in [0, 1]
        obs = normalize(obs)
        actions = normalize(actions)

        return {"images": images, "obs": obs, "actions": actions}


# =============================================================================
# 3. Model
# =============================================================================
class ImageEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT
        resnet = models.resnet34(weights=weights)

        # Reuse the exact normalization the pretrained weights expect, sourced
        # from the weights metadata so it can't drift out of sync.
        preprocess = weights.transforms()
        self.register_buffer(
            "img_mean", torch.tensor(preprocess.mean).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor(preprocess.std).view(1, 3, 1, 1)
        )

        resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        resnet.maxpool = nn.Identity()

        feat_dim = resnet.fc.in_features  # 512 for resnet34
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Linear(feat_dim, out_dim)

    def forward(self, images):
        # images: (batch, 3, 96, 96)
        images = (images - self.img_mean) / self.img_std
        
        x = self.backbone(images)
        x = x.flatten(1) 
        return self.fc(x)


class StateEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(ROBOT_DOF, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, state):
        # state: (batch, ROBOT_DOF)
        return self.mlp(state)


# turns the scalar flow-time t into a vector "fingerprint" the network can use,
# rather than a single float which is hard for an MLP to interact with
class TimestepEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4), nn.GELU(), nn.Linear(out_dim * 4, out_dim)
        )

    def forward(self, t):
        # t: (batch,)
        half = self.out_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / half).to(t.device)
        x = t[:, None] * freqs[None, :]  # (batch, half)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (batch, out_dim)
        return self.mlp(x)


class AdaLNBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        # turns the conditioning vector into scale1, shift1, alpha1, scale2, shift2, alpha2
        self.adaLN_modulation = nn.Linear(hidden_dim, hidden_dim * 6)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, cond):
        # x: (batch, PREDICTION_HORIZON, hidden_dim), cond: (batch, hidden_dim)
        scale1, shift1, alpha1, scale2, shift2, alpha2 = self.adaLN_modulation(
            cond
        ).chunk(6, dim=-1)

        # attention sub-block with adaptive LayerNorm
        x1 = self.norm1(x)
        x1 = scale1.unsqueeze(1) * x1 + shift1.unsqueeze(1)
        x1, _ = self.attention(x1, x1, x1)
        x = x + alpha1.unsqueeze(1) * x1

        # MLP sub-block with adaptive LayerNorm
        x2 = self.norm2(x)
        x2 = scale2.unsqueeze(1) * x2 + shift2.unsqueeze(1)
        x = x + alpha2.unsqueeze(1) * self.mlp(x2)
        return x


class DiTPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        prediction_horizon=PREDICTION_HORIZON,
    ):
        super().__init__()
        # conditioning = image + state + timestep, all hidden_dim wide
        self.image_encoder = ImageEncoder(out_dim=hidden_dim)
        self.timestep_encoder = TimestepEncoder(out_dim=hidden_dim)
        self.state_encoder = StateEncoder(out_dim=hidden_dim)

        self.action_proj = nn.Linear(ROBOT_DOF, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, ROBOT_DOF)

        self.action_pos_emb = nn.Parameter(
            torch.zeros(1, prediction_horizon, hidden_dim)
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(hidden_dim=hidden_dim, n_heads=n_heads)
                for _ in range(n_layers)
            ]
        )

    def encode_context(self, images, obs):
        # context depends only on the observation, NOT on the noisy action or the
        # flow time t, so it's constant across a denoising loop and can be cached.
        # This is the expensive part (ResNet34 backbone).
        images_cond = self.image_encoder(images)
        obs_cond = self.state_encoder(obs)
        return images_cond, obs_cond

    def forward_with_context(self, x_t, t, context):
        # per-step network given a precomputed observation context.
        images_cond, obs_cond = context
        timestep_cond = self.timestep_encoder(t)
        cond = torch.cat([images_cond, obs_cond, timestep_cond], dim=-1)
        cond = self.cond_proj(cond)

        x = self.action_proj(x_t)
        x = x + self.action_pos_emb

        for block in self.blocks:
            x = block(x, cond)
        return self.action_out(x)

    def forward(self, x_t, t, images, obs):
        # pure network: noisy actions + time + context -> predicted velocity
        # x_t: (batch, PREDICTION_HORIZON, ROBOT_DOF), t: (batch,)
        # images: (batch, 3, 96, 96), obs: (batch, ROBOT_DOF)
        context = self.encode_context(images, obs)
        return self.forward_with_context(x_t, t, context)

    def training_step(self, images, obs, actions):
        t = torch.rand(actions.shape[0]).to(actions.device)
        noise = torch.randn_like(actions)
        x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * actions

        v_pred = self.forward(x_t, t, images, obs)
        target = actions - noise  
        return torch.nn.functional.mse_loss(v_pred, target)

    @torch.no_grad()
    def inference(self, images, obs, n_steps=N_DENOISING_STEPS):
        batch_size = images.shape[0]
        x = torch.randn(batch_size, PREDICTION_HORIZON, ROBOT_DOF).to(images.device)

        dt = 1.0 / n_steps
        # encode the observation context ONCE and reuse it for every denoising
        # step (the backbone depends only on images/obs, not on x or t).
        context = self.encode_context(images, obs)
        for i in range(n_steps):
            t = torch.full((batch_size,), i / n_steps).to(images.device)
            v = self.forward_with_context(x, t, context)
            x = x + v * dt
        return x.clamp(-1.0, 1.0)


# =============================================================================
# 4. Training
# =============================================================================
def save_checkpoint(policy, optimizer, global_step, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")

    # torch.compile wraps the model and stores the real model in _orig_mod.
    # Save the real model so checkpoint keys match a normal, uncompiled model.
    model = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


def _sync(device):
    # make CUDA kernels finish so wall-clock timings reflect real GPU work
    if device.type == "cuda":
        torch.cuda.synchronize()


def train(
    policy,
    dataloader,
    total_steps=TOTAL_STEPS,
    lr=LR,
    log_every=LOG_EVERY,
    eval_every=EVAL_EVERY,
    save_every=SAVE_EVERY,
    checkpoint_dir=CHECKPOINT_DIR,
    perf=None,
):
    perf = perf or PerfConfig()

    def run_eval():
        return evaluate(
            policy, device, n_episodes=EVAL_EPISODES, render=False,
            n_steps=N_DENOISING_STEPS,
        )
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training on {device}")
    print(f"Perf config: {perf}")

    # ---- optional global backend toggles (matmul precision / TF32) ----
    if perf.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    policy = policy.to(device)
    if perf.channels_last and device.type == "cuda":
        policy = policy.to(memory_format=torch.channels_last)

    # ---- compile strategy ----
    # "wrap":  torch.compile(policy) — note training calls policy.training_step,
    #          which routes through the *uncompiled* forward, so this is ~a no-op.
    # "step":  compile the training_step callable so the fwd/bwd graph is actually
    #          compiled (the real fix).
    # "none":  eager.
    train_step_fn = policy.training_step
    if perf.compile_mode == "wrap":
        policy = torch.compile(policy)
        train_step_fn = policy.training_step
    elif perf.compile_mode == "step":
        train_step_fn = torch.compile(policy.training_step)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- AMP setup ----
    if perf.amp == "bf16" and device.type == "cuda":
        autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        scaler = None
    elif perf.amp == "fp16" and device.type == "cuda":
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16)
        scaler = torch.amp.GradScaler("cuda")
    else:
        autocast_ctx = nullcontext()
        scaler = None

    last_loss = 0.0
    global_step = 0
    dataloader_iter = iter(dataloader)

    # ---- perf accumulators (reset every perf.log_every window) ----
    win_data = 0.0      # time spent waiting on next(dataloader)
    win_h2d = 0.0       # host->device transfer
    win_compute = 0.0   # fwd + bwd + optimizer (GPU work)
    win_wall0 = time.perf_counter()
    train_wall0 = time.perf_counter()
    first_step_done = False

    for global_step in range(1, total_steps + 1):
        t0 = time.perf_counter()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        t_data = time.perf_counter()

        images = batch["images"].to(device, non_blocking=perf.pin_memory)
        obs = batch["obs"].to(device, non_blocking=perf.pin_memory)
        actions = batch["actions"].to(device, non_blocking=perf.pin_memory)
        if perf.channels_last and device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        _sync(device)
        t_h2d = time.perf_counter()

        with autocast_ctx:
            loss = train_step_fn(images, obs, actions)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        _sync(device)
        t_compute = time.perf_counter()

        last_loss = loss.item()

        win_data += t_data - t0
        win_h2d += t_h2d - t_data
        win_compute += t_compute - t_h2d

        # cold start vs steady state: the first step pays compile + cudnn
        # autotune + lazy CUDA init, so report it separately.
        if not first_step_done:
            first_step_done = True
            cold_start_s = t_compute - train_wall0
            first_step_s = t_compute - t0
            print(
                f"[perf] first step done: cold_start={cold_start_s:.2f}s "
                f"(setup+compile+step1), step1_wall={first_step_s:.2f}s, "
                f"data={t_data - t0:.3f}s h2d={t_h2d - t_data:.3f}s "
                f"compute={t_compute - t_h2d:.3f}s",
                flush=True,
            )
            wandb.log(
                {
                    "perf/cold_start_s": cold_start_s,
                    "perf/first_step_s": first_step_s,
                },
                step=global_step,
            )
            # don't let the cold first step pollute steady-state averages
            win_data = win_h2d = win_compute = 0.0
            win_wall0 = time.perf_counter()

        # collect everything due on this step into one log call so wandb sees
        # a single record per step (log and eval cadences can coincide)
        log_dict = {}
        if global_step % log_every == 0:
            log_dict["loss"] = last_loss
            print(f"step {global_step}: loss={last_loss:.4f}")

        # perf window report (skips the cold first step)
        if global_step % perf.log_every == 0 and global_step > 1:
            win_wall = time.perf_counter() - win_wall0
            n = perf.log_every if global_step != perf.log_every else perf.log_every - 1
            steps_per_s = n / win_wall if win_wall > 0 else 0.0
            data_ms = 1000 * win_data / n
            h2d_ms = 1000 * win_h2d / n
            compute_ms = 1000 * win_compute / n
            log_dict.update({
                "perf/steps_per_s": steps_per_s,
                "perf/ms_per_step": 1000 * win_wall / n,
                "perf/data_ms": data_ms,
                "perf/h2d_ms": h2d_ms,
                "perf/compute_ms": compute_ms,
            })
            print(
                f"[perf] step {global_step}: {steps_per_s:.1f} it/s "
                f"({1000 * win_wall / n:.1f} ms/step) | "
                f"data={data_ms:.1f} h2d={h2d_ms:.1f} compute={compute_ms:.1f} ms",
                flush=True,
            )
            win_data = win_h2d = win_compute = 0.0
            win_wall0 = time.perf_counter()

        if eval_every and global_step % eval_every == 0:
            log_dict.update(run_eval())

        if log_dict:
            wandb.log(log_dict, step=global_step)

        if save_every and global_step % save_every == 0:
            save_checkpoint(
                policy, optimizer, global_step, last_loss, checkpoint_dir
            )

    total_wall = time.perf_counter() - train_wall0
    print(
        f"[perf] finished {total_steps} steps in {total_wall:.1f}s "
        f"(avg {total_steps / total_wall:.1f} it/s incl. cold start)",
        flush=True,
    )

    # final eval + checkpoint, unless the last step already triggered them
    final_dict = {}
    if eval_every and total_steps % eval_every != 0:
        final_dict.update(run_eval())
    if final_dict:
        wandb.log(final_dict, step=global_step)
    if save_every and total_steps % save_every != 0:
        save_checkpoint(policy, optimizer, global_step, last_loss, checkpoint_dir)

def parse_args():
    p = argparse.ArgumentParser(description="Train PushT DiT policy (+ perf knobs).")
    p.add_argument("--steps", type=int, default=TOTAL_STEPS,
                   help="total training steps (use e.g. 1000 for perf runs)")
    p.add_argument("--compile-mode", default="step",
                   choices=["none", "wrap", "step"],
                   help="torch.compile strategy ('wrap' replicates the old no-op)")
    p.add_argument("--amp", default="bf16", choices=["none", "bf16", "fp16"])
    p.add_argument("--tf32", dest="tf32", action="store_true", default=True)
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.add_argument("--channels-last", action="store_true", default=False)
    p.add_argument("--perf-log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    p.add_argument("--save-every", type=int, default=SAVE_EVERY)
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-mode", default="online",
                   choices=["online", "offline", "disabled"])
    return p.parse_args()


def main():
    args = parse_args()
    perf = PerfConfig(
        compile_mode=args.compile_mode,
        amp=args.amp,
        tf32=args.tf32,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
        channels_last=args.channels_last,
        log_every=args.perf_log_every,
    )

    run_name = args.run_name or f"pusht-dit-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"

    policy = DiTPolicy()
    dataset = PushTDataset(
        dataset_id=DATASET_ID,
        prediction_horizon=PREDICTION_HORIZON,
    )
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=perf.num_workers,
        pin_memory=perf.pin_memory,
    )
    if perf.num_workers > 0:
        loader_kwargs["persistent_workers"] = perf.persistent_workers
        loader_kwargs["prefetch_factor"] = perf.prefetch_factor
    dataloader = DataLoader(dataset, **loader_kwargs)

    steps_per_epoch = len(dataloader)
    print(
        f"steps/epoch={steps_per_epoch}, training for {args.steps} total steps "
        f"(~{math.ceil(args.steps / steps_per_epoch)} passes over the data)"
    )

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        mode=args.wandb_mode,
        config={
            "total_steps": args.steps,
            "steps_per_epoch": steps_per_epoch,
            "log_every": LOG_EVERY,
            "eval_every": args.eval_every,
            "eval_episodes": EVAL_EPISODES,
            "save_every": args.save_every,
            "lr": LR,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "hidden_dim": HIDDEN_DIM,
            "backbone": "resnet34",
            "robot_dof": ROBOT_DOF,
            "prediction_horizon": PREDICTION_HORIZON,
            "action_chunk_size": ACTION_CHUNK_SIZE,
            "n_denoising_steps": N_DENOISING_STEPS,
            "batch_size": BATCH_SIZE,
            "compile_mode": perf.compile_mode,
            "amp": perf.amp,
            "tf32": perf.tf32,
            "num_workers": perf.num_workers,
            "pin_memory": perf.pin_memory,
            "channels_last": perf.channels_last,
        },
    )

    train(
        policy=policy,
        dataloader=dataloader,
        total_steps=args.steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        perf=perf,
    )

    wandb.finish()


if __name__ == "__main__":
    main()