import copy
import math
import os
from datetime import datetime
import time

import wandb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import gymnasium as gym
import gym_pusht  

def _envi(name, default):
    return int(os.environ.get(name, default))


def _envs(name, default):
    return os.environ.get(name, default)


ROBOT_DOF = 2
PREDICTION_HORIZON = 16
ACTION_CHUNK_SIZE = 8
# Observation history fed to the policy. 1 = single frame (original). 2 gives the
# policy velocity/temporal context and matches lerobot's Diffusion Policy default;
# the extra frames are flattened into the global conditioning vector.
# Key experiment knobs are env-overridable so a driver can queue runs without
# editing code; each experiment still gets its own commit + report entry.
OBS_HORIZON = _envi("OBS_HORIZON", 2)
# Generative objective for the action head.
#   "flow"      -> flow matching (velocity target, Euler integration).
#   "diffusion" -> DDPM with a cosine (squaredcos_cap_v2) beta schedule and
#                  epsilon prediction, i.e. lerobot Diffusion Policy's objective.
OBJECTIVE = _envs("OBJECTIVE", "flow")
N_DENOISING_STEPS = 10          # flow-matching Euler steps at inference
DIFFUSION_TRAIN_STEPS = 100     # DDPM forward-process resolution
DIFFUSION_INFER_STEPS = _envi("DIFFUSION_INFER_STEPS", 100)  # DDPM reverse steps
DATASET_ID = "lerobot/pusht"
# DiT sized to ~47.5M (matching lerobot's DiT backbone): hidden 600, 6 layers, 8 heads
N_HEADS = _envi("N_HEADS", 8)
N_LAYERS = _envi("N_LAYERS", 6)
HIDDEN_DIM = _envi("HIDDEN_DIM", 600)
TOTAL_STEPS = _envi("TOTAL_STEPS", 50_000)
LR = 1e-4
# Gradient clipping a la nanoGPT. torch.nn.utils.clip_grad_norm_ returns the
# total pre-clip grad norm, which we log to watch for instability. Set to a
# float (nanoGPT uses 1.0) to also clip; None measures the norm without clipping.
GRAD_CLIP = None
WARMUP_STEPS = 500   # linear LR warmup
LR_MIN = 1e-6        # cosine decay floor
USE_EMA = True       # evaluate an EMA copy of the weights (lerobot-style)
LOG_EVERY = 200
EVAL_EVERY = _envi("EVAL_EVERY", 10_000)
EVAL_EPISODES = _envi("EVAL_EPISODES", 50)  # more episodes -> less noisy SR (was 20, +-0.11)
SAVE_EVERY = 25_000
RUN_NAME = os.environ.get("RUN_NAME") or f"claude-baseline-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
CHECKPOINT_DIR = f"checkpoints_{RUN_NAME}"
BATCH_SIZE = 512
# Background dataloader workers. With num_workers=0 the loader runs in the train
# process and each step blocks on CPU collation, which (with the GPU step at
# ~27ms) makes training data-loading bound. A few workers prefetch batches so
# the GPU step becomes the bottleneck.
NUM_WORKERS = 8
WANDB_PROJECT = "pushT-slim"
WANDB_ENTITY = "robot_learning_collective"

# PushT positions and actions are pixel coordinates in [0, 512]
# We normalize everything the model sees to [-1, 1]
# so it lives at the same scale as the Gaussian noise used by flow matching,
# then unnormalize the model's output before sending it back to the env.
COORD_MIN, COORD_MAX = 0.0, 512.0
# PushT frames are 96x96. lerobot-style: crop at native resolution (no upsample)
# and let SpatialSoftmax pool the small feature map (see ImageEncoder).
NATIVE_IMG_SIZE = 96
CROP_SIZE = 84  # random crop (train) / center crop (eval), ~0.875 of native
NUM_KEYPOINTS = 32  # SpatialSoftmax keypoints
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

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

    def __init__(self, device, obs_horizon=OBS_HORIZON):
        self.device = device
        self.obs_horizon = obs_horizon
        self._img_hist = None
        self._state_hist = None

    def _parse(self, obs):
        image = torch.from_numpy(obs["pixels"]).permute(2, 0, 1).float() / 255.0
        state = normalize(torch.from_numpy(obs["agent_pos"]).float())
        return image, state

    def reset(self, obs):
        # Prime the history with the first frame repeated, so the very first
        # decision sees a full obs_horizon stack (no velocity yet, as expected).
        image, state = self._parse(obs)
        self._img_hist = [image] * self.obs_horizon
        self._state_hist = [state] * self.obs_horizon

    def push(self, obs):
        # Call once per ENV STEP so the history holds CONSECUTIVE frames, matching
        # the dataset's consecutive-frame stacking (idx-1, idx) used in training.
        # (If pushed only per replan, the stacked frames would be ACTION_CHUNK_SIZE
        # env-steps apart and the policy would read a bogus, oversized velocity.)
        image, state = self._parse(obs)
        self._img_hist = (self._img_hist + [image])[-self.obs_horizon :]
        self._state_hist = (self._state_hist + [state])[-self.obs_horizon :]

    def get(self):
        # Current obs_horizon-frame stack -> batched, normalized, on device.
        # Feed the raw native-resolution frame; the encoder owns cropping (center
        # crop in eval, random crop in train) so train and eval share one path.
        images = torch.stack(self._img_hist, dim=0)  # (T, 3, 96, 96)
        states = torch.stack(self._state_hist, dim=0)  # (T, 2)
        return images.unsqueeze(0).to(self.device), states.unsqueeze(0).to(self.device)

    def act(self, actions):
        # policy output in [-1, 1] -> env actions in [0, 512]
        return unnormalize(actions).squeeze(0).cpu().numpy()


def evaluate(policy, device, n_episodes, render, n_steps):
    render_mode = "human" if render else "rgb_array"
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode=render_mode
    )
    adapter = PushTAdapter(device)
    policy.eval()

    successes = 0
    max_rewards = []
    for episode in range(n_episodes):
        obs, _ = env.reset()
        adapter.reset(obs)
        done = False
        max_reward = -float("inf")

        info = {}
        while not done:
            images_in, states_in = adapter.get()

            # Predict a full action horizon, then map it back to env actions.
            actions = policy.inference(images_in, states_in, n_steps=n_steps)
            actions = adapter.act(actions)

            # Receding horizon: execute one chunk, then replan from the new state.
            for action in actions[:ACTION_CHUNK_SIZE]:
                obs, reward, terminated, truncated, info = env.step(action)
                adapter.push(obs)  # keep obs history at env-step cadence
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
    metrics = {
        "success_rate": successes / n_episodes,
        "avg_max_reward": sum(max_rewards) / n_episodes,
    }
    print(
        f"[eval] success_rate={metrics['success_rate']:.3f} "
        f"avg_max_reward={metrics['avg_max_reward']:.3f} (n_episodes={n_episodes}, "
        f"n_steps={n_steps})"
    )
    return metrics


# =============================================================================
# 2. Dataset
# =============================================================================
class PushTDataset(Dataset):
    def __init__(self, dataset_id, prediction_horizon, obs_horizon=OBS_HORIZON):
        self.dataset = LeRobotDataset(dataset_id)
        self.prediction_horizon = prediction_horizon
        self.obs_horizon = obs_horizon

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
        self.episode_indices = np.array(self.dataset.hf_dataset["episode_index"])
        print("Done caching.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        current_episode = self.episode_indices[idx]

        # Return the last obs_horizon raw native-resolution frames ([0, 1],
        # (H, 3, 96, 96)) stacked oldest->newest, clamped at the episode start
        # (repeat the current frame). Cropping and normalization happen inside
        # the encoder so train/eval stay in sync.
        imgs, states = [], []
        for offset in range(-(self.obs_horizon - 1), 1):
            i = idx + offset
            if i < 0 or self.episode_indices[i] != current_episode:
                i = idx  # before episode start -> repeat current frame
            imgs.append(self.cache[i]["observation.image"])
            states.append(normalize(self.cache[i]["observation.state"]))
        img = torch.stack(imgs, dim=0)  # (H, 3, 96, 96)
        obs = torch.stack(states, dim=0)  # (H, 2)

        actions = torch.zeros(self.prediction_horizon, 2)
        mask = torch.zeros(self.prediction_horizon, 2)
        
        for k in range(self.prediction_horizon):
            i = idx + k
            if i < len(self.dataset) and self.episode_indices[i] == current_episode:
                actions[k] = normalize(self.cache[i]["action"])
                mask[k] = 1.0

        return img, obs, actions.flatten(), mask.flatten()

# =============================================================================
# 3. Model
# =============================================================================
class SpatialSoftmax(nn.Module):
    """Spatial soft-argmax pooling (Finn et al. 2015), ported from lerobot/robomimic.

    Turns a (B, C, H, W) feature map into (B, K, 2) keypoint coordinates: the
    softmax-weighted "center of mass" of each channel's activations. Unlike global
    average pooling, this preserves *where* features fire, so it stays informative
    even on the tiny feature maps a stock ResNet produces from small inputs (an
    84x84 crop -> 512x3x3). A 1x1 conv first remaps C -> num_kp keypoint channels.
    """

    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features):
        if self.nets is not None:
            features = self.nets(features)
        batch = features.shape[0]
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        keypoints = attention @ self.pos_grid  # (B * out_c, 2)
        return keypoints.reshape(batch, self._out_c, 2)


class ImageEncoder(nn.Module):
    """lerobot/Diffusion-Policy-style vision encoder.

    Solves the "small image + ResNet built for 224" problem the way lerobot does:
    keep the stock ResNet stem, feed a native-resolution crop (no upsampling), and
    let SpatialSoftmax pool the resulting tiny feature map into keypoints — instead
    of preserving resolution via a modified stem or upsampling the input to 224.
    Cropping lives here (random in train, center in eval) so train and eval share
    a single code path.
    """

    def __init__(self, out_dim, crop_size=CROP_SIZE, num_kp=NUM_KEYPOINTS):
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

        # Same crop window for the whole batch (matching lerobot's RandomCrop).
        # We index with tensors instead of torchvision's RandomCrop because the
        # latter calls torch.randint(...).item(), and that .item() forces a
        # CUDA sync + a torch.compile graph break at the top of the model.
        self.crop_size = crop_size
        self.center_crop = T.CenterCrop(crop_size)

        # Drop avgpool + fc; keep the stock stem and conv stages -> feature map.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Dry run to get the feature-map shape that SpatialSoftmax must index.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, crop_size, crop_size)
            feat_shape = self.backbone(dummy).shape[1:]  # (C, H, W)
        self.pool = SpatialSoftmax(feat_shape, num_kp=num_kp)
        self.fc = nn.Linear(num_kp * 2, out_dim)

    def random_crop(self, images):
        # One random crop window for the whole batch, expressed entirely in
        # tensor ops (no .item()) so it stays inside the compiled graph.
        _, _, h, w = images.shape
        size = self.crop_size
        top = torch.randint(0, h - size + 1, (1,), device=images.device)
        left = torch.randint(0, w - size + 1, (1,), device=images.device)
        rows = top + torch.arange(size, device=images.device)
        cols = left + torch.arange(size, device=images.device)
        return images[:, :, rows[:, None], cols[None, :]]

    def forward(self, images):
        # images: (batch, 3, 96, 96) in [0, 1]
        images = self.random_crop(images) if self.training else self.center_crop(images)
        images = (images - self.img_mean) / self.img_std

        x = self.backbone(images)
        x = self.pool(x).flatten(1)  # (batch, num_kp * 2)
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


def cosine_beta_schedule(num_steps, s=0.008):
    """squaredcos_cap_v2 schedule (Nichol & Dhariwal), as used by lerobot's DDPM."""
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999).float()


class DiTPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim=HIDDEN_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        prediction_horizon=PREDICTION_HORIZON,
        obs_horizon=OBS_HORIZON,
        objective=OBJECTIVE,
        diffusion_train_steps=DIFFUSION_TRAIN_STEPS,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.objective = objective
        self.diffusion_train_steps = diffusion_train_steps
        # conditioning = obs_horizon*(image + state) + timestep, all hidden_dim
        # wide. The per-frame image/state features are flattened across time and
        # concatenated with the timestep embedding into one global cond vector.
        self.image_encoder = ImageEncoder(out_dim=hidden_dim)
        self.timestep_encoder = TimestepEncoder(out_dim=hidden_dim)
        self.state_encoder = StateEncoder(out_dim=hidden_dim)

        self.action_proj = nn.Linear(ROBOT_DOF, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, ROBOT_DOF)

        self.action_pos_emb = nn.Parameter(
            torch.zeros(1, prediction_horizon, hidden_dim)
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_dim * (2 * obs_horizon + 1), hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [
                AdaLNBlock(hidden_dim=hidden_dim, n_heads=n_heads)
                for _ in range(n_layers)
            ]
        )

        # DDPM schedule buffers (only used when objective == "diffusion"). Kept as
        # buffers so they follow .to(device) and get copied into the EMA clone.
        if objective == "diffusion":
            betas = cosine_beta_schedule(diffusion_train_steps)
            alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
            self.register_buffer("betas", betas)
            self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(self, x_t, t, images, obs):
        # pure network: noisy actions + time + context -> predicted velocity
        # x_t: (batch, PREDICTION_HORIZON, ROBOT_DOF), t: (batch,)
        # images: (batch, obs_horizon, 3, 96, 96), obs: (batch, obs_horizon, DOF)
        B, H = images.shape[0], images.shape[1]
        images_cond = self.image_encoder(images.reshape(B * H, *images.shape[2:]))
        images_cond = images_cond.reshape(B, -1)  # (B, H * hidden)
        obs_cond = self.state_encoder(obs.reshape(B * H, obs.shape[-1]))
        obs_cond = obs_cond.reshape(B, -1)  # (B, H * hidden)
        timestep_cond = self.timestep_encoder(t)
        cond = torch.cat([images_cond, obs_cond, timestep_cond], dim=-1)
        cond = self.cond_proj(cond)

        x = self.action_proj(x_t)
        x = x + self.action_pos_emb
        
        for block in self.blocks:
            x = block(x, cond)
        return self.action_out(x)

    @torch.no_grad()
    def inference(self, images, obs, n_steps=N_DENOISING_STEPS):
        if self.objective == "diffusion":
            return self._ddpm_sample(images, obs)

        batch_size = images.shape[0]
        x = torch.randn(batch_size, PREDICTION_HORIZON, ROBOT_DOF).to(images.device)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((batch_size,), i / n_steps).to(images.device)
            v = self.forward(x, t, images, obs)
            x = x + v * dt
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def _ddpm_sample(self, images, obs):
        # Ancestral DDPM sampling with clip_sample (matches lerobot). Runs the
        # full forward-process resolution in reverse (diffusion_train_steps).
        device = images.device
        batch_size = images.shape[0]
        x = torch.randn(batch_size, PREDICTION_HORIZON, ROBOT_DOF, device=device)
        betas, acp = self.betas, self.alphas_cumprod
        alphas = 1.0 - betas
        for t in range(self.diffusion_train_steps - 1, -1, -1):
            t_batch = torch.full((batch_size,), float(t), device=device)
            eps = self.forward(x, t_batch, images, obs)
            acp_t = acp[t]
            acp_prev = acp[t - 1] if t > 0 else torch.ones_like(acp_t)
            # predict x0 from eps, clip to the normalized action range
            x0 = (x - (1.0 - acp_t).sqrt() * eps) / acp_t.sqrt()
            x0 = x0.clamp(-1.0, 1.0)
            coef_x0 = (acp_prev.sqrt() * betas[t]) / (1.0 - acp_t)
            coef_xt = (alphas[t].sqrt() * (1.0 - acp_prev)) / (1.0 - acp_t)
            mean = coef_x0 * x0 + coef_xt * x
            if t > 0:
                var = betas[t] * (1.0 - acp_prev) / (1.0 - acp_t)
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1.0, 1.0)


# =============================================================================
# 4. Training
# =============================================================================
def save_checkpoint(policy, ema_policy, optimizer, scheduler, global_step, loss, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")

    # torch.compile wraps the model and stores the real model in _orig_mod.
    # Save the real model so checkpoint keys match a normal, uncompiled model.
    model = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
    torch.save(
        {
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "ema_model_state_dict": ema_policy.state_dict() if ema_policy is not None else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


# Kept as a free function rather than a DiTPolicy method on purpose: we compile
# the module (torch.compile(policy)) and call policy(...) directly, so the loss
# must live outside forward. A method would route through the uncompiled
# self.forward, making the compile a no-op.
def flow_matching_loss(policy, images, obs, actions, masks):
    B = actions.shape[0]
    actions = actions.reshape(B, PREDICTION_HORIZON, ROBOT_DOF)
    masks = masks.reshape(B, PREDICTION_HORIZON, ROBOT_DOF)

    t = torch.rand(B, device=actions.device)
    noise = torch.randn_like(actions)
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
    v_pred = policy(x_t, t, images, obs)
    loss = nn.functional.mse_loss(v_pred, actions - noise, reduction="none")
    # Masked mean over valid (in-episode) timesteps. Averaging — rather than
    # summing over the horizon — keeps the loss on the same scale as an unmasked
    # mean, so it's comparable across runs and doesn't inflate the effective
    # learning rate by ~PREDICTION_HORIZON.
    return (loss * masks).sum() / masks.sum().clamp(min=1)


def diffusion_loss(policy, images, obs, actions, masks, alphas_cumprod, num_train_steps):
    # DDPM epsilon-prediction loss. Same masked-mean reduction as the flow loss.
    B = actions.shape[0]
    actions = actions.reshape(B, PREDICTION_HORIZON, ROBOT_DOF)
    masks = masks.reshape(B, PREDICTION_HORIZON, ROBOT_DOF)

    t_int = torch.randint(0, num_train_steps, (B,), device=actions.device)
    noise = torch.randn_like(actions)
    acp = alphas_cumprod[t_int][:, None, None]
    x_t = acp.sqrt() * actions + (1.0 - acp).sqrt() * noise
    eps_pred = policy(x_t, t_int.float(), images, obs)
    loss = nn.functional.mse_loss(eps_pred, noise, reduction="none")
    return (loss * masks).sum() / masks.sum().clamp(min=1)


def train(
    policy,
    dataloader,
    total_steps=TOTAL_STEPS,
    lr=LR,
    log_every=LOG_EVERY,
    eval_every=EVAL_EVERY,
    save_every=SAVE_EVERY,
    checkpoint_dir=CHECKPOINT_DIR,
):
    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    policy = policy.to(device)

    # EMA copy evaluated in place of the live weights (lerobot-style). Kept
    # uncompiled and updated by hand each step from the live (compiled) params.
    ema_policy = copy.deepcopy(policy) if USE_EMA else None
    if ema_policy is not None:
        ema_policy.eval()

    policy = torch.compile(policy)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN / lr + (1.0 - LR_MIN / lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # The model we hand to evaluate(): EMA copy when enabled, else the live one.
    eval_policy = ema_policy if ema_policy is not None else policy

    # Objective-specific bits: DDPM needs the alphas_cumprod buffer (kept on the
    # uncompiled module) for the forward noising; eval step count differs too.
    orig = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
    alphas_cumprod = orig.alphas_cumprod if OBJECTIVE == "diffusion" else None
    eval_n_steps = DIFFUSION_INFER_STEPS if OBJECTIVE == "diffusion" else N_DENOISING_STEPS

    last_loss = 0.0
    global_step = 0
    dataloader_iter = iter(dataloader)

    # Rolling per-step timers (reset each log window) so we can see whether we're
    # data-loading bound or compute bound. data_time covers fetching/collating
    # the batch and the host->device copy; compute_time covers the forward,
    # backward and optimizer step (loss.item() below forces a CUDA sync, so the
    # measured compute time includes the GPU work rather than just the launch).
    data_time = 0.0
    compute_time = 0.0

    for global_step in range(1, total_steps + 1):
        t_data_start = time.perf_counter()
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        images, obs, actions, mask = [
            t.to(device, non_blocking=True) for t in batch
        ]
        t_compute_start = time.perf_counter()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            if OBJECTIVE == "diffusion":
                loss = diffusion_loss(
                    policy, images, obs, actions, mask,
                    alphas_cumprod, DIFFUSION_TRAIN_STEPS,
                )
            else:
                loss = flow_matching_loss(policy, images, obs, actions, mask)
        optimizer.zero_grad()
        loss.backward()
        # nanoGPT-style: clip_grad_norm_ returns the total grad norm pre-clip.
        # max_norm=inf measures without actually clipping (GRAD_CLIP=None).
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=GRAD_CLIP if GRAD_CLIP is not None else float("inf"),
        )
        optimizer.step()
        scheduler.step()

        # EMA: warm-ramp decay (fast early, asymptotes to 0.9999 ~31k steps)
        if ema_policy is not None:
            ema_decay = min(1 - (1 + global_step) ** (-0.75), 0.9999)
            src = policy._orig_mod if hasattr(policy, "_orig_mod") else policy
            with torch.no_grad():
                for ema_p, p in zip(ema_policy.parameters(), src.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
                # copy buffers (e.g. BN running stats) so the EMA model sees
                # up-to-date normalization rather than its frozen init values.
                for ema_b, b in zip(ema_policy.buffers(), src.buffers()):
                    ema_b.data.copy_(b.data)

        last_loss = loss.item()
        t_step_end = time.perf_counter()

        data_time += t_compute_start - t_data_start
        compute_time += t_step_end - t_compute_start

        # collect everything due on this step into one log call so wandb sees
        # a single record per step (log and eval cadences can coincide)
        log_dict = {}
        if global_step % log_every == 0:
            data_ms = data_time / log_every * 1e3
            compute_ms = compute_time / log_every * 1e3
            step_ms = data_ms + compute_ms
            it_per_s = 1e3 / step_ms if step_ms > 0 else 0.0
            grad_norm_val = grad_norm.item()
            log_dict["loss"] = last_loss
            log_dict["lr"] = scheduler.get_last_lr()[0]
            log_dict["grad_norm"] = grad_norm_val
            log_dict["time/data_ms"] = data_ms
            log_dict["time/compute_ms"] = compute_ms
            log_dict["time/step_ms"] = step_ms
            log_dict["time/it_per_s"] = it_per_s
            print(
                f"step {global_step}: loss={last_loss:.4f} grad_norm={grad_norm_val:.3f} | "
                f"data={data_ms:.1f}ms compute={compute_ms:.1f}ms "
                f"step={step_ms:.1f}ms ({it_per_s:.1f} it/s)"
            )
            data_time = 0.0
            compute_time = 0.0

        if global_step % eval_every == 0:
            log_dict.update(evaluate(
                eval_policy, device, n_episodes=EVAL_EPISODES, render=False,
                n_steps=eval_n_steps,
            ))

        if log_dict:
            wandb.log(log_dict, step=global_step)

        if global_step % save_every == 0:
            save_checkpoint(
                policy, ema_policy, optimizer, scheduler, global_step, last_loss, checkpoint_dir
            )

    # final eval + checkpoint, unless the last step already triggered them
    final_dict = {}
    if total_steps % eval_every != 0:
        final_dict.update(evaluate(
            eval_policy, device, n_episodes=EVAL_EPISODES, render=False,
            n_steps=eval_n_steps,
        ))
    if final_dict:
        wandb.log(final_dict, step=global_step)
    if total_steps % save_every != 0:
        save_checkpoint(policy, ema_policy, optimizer, scheduler, global_step, last_loss, checkpoint_dir)

def main():

    policy = DiTPolicy()
    dataset = PushTDataset(
        dataset_id=DATASET_ID,
        prediction_horizon=PREDICTION_HORIZON,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )

    steps_per_epoch = len(dataloader)
    print(
        f"steps/epoch={steps_per_epoch}, training for {TOTAL_STEPS} total steps "
        f"(~{math.ceil(TOTAL_STEPS / steps_per_epoch)} passes over the data)"
    )

    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=RUN_NAME,
        config={
            "total_steps": TOTAL_STEPS,
            "steps_per_epoch": steps_per_epoch,
            "log_every": LOG_EVERY,
            "eval_every": EVAL_EVERY,
            "eval_episodes": EVAL_EPISODES,
            "save_every": SAVE_EVERY,
            "lr": LR,
            "warmup_steps": WARMUP_STEPS,
            "lr_min": LR_MIN,
            "use_ema": USE_EMA,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "hidden_dim": HIDDEN_DIM,
            "backbone": "resnet34",
            "pooling": "spatial_softmax",
            "crop_size": CROP_SIZE,
            "num_keypoints": NUM_KEYPOINTS,
            "robot_dof": ROBOT_DOF,
            "obs_horizon": OBS_HORIZON,
            "objective": OBJECTIVE,
            "diffusion_train_steps": DIFFUSION_TRAIN_STEPS,
            "diffusion_infer_steps": DIFFUSION_INFER_STEPS,
            "prediction_horizon": PREDICTION_HORIZON,
            "action_chunk_size": ACTION_CHUNK_SIZE,
            "n_denoising_steps": N_DENOISING_STEPS,
            "batch_size": BATCH_SIZE,
        },
    )

    train(policy=policy, dataloader=dataloader)

    wandb.finish()


if __name__ == "__main__":
    main()