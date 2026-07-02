# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import math
import os
from datetime import datetime

import numpy as np
import wandb
import gymnasium as gym
import gym_pusht
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from transformers import AutoModel

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ── Config ─────────────────────────────────────────────────────────────────────
N_ACTIONS = 16
N_ACTION_STEPS = 8
HIDDEN_DIM = 512
BATCH_SIZE = 64
TOTAL_STEPS = 100_000
LR = 1e-4
LR_MIN = 1e-6
WARMUP_STEPS = 500
LOG_EVERY = 200
EVAL_EVERY = 5_000
SAVE_EVERY = 5_000
EVAL_EPISODES=50
RUN_NAME = f"dit-pusht-slim-DINO-{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
CHECKPOINT_DIR = f"checkpoints_dit-{RUN_NAME}"
DATASET_ID = "lerobot/pusht"
COORD_MIN, COORD_MAX = 0.0, 512.0
# DINOv3 ViT-S/16: patch_size=16, embed_dim=384.
# Crop must be divisible by 16; 80×80 → 5×5 = 25 patch tokens.
CROP_SIZE = 80
DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
DINOV3_EMBED_DIM = 384


def normalize(x):
    return 2.0 * (x - COORD_MIN) / (COORD_MAX - COORD_MIN) - 1.0


def unnormalize(x):
    return (x + 1.0) / 2.0 * (COORD_MAX - COORD_MIN) + COORD_MIN


# ── Dataset ────────────────────────────────────────────────────────────────────
class PushTDataset(Dataset):
    def __init__(self, dataset_id=DATASET_ID, n_actions=N_ACTIONS):
        self.dataset = LeRobotDataset(dataset_id)
        self.n_actions = n_actions

        # decode every frame once and keep it in RAM; afterwards each __getitem__
        print("Caching dataset in memory...")
        self.cache = {}
        for idx in tqdm(range(len(self.dataset))):
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

        img = self.cache[idx]["observation.image"]  # (3, 96, 96) in [0, 1]
        obs = normalize(self.cache[idx]["observation.state"])

        actions = torch.zeros(self.n_actions, 2)
        mask = torch.zeros(self.n_actions, 2)
        
        for k in range(self.n_actions):
            i = idx + k
            if i < len(self.dataset) and self.episode_indices[i] == current_episode:
                actions[k] = normalize(self.cache[i]["action"])
                mask[k] = 1.0

        return img, obs, actions.flatten(), mask.flatten()


# ── Model ──────────────────────────────────────────────────────────────────────
class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 ViT-S/16 returning spatial patch tokens.

    Replaces ResNet(avg_pool=False) which gave 49 tokens from a 7×7 conv map.
    DINOv3 on an 80×80 crop gives 25 patch tokens (5×5) of shape
    (B, 25, HIDDEN_DIM) after a linear projection from the 384-dim ViT embedding.
    The backbone is frozen — the dataset is too small to fine-tune a ViT safely.
    """

    def __init__(self, out_dim, crop_size=CROP_SIZE):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(DINOV3_MODEL)
        self.backbone.requires_grad_(False)

        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        self.crop_size = crop_size
        self.center_crop = T.CenterCrop(crop_size)
        self._n_patches = (crop_size // 16) ** 2  # 25 for 80×80
        self.proj = nn.Linear(DINOV3_EMBED_DIM, out_dim)

    def random_crop(self, images):
        _, _, h, w = images.shape
        size = self.crop_size
        top = torch.randint(0, h - size + 1, (1,), device=images.device)
        left = torch.randint(0, w - size + 1, (1,), device=images.device)
        rows = top + torch.arange(size, device=images.device)
        cols = left + torch.arange(size, device=images.device)
        return images[:, :, rows[:, None], cols[None, :]]

    def forward(self, images):
        images = self.random_crop(images) if self.training else self.center_crop(images)
        images = (images - self.img_mean) / self.img_std
        # last_hidden_state: (B, 1 + n_patches, 384) — index 0 is CLS, rest are patches
        patch_tokens = self.backbone(pixel_values=images).last_hidden_state[:, 1:]
        return self.proj(patch_tokens)  # (B, n_patches, out_dim)

    @property
    def embed_dim(self):
        return self.proj.out_features

    @property
    def n_tokens(self):
        return self._n_patches

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tensor of shape (seq_len, batch_size, d_model) with positional encodings added
        """
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return self.out_net(x)


class _SelfAttnEncoder(nn.Module):
    def __init__(
        self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU(approximate="tanh")

    def forward(self, src, pos):
        q = k = src + pos
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None] + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.GELU(approximate="tanh")

        # create modulation layers
        self.attn_mod1 = _ShiftScaleMod(d_model)
        self.attn_mod2 = _ZeroScaleMod(d_model)
        self.mlp_mod1 = _ShiftScaleMod(d_model)
        self.mlp_mod2 = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, dim=0)
        cond = cond + t

        x2 = self.attn_mod1(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), cond) + x

        x2 = self.mlp_mod1(self.norm2(x), cond)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), cond)
        return x + x2

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = torch.mean(cond, dim=0)
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = x * scale[None] + shift[None]
        x = self.linear(x)
        return x.transpose(0, 1)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerEncoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )

        for l in self.layers:
            l.reset_parameters()

    def forward(self, src, pos):
        x, outputs = src, []
        for layer in self.layers:
            x = layer(x, pos)
            outputs.append(x)
        return outputs

class _TransformerDecoder(_TransformerEncoder):
    def forward(self, src, t, all_conds):
        x = src
        for layer, cond in zip(self.layers, all_conds):
            x = layer(x, t, cond)
        return x

class _DiTNoiseNet(nn.Module):
    def __init__(
        self,
        ac_dim,
        ac_chunk,
        time_dim=256,
        hidden_dim=512,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
    ):
        super().__init__()

        # positional encoding blocks
        self.enc_pos = _PositionalEncoding(hidden_dim)
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # input encoder mlps
        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )

        # encoder blocks
        encoder_module = _SelfAttnEncoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = _TransformerEncoder(encoder_module, num_blocks)

        # decoder blocks
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # turns predicted tokens into epsilons
        self.eps_out = _FinalLayer(hidden_dim, ac_dim)


    def forward(self, noise_actions, time, obs_enc, enc_cache=None):
        if enc_cache is None:
            enc_cache = self.forward_enc(obs_enc)
        return enc_cache, self.forward_dec(noise_actions, time, enc_cache)
    
    def forward_enc(self, obs_enc):
        obs_enc = obs_enc.transpose(0, 1)
        pos = self.enc_pos(obs_enc)
        enc_cache = self.encoder(obs_enc, pos)
        return enc_cache

    def forward_dec(self, noise_actions, time, enc_cache):
        time_enc = self.time_net(time)
        
        ac_tokens = self.ac_proj(noise_actions)
        ac_tokens = ac_tokens.transpose(0, 1)
        ## ?? 
        dec_in = ac_tokens + self.dec_pos

        # apply decoder
        dec_out = self.decoder(dec_in, time_enc, enc_cache)

        # apply final epsilon prediction layer
        return self.eps_out(dec_out, time_enc, enc_cache[-1])

class _BatchNorm1DHelper(nn.BatchNorm1d):
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.transpose(1, 2)
            x = super().forward(x)
            return x.transpose(1, 2)
        return super().forward(x)

    
class FlowMatchingTransformerAgent(nn.Module):
    def __init__(
        self,
        image_encoder,
        odim,
        ac_dim,
        ac_chunk,
        dropout=0,
        noise_net_kwargs=dict(),
    ):

        # initialize obs and img tokenizers
        super().__init__()

        self.image_encoder = image_encoder

        self._token_dim = image_encoder.embed_dim 
        self._n_tokens = image_encoder.n_tokens + 1 # added 1 for observation token

        # handle obs tokenization strategies
        self._obs_proc = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(odim, self._token_dim)
        )

        norm = _BatchNorm1DHelper(self._token_dim)

        self.post_proc = nn.Sequential(norm, nn.Dropout(dropout))

        self.noise_net = _DiTNoiseNet(
            ac_dim=ac_dim,
            ac_chunk=ac_chunk,
            **noise_net_kwargs,
        )
        self.ac_dim, self.ac_chunk = ac_dim, ac_chunk


    def tokenize_obs(self, imgs, obs):
        image_tokens = self.image_encoder(imgs)
        obs_token = self._obs_proc(obs)[:, None]
        tokens = torch.cat((image_tokens, obs_token), 1)
        return self.post_proc(tokens)

    def forward(self, imgs, obs, ac_flat, mask_flat):
        # get observation encoding and sample noise/timesteps
        B, device = obs.shape[0], obs.device
        s_t = self.tokenize_obs(imgs, obs)

        actions = ac_flat.reshape(B, self.ac_chunk, self.ac_dim)
        mask = mask_flat.reshape((B, self.ac_chunk, self.ac_dim))
        timestep = torch.rand(B, device=device)
        noise = torch.randn_like(actions)
        noise_acs = (1 - timestep[:, None, None]) * noise + timestep[:, None, None] * actions
        _, v_pred = self.noise_net(noise_acs, timestep, s_t)

        loss = nn.functional.mse_loss(v_pred, actions - noise, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def get_actions(self, imgs, obs, n_steps=10):
        B, device = obs.shape[0], obs.device
        s_t = self.tokenize_obs(imgs, obs)
        enc_cache = self.noise_net.forward_enc(s_t)
        x = torch.randn(B, self.ac_chunk, self.ac_dim, device=device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i / n_steps, device=device)
            _, v = self.noise_net(x, t, s_t, enc_cache)
            x = x + v * dt
        return x.clamp(-1.0, 1.0)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(agent, device, n_episodes=EVAL_EPISODES, n_steps=10):
    env = gym.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array"
    )
    agent.eval()
    successes = 0

    for _ in range(n_episodes):
        obs_env, _ = env.reset()
        done = False
        img_t = torch.from_numpy(obs_env["pixels"]).permute(2, 0, 1).float() / 255.0
        state_t = normalize(torch.from_numpy(obs_env["agent_pos"]).float())

        while not done:
            imgs = img_t.unsqueeze(0).to(device)  # encoder owns crop + normalize
            obs = state_t.unsqueeze(0).to(device)

            acs = agent.get_actions(imgs, obs, n_steps=n_steps)  # (1, 16, 2) normalized
            acs = unnormalize(acs.squeeze(0).cpu().numpy())  # (16, 2) in [0, 512]

            for ac in acs[:N_ACTION_STEPS]:
                obs_env, _, terminated, truncated, info = env.step(ac)
                done = terminated or truncated
                img_t = (
                    torch.from_numpy(obs_env["pixels"]).permute(2, 0, 1).float() / 255.0
                )
                state_t = normalize(torch.from_numpy(obs_env["agent_pos"]).float())
                if done:
                    break

        if info.get("is_success", False):
            successes += 1

    env.close()
    agent.train()
    return successes / n_episodes


# ── Training loop ─────────────────────────────────────────────────────────────
def train(agent, loader):
    device = torch.device("cuda")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    agent = agent.to(device)

    ema_agent = copy.deepcopy(agent)
    ema_agent.eval()

    agent = torch.compile(agent)

    src = agent._orig_mod if hasattr(agent, "_orig_mod") else agent
    trainable = [p for p in src.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return LR_MIN / LR + (1.0 - LR_MIN / LR) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    running_loss, running_n = 0.0, 0
    pbar = tqdm(total=TOTAL_STEPS, desc="train")
    dataloader_iter = iter(loader)

    for step in range(1, TOTAL_STEPS + 1):
        try:
            imgs, obs, ac_flat, mask_flat = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(loader)
            imgs, obs, ac_flat, mask_flat = next(dataloader_iter)

        imgs, obs, ac_flat, mask_flat = [
            t.to(device, non_blocking=True) for t in (imgs, obs, ac_flat, mask_flat)
        ]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = agent(imgs, obs, ac_flat, mask_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # EMA: warm-ramp decay, copy buffers (e.g. BatchNorm running stats) directly
        ema_decay = min(1 - (1 + step) ** (-0.75), 0.9999)
        with torch.no_grad():
            for ema_p, p in zip(ema_agent.parameters(), src.parameters()):
                ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
            for ema_buf, buf in zip(ema_agent.buffers(), src.buffers()):
                ema_buf.data.copy_(buf.data)

        running_loss += loss.item()
        running_n += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        log = {}
        if step % LOG_EVERY == 0:
            log["train/loss"] = running_loss / running_n
            log["lr"] = scheduler.get_last_lr()[0]
            running_loss, running_n = 0.0, 0

        if step % EVAL_EVERY == 0:
            sr = evaluate(ema_agent, device, n_episodes=EVAL_EPISODES, n_steps=10)
            ema_agent.eval()  # evaluate() calls agent.train() at end; undo that
            log["success_rate"] = sr
            print(f"step {step}: success={sr:.2f}")

        if log:
            wandb.log(log, step=step)

        if step % SAVE_EVERY == 0:
            model = agent._orig_mod if hasattr(agent, "_orig_mod") else agent
            ckpt = os.path.join(CHECKPOINT_DIR, f"step_{step}.pt")
            torch.save(
                {
                    "step": step,
                    "model": model.state_dict(),
                    "ema_model": ema_agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                ckpt,
            )
            print(f"Saved {ckpt}")

    pbar.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    wandb.init(
        project="pushT-slim",
        entity="robot_learning_collective",
        name=RUN_NAME,
        config=dict(
            total_steps=TOTAL_STEPS,
            lr=LR,
            hidden_dim=HIDDEN_DIM,
            n_actions=N_ACTIONS,
            batch_size=BATCH_SIZE,
        ),
    )
    image_encoder = DINOv3Encoder(out_dim=HIDDEN_DIM)

    agent = FlowMatchingTransformerAgent(
        image_encoder=image_encoder,
        odim=2,  
        ac_dim=2,
        ac_chunk=N_ACTIONS,
        dropout=0.1,
        noise_net_kwargs=dict(
            time_dim=256,
            hidden_dim=HIDDEN_DIM,
            num_blocks=6,
            dim_feedforward=HIDDEN_DIM * 4,
            nhead=8,
            dropout=0.1,
        ),
    )
    dataset = PushTDataset()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    print(
        f"Steps/epoch: {len(loader)},  ~{math.ceil(TOTAL_STEPS / len(loader))} epochs"
    )
    train(agent, loader)
    wandb.finish()

if __name__ == "__main__":
    main()
