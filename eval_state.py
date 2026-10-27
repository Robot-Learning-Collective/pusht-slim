"""Batched CPU evaluation of state policies, without rendering."""
import copy
import secrets
import time
import gymnasium as gym
import numpy as np
import torch

def evaluation_seeds(episodes, seed, random_seeds=False, replay=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if replay is not None:
        if len(replay["reset_seeds"]) != episodes:
            raise ValueError("Replay seed count does not match episodes")
        return replay["master_seed"], replay["reset_seeds"], replay["noise_seed"]
    if random_seeds:
        master_seed = secrets.randbits(64)
        rng = np.random.default_rng(master_seed)
        return (master_seed, rng.choice(2**32, size=episodes, replace=False).tolist(),
                int(rng.integers(0, 2**63)))
    return seed, list(range(seed, seed + episodes)), seed


@torch.no_grad()
def evaluate(model, episodes=50, chunk=8, denoise_steps=10, seed=10000,
             env_id="gym_pusht/PushT-v0", *, random_seeds=False, return_details=False, replay=None):
    """Fixed resets by default; optional fresh random seeds or saved replay details."""
    start = time.perf_counter()
    model = copy.deepcopy(model).cpu().eval()
    master_seed, reset_seeds, noise_seed = evaluation_seeds(episodes, seed, random_seeds, replay)
    if env_id == "PushTMobile-v0":
        import pusht_mobile_env  # noqa: F401
    else:
        import gym_pusht  # noqa: F401
    obs_type = "state" if model.representation == "pose" else "environment_state_agent_pos"
    def vector(obs):
        if isinstance(obs, dict):
            return np.concatenate([obs["agent_pos"], obs["environment_state"]])
        return obs
    envs = [gym.make(env_id, obs_type=obs_type, render_mode=None)
            for _ in range(episodes)]
    generator = torch.Generator().manual_seed(noise_seed)
    obs = [env.reset(seed=reset_seeds[i])[0] for i, env in enumerate(envs)]
    initial_states = [vector(o).tolist() for o in obs]
    lengths = np.zeros(episodes, dtype=int)
    active = list(range(episodes))
    best = np.zeros(episodes)
    success = np.zeros(episodes, dtype=bool)
    try:
        while active:
            actions = model.inference(torch.from_numpy(np.stack([vector(obs[i]) for i in active]).astype(np.float32)),
                                      denoise_steps, generator).numpy()
            remaining = []
            for j, i in enumerate(active):
                done = False
                for action in actions[j, :chunk]:
                    obs[i], reward, terminated, truncated, info = envs[i].step(action)
                    lengths[i] += 1
                    best[i] = max(best[i], info["coverage"])
                    success[i] |= info["is_success"]
                    done = terminated or truncated
                    if done:
                        break
                if not done:
                    remaining.append(i)
            active = remaining
    finally:
        for env in envs:
            env.close()
    metrics = {"eval/seconds": time.perf_counter() - start, "eval/success_rate": float(success.mean()),
            "eval/mean_max_coverage": float(best.mean()),
            "eval/successes": int(success.sum()), "eval/episodes": episodes}
    details = dict(master_seed=master_seed, reset_seeds=reset_seeds, noise_seed=noise_seed,
                   initial_states=initial_states, successes=success.tolist(),
                   max_coverage=best.tolist(), lengths=lengths.tolist())
    return (metrics, details) if return_details else metrics
