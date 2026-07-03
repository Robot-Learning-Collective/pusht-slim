"""E6: eval-only sweep on a trained checkpoint — tests whether the last-5%-coverage
plateau is helped by tighter closed-loop control (smaller action chunk) or more
denoising steps. No retraining."""
import sys, torch, train

CKPT = "checkpoints_E1-obs2-flow/checkpoint_step_50000.pt"
N_EP = 30  # fewer episodes to keep the sweep cheap

dev = torch.device("cuda")
ckpt = torch.load(CKPT, map_location=dev, weights_only=False)
sd = ckpt["ema_model_state_dict"]
m = train.DiTPolicy(hidden_dim=600, n_layers=6, n_heads=8,
                    obs_horizon=2, objective="flow", vision_mode="global").to(dev)
m.load_state_dict(sd)
m.eval()
print(f"loaded {CKPT} (step {ckpt.get('step')})")

print("== action-chunk sweep (n_steps=10) ==")
for cs in [8, 4, 2, 1]:
    train.ACTION_CHUNK_SIZE = cs
    r = train.evaluate(m, dev, n_episodes=N_EP, render=False, n_steps=10)
    print(f"RESULT chunk={cs} n_steps=10 SR={r['success_rate']:.3f} AMR={r['avg_max_reward']:.3f}")

print("== denoising-steps sweep (chunk=8) ==")
train.ACTION_CHUNK_SIZE = 8
for ns in [10, 16, 50]:
    r = train.evaluate(m, dev, n_episodes=N_EP, render=False, n_steps=ns)
    print(f"RESULT chunk=8 n_steps={ns} SR={r['success_rate']:.3f} AMR={r['avg_max_reward']:.3f}")
print("SWEEP DONE")
