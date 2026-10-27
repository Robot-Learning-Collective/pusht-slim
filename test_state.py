"""Regression checks for non-visual PushT representations and headless inference."""
import unittest
from unittest.mock import patch

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
import torch

from train_state import StateFlowPolicy, poses_from_keypoints, flow_loss, evaluate


class StateTests(unittest.TestCase):
    def test_flow_endpoints_have_distinct_time_features(self):
        policy = StateFlowPolicy()
        captured = []
        hook = policy.input.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        policy(torch.zeros(2, 16, 2), torch.tensor([0., 1.]), torch.zeros(2, 5))
        hook.remove()
        self.assertFalse(torch.equal(captured[0][0], captured[0][1]))
        self.assertEqual(captured[0][1, 16 * 2 + 5].item(), 1.)

    def test_pose_matches_simulator_body_origin(self):
        env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=None)
        try:
            with patch.object(env.unwrapped, "_render", side_effect=AssertionError("rendered")):
                for seed in range(5):
                    obs, _ = env.reset(seed=seed)
                    for _ in range(5):
                        points = env.unwrapped.get_keypoints(env.unwrapped._block_shapes)
                        pose, error = poses_from_keypoints(points.reshape(1, 16))
                        np.testing.assert_allclose(pose[0, :2], obs[2:4], atol=3e-5)
                        angle_error = (pose[0, 2] - obs[4] + np.pi) % (2 * np.pi) - np.pi
                        self.assertLess(abs(angle_error), 1e-6)
                        self.assertLess(error, 1e-10)
                        obs, *_ = env.step(np.array([200., 200.]))
        finally:
            env.close()

    def test_nonrigid_keypoints_rejected(self):
        with self.assertRaises(ValueError):
            poses_from_keypoints(np.zeros((1, 16)))

    def test_both_representations_have_finite_gradients_and_bounded_actions(self):
        torch.set_num_threads(4)
        for representation, dim in [("pose", 5), ("keypoints", 18)]:
            policy = StateFlowPolicy(representation=representation)
            state = torch.rand(4, dim)
            actions = torch.rand(4, 16, 2)
            mask = torch.ones_like(actions)
            mask[:, 8:] = 0
            loss = flow_loss(policy, (state, actions, mask))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters()))
            predicted = policy.inference(state)
            self.assertEqual(tuple(predicted.shape), (4, 16, 2))
            self.assertTrue(((predicted >= 0) & (predicted <= 512)).all())

    def test_both_evaluations_are_headless(self):
        from gym_pusht.envs.pusht import PushTEnv
        with patch.object(PushTEnv, "_render", side_effect=AssertionError("rendered")):
            for representation in ("pose", "keypoints"):
                result = evaluate(StateFlowPolicy(representation=representation), episodes=1)
                self.assertEqual(result["eval/episodes"], 1)


if __name__ == "__main__":
    unittest.main()
