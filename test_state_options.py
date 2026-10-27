"""Regression checks for evaluation options and CUDA graph state handling."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from eval_state import evaluation_seeds, evaluate
from train_state import StateFlowPolicy, flow_loss, main, parse_args


class OptionsTests(unittest.TestCase):
    def test_original_defaults(self):
        args = parse_args([])
        self.assertEqual((args.eval_every, args.eval_episodes, args.test_episodes), (10000, 50, 100))
        self.assertFalse(args.random_eval_seeds)
        self.assertEqual((args.entity, args.project), ('robot_learning_collective', 'pushT-slim'))
        self.assertIsNone(args.decay_steps)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(['--eval-every', '-1'])

    def test_fixed_random_and_replayed_seeds(self):
        self.assertEqual(evaluation_seeds(3, 10000), (10000, [10000, 10001, 10002], 10000))
        with patch('eval_state.secrets.randbits', side_effect=[123, 124]):
            a = evaluation_seeds(200, 10000, True)
            b = evaluation_seeds(200, 10000, True)
        self.assertNotEqual(a[1], b[1])
        self.assertEqual(len(set(a[1])), 200)
        record = dict(master_seed=a[0], reset_seeds=a[1], noise_seed=a[2])
        self.assertEqual(evaluation_seeds(200, 0, replay=record), a)

    def test_random_rollouts_replay_exactly(self):
        torch.set_num_threads(2)
        model = StateFlowPolicy(hidden=16, layers=1)
        with patch('eval_state.secrets.randbits', return_value=123):
            _, a = evaluate(model, episodes=2, random_seeds=True, return_details=True)
        _, b = evaluate(model, episodes=2, replay=a, return_details=True)
        self.assertEqual(a, b)
        self.assertTrue(all(success or length == 300 for success, length in zip(a['successes'], a['lengths'])))

    def test_reconstructed_data_preserves_actions_and_episode_boundaries(self):
        import hashlib
        import pandas as pd
        from train_state import load_reconstructed_data
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            table = pd.DataFrame({
                'index': [0, 1, 2, 3], 'episode_index': [0, 0, 1, 1],
                'frame_index': [0, 1, 0, 1], 'timestamp': [0., 0.1, 0., 0.1],
                'observation.state': [np.array([1., 2.], dtype=np.float32)] * 4,
                'action': [np.array([i, i+1], dtype=np.float32) for i in range(4)]})
            table.to_parquet(root / 'source.parquet')
            source_digest = hashlib.sha256((root / 'source.parquet').read_bytes()).hexdigest()
            table['observation.state'] = [np.array([1., 2., 3., 4., 0.], dtype=np.float32)] * 4
            table.to_parquet(root / 'states.parquet')
            (root / 'manifest.json').write_text(json.dumps(dict(
                dataset_id='zaringleb/pusht_mobile', revision='test', parquet_sha256=source_digest)))
            (root / 'audit.json').write_text(json.dumps({'artifact_sha256': {
                'states.parquet': hashlib.sha256((root / 'states.parquet').read_bytes()).hexdigest()}}))
            data, _, meta = load_reconstructed_data(root / 'states.parquet', 'zaringleb/pusht_mobile', None, 3)
            self.assertEqual(meta['total_episodes'], 2)
            self.assertTrue(torch.all(data[2][[1, 3]].sum((1, 2)) == 2))
            self.assertTrue(torch.all(data[1][[1, 3], 1:] == 0))
            np.testing.assert_array_equal(data[1][:, 0], np.stack(table['action']))
            with (root / 'states.parquet').open('ab') as f: f.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_reconstructed_data(root / 'states.parquet', 'zaringleb/pusht_mobile', None, 3)

    def run_training(self, output, extra_args):
        data = (torch.rand(16, 5), torch.rand(16, 2, 2), torch.ones(16, 2, 2))
        calls = []
        def fake_eval(model, episodes, *args, **kwargs):
            calls.append((episodes, kwargs))
            score = 0.8 if len(calls) == 1 else 0.2
            return ({'eval/success_rate': score, 'eval/mean_max_coverage': score,
                     'eval/episodes': episodes}, {'reset_seeds': list(range(episodes))})
        run = MagicMock(summary={}, url=None, id='test', dir=str(output))
        with patch('train_state.load_standard_data', return_value=(data, None, {})), \
             patch('train_state.evaluate', side_effect=fake_eval), \
             patch('train_state.wandb.init', return_value=run), \
             contextlib.redirect_stdout(io.StringIO()):
            main(['--steps', '3', '--device', 'cpu', '--hidden', '16', '--layers', '1',
                  '--horizon', '2', '--action-chunk', '1', '--batch-size', '4', '--threads', '2',
                  '--output', str(output), *extra_args])
        return calls, run

    def test_periodic_evaluation_selects_best_then_tests_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            calls, run = self.run_training(output, ['--eval-every', '2'])
            self.assertEqual([n for n, _ in calls], [50, 50, 100])
            self.assertEqual([kw['seed'] for _, kw in calls], [10000, 10000, 20000])
            self.assertTrue(all(not kw['random_seeds'] for _, kw in calls))
            results = json.loads((output / 'results.json').read_text())
            self.assertEqual(results['selected_step'], 2)
            self.assertEqual(torch.load(output / 'last.pt', weights_only=False)['step'], 3)
            self.assertEqual(torch.load(output / 'best.pt', weights_only=False)['step'], 2)
            run.finish.assert_called_once_with()

    def test_end_only_runs_one_requested_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            calls, _ = self.run_training(output,
                ['--eval-every', '0', '--eval-episodes', '200', '--random-eval-seeds'])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], 200)
            self.assertTrue(calls[0][1]['random_seeds'])
            self.assertFalse((output / 'best.pt').exists())
            self.assertEqual(json.loads((output / 'results.json').read_text())['selected_step'], 3)
            self.assertEqual(len((output / 'evaluations.jsonl').read_text().splitlines()), 1)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_graph_restores_setup_and_matches_updates(self):
        from cuda_step import GraphStep
        for representation, dim in [('pose', 5), ('keypoints', 18)]:
            torch.manual_seed(42)
            model = StateFlowPolicy(hidden=32, layers=1, representation=representation).cuda()
            fast_model = copy.deepcopy(model)
            ema, fast_ema = copy.deepcopy(model), copy.deepcopy(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            fast_optimizer = torch.optim.AdamW(fast_model.parameters(), lr=3e-4,
                                               weight_decay=1e-4, fused=True, capturable=True)
            graph = GraphStep(fast_model, fast_ema, fast_optimizer, 8)
            for left, right in zip(model.parameters(), fast_model.parameters()):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            self.assertTrue(all(s['step'].item() == 0 for s in fast_optimizer.state.values()))
            batch = (torch.rand(8, dim), torch.rand(8, 16, 2), torch.ones(8, 16, 2))
            rng_a, rng_b = torch.Generator().manual_seed(44), torch.Generator().manual_seed(44)
            for step in range(20):
                lr = 1e-4 if step < 10 else 3e-4
                optimizer.param_groups[0]['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                loss = flow_loss(model, tuple(t.cuda() for t in batch), rng_a)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                with torch.no_grad():
                    for target, source in zip(ema.parameters(), model.parameters()): target.lerp_(source, 0.005)
                fast_loss, _ = graph.step(batch, rng_b, lr)
                self.assertAlmostEqual(loss.item(), fast_loss, places=5)
            for left, right in zip(ema.parameters(), fast_ema.parameters()):
                torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-4)
            self.assertTrue(all(s['step'].item() == 20 for s in fast_optimizer.state.values()))


if __name__ == '__main__':
    unittest.main()
