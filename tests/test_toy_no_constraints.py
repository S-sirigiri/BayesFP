"""Regression checks for guidance when no inference constraints are present."""
import importlib
import unittest

import torch


class NoConstraintsTests(unittest.TestCase):
    def test_zero_cost_is_differentiable(self):
        for name, env_class in [
            ('square_obstacle_2d', 'SquareObstacleEnv'),
            ('triangle_obstacles_2d', 'TriangleObstacleEnv'),
        ]:
            with self.subTest(example=name):
                module = f'src.ToyExamples.{name}'
                env = getattr(importlib.import_module(module + '.env'), env_class)()
                env.inference_constraints = []
                cost = importlib.import_module(module + '.cost')
                traj = torch.randn(3, 8, 2, dtype=torch.float64, requires_grad=True)
                values = cost.constraint_cost(traj, env)
                grad = torch.autograd.grad(values.sum(), traj)[0]
                total, helper_grad = cost.constraint_cost_and_grad(traj, env)
                self.assertEqual(values.shape, (3,))
                self.assertEqual(values.dtype, traj.dtype)
                self.assertEqual(total.item(), 0.0)
                torch.testing.assert_close(grad, torch.zeros_like(traj))
                torch.testing.assert_close(helper_grad, grad)

    def test_guidance_matches_vanilla_without_constraints(self):
        from src.ToyExamples.square_obstacle_2d.diffusion import DDPM
        from src.ToyExamples.square_obstacle_2d.env import SquareObstacleEnv
        from src.ToyExamples.square_obstacle_2d.samplers import (
            FKCConfig, sample_fkc, sample_linear_combo, sample_vanilla,
        )
        env = SquareObstacleEnv(scenario='none')
        ddpm = DDPM(num_steps=8, schedule='cosine')
        def model(x, t):
            return torch.zeros_like(x)
        common = dict(traj_len=8, env=env, device='cpu')
        torch.manual_seed(0)
        vanilla = sample_vanilla(model, ddpm, num_samples=4, **common)
        torch.manual_seed(0)
        guided = sample_linear_combo(model, ddpm, num_samples=4, **common)
        torch.testing.assert_close(guided, vanilla)
        for mode in ['energy', 'girsanov']:
            with self.subTest(weight_mode=mode):
                torch.manual_seed(0)
                particles, info = sample_fkc(
                    model, ddpm,
                    cfg=FKCConfig(num_particles=4, weight_mode=mode, return_best=False),
                    **common,
                )
                torch.testing.assert_close(particles, vanilla)
                self.assertEqual(info['resample_steps'], [])
                for weights in info['weight_log']:
                    torch.testing.assert_close(weights, torch.zeros_like(weights))


if __name__ == '__main__':
    unittest.main()
