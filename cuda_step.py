"""Capture one complete training update while retaining CPU data/RNG streams."""
import copy
import math
import torch
from torch import nn


class GraphStep:
    def __init__(self, model, ema, optimizer, batch_size):
        self.model, self.ema, self.optimizer = model, ema, optimizer
        horizon = model.horizon
        shapes = [(batch_size, 5 if model.representation == "pose" else 18), (batch_size, horizon, 2),
                  (batch_size, horizon, 2), (batch_size, horizon, 2), (batch_size,)]
        sizes = [math.prod(shape) for shape in shapes]
        self.host = torch.zeros(sum(sizes) + 1, pin_memory=True)
        self.device = torch.zeros_like(self.host, device='cuda')
        self.host_views, self.views = [], []
        offset = 0
        for shape, size in zip(shapes, sizes):
            self.host_views.append(self.host[offset:offset+size].view(shape))
            self.views.append(self.device[offset:offset+size].view(shape))
            offset += size
        self.lr = self.device[-1]
        self.device[-1] = 3e-4
        self.views[2].fill_(1)
        self.views[4].fill_(0.5)
        for group in optimizer.param_groups:
            group['lr'] = self.lr
        initial = copy.deepcopy(model.state_dict())
        initial_ema = copy.deepcopy(ema.state_dict())
        # Allocate optimizer and gradient state on a side stream before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._update()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._update()
        # Warmup/capture are setup, not demonstration updates. Restore every
        # state in place because the graph holds pointers to these allocations.
        with torch.no_grad():
            model.load_state_dict(initial)
            ema.load_state_dict(initial_ema)
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        value.zero_()
        torch.cuda.synchronize()

    def _update(self):
        self.optimizer.zero_grad(set_to_none=False)
        state, action, mask, noise, t = self.views
        target = action / 256 - 1
        noisy = (1 - t[:, None, None]) * noise + t[:, None, None] * target
        error = (self.model(noisy, t, state) - (target - noise)).square()
        loss = (error * mask).sum() / mask.sum()
        loss.backward()
        # Check finiteness on the returned statistics after replay. Checking
        # inside capture would introduce a forbidden host synchronization.
        grad = nn.utils.clip_grad_norm_(self.model.parameters(), 10., error_if_nonfinite=False)
        self.optimizer.step()
        with torch.no_grad():
            for target_param, source in zip(self.ema.parameters(), self.model.parameters()):
                target_param.lerp_(source, 0.005)
        self.stats = torch.stack([loss.detach(), grad])

    def step(self, batch, generator, lr):
        for destination, value in zip(self.host_views[:3], batch):
            destination.copy_(value)
        # Generate on CPU in exactly the reference run's order, then transfer
        # the full input and learning rate together in one pinned copy.
        self.host_views[3].copy_(torch.randn(self.host_views[3].shape, generator=generator))
        self.host_views[4].copy_(torch.rand(len(batch[0]), generator=generator))
        self.host[-1] = lr
        self.device.copy_(self.host, non_blocking=True)
        self.graph.replay()
        loss, grad = self.stats.cpu().tolist()
        if not math.isfinite(loss) or not math.isfinite(grad):
            raise RuntimeError('Nonfinite training update; checkpoint not saved')
        return loss, grad
