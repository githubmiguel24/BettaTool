# exponential moving average of model weights; the EMA copy is what gets evaluated, saved and used as the teacher

from __future__ import annotations

import copy

import torch


class ModelEMA:
    # keeps a second copy of the model whose weights follow the training weights as ema = d * ema + (1 - d) * current
    def __init__(self, model: torch.nn.Module, decay: float = 0.998) -> None:
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        # warm up the decay so the first steps are not dominated by the random / pretrained start
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        current = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(d).add_(current[key].detach(), alpha=1.0 - d)
            else:
                value.copy_(current[key])
