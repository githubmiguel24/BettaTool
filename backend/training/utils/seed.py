"""Global determinism: one function seeds Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_cudnn: bool = True) -> None:
    """Seeds Python's `random`, NumPy, and PyTorch (CPU + all CUDA devices).

    Args:
        seed: the global seed.
        deterministic_cudnn: if True, sets
            `torch.backends.cudnn.deterministic = True` and
            `torch.backends.cudnn.benchmark = False`. This trades some
            throughput for reproducibility, as required by Build Prompt v2 §1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
