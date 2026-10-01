# helper to lock down global seeds across python, numpy, and torch for Reproducibility

from __future__ import annotations

import os
import random

import numpy as np
import torch


# sets the global seed for all random generators and cudnn
def seed_everything(seed: int, deterministic_cudnn: bool = True) -> None:
    random.seed(seed)  # standard python rng
    np.random.seed(seed)  # numpy rng
    torch.manual_seed(seed)  # torch cpu
    torch.cuda.manual_seed_all(seed)  # seed all availble gpus
    os.environ["PYTHONHASHSEED"] = str(seed)  # lock python hash seed

    # forces deterministic cudnn ops, trades off a bit of speed for consistency
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False