"""Global seed management for full reproducibility.

Usage:
    from app.seed import set_global_seed
    set_global_seed(42)  # call once at startup

Guarantees:
    - Identical results across runs on same platform
    - Deterministic data generation, model training, and evaluation
    - Seed propagated to: random, numpy, torch, sklearn, LightGBM
"""

from __future__ import annotations

import os
import random

import numpy as np

GLOBAL_SEED = 42

_rng: np.random.Generator | None = None


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value. Default 42.
    """
    global _rng, GLOBAL_SEED

    GLOBAL_SEED = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # Legacy global seed (kept for backward compatibility with libraries that
    # still read the global RandomState).
    np.random.seed(seed)

    # Preferred: modern Generator API (avoids the deprecated global state).
    _rng = np.random.default_rng(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_rng() -> np.random.Generator:
    """Return the module-level numpy Generator, initialising on first call."""
    global _rng
    if _rng is None:
        set_global_seed()
    return _rng  # type: ignore[return-value]


def get_seed() -> int:
    """Return the global seed value."""
    return GLOBAL_SEED
