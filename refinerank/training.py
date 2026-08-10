"""Training hyperparameter contract for RefineNet.

Extracted from the research repo's ``ranking/training.py``. Only
``OOFConfig`` is kept; the historical selector OOF/gate helpers are
comparison-path code and are intentionally not part of this package.

``OOFConfig`` field names are load-bearing: RefineNet checkpoints store
``train_config`` as ``asdict(OOFConfig)`` and deserialization reconstructs
the dataclass from those keys.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OOFConfig:
    """Fixed optimizer and grouped-OOF configuration."""

    folds: int = 5
    epochs: int = 40
    groups_per_dataset: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    seed: int = 20260719
    eval_batch_size: int = 48
