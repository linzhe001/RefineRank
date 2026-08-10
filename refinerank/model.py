"""Candidate-ranking loss shared by the RefineNet adapter objective.

Extracted from the research repo's ``ranking/model.py``. Only the
parameter-free ``ranking_loss`` is kept; the ablation ranker classes
(QueryAwareSetRanker, HierarchicalParentChildRanker, BilinearQueryRoiRanker,
SameFeatureMLPRanker, FinalL31CandidateMLPRanker, hierarchical_ranking_loss)
are comparison baselines and are intentionally not part of this package.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def ranking_loss(
    logits: Tensor,
    ious: Tensor,
    padding_mask: Tensor,
    *,
    temperature: float = 0.1,
    regression_weight: float = 0.25,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute listwise CE plus calibrated SmoothL1 regression."""

    if logits.shape != ious.shape or logits.shape != padding_mask.shape:
        raise ValueError("logits, ious, and padding_mask must have identical shapes")
    valid = ~padding_mask
    if not valid.any(dim=1).all():
        raise ValueError("every ranking group must contain a valid candidate")
    safe_logits = logits.masked_fill(padding_mask, torch.finfo(logits.dtype).min)
    valid_ious = ious.masked_fill(padding_mask, 0.0)
    nonzero_groups = valid_ious.max(dim=1).values > 0
    log_prob = F.log_softmax(safe_logits, dim=1)
    targets = torch.softmax(
        (valid_ious / temperature).masked_fill(
            padding_mask,
            torch.finfo(logits.dtype).min,
        ),
        dim=1,
    )
    per_group = -(targets * log_prob.masked_fill(padding_mask, 0.0)).sum(dim=1)
    listwise = (
        per_group[nonzero_groups].mean() if nonzero_groups.any() else logits.sum() * 0.0
    )
    regression = F.smooth_l1_loss(
        torch.sigmoid(logits[valid]),
        ious[valid],
    )
    total = listwise + regression_weight * regression
    return total, {
        "listwise": listwise.detach(),
        "regression": regression.detach(),
        "nonzero_group_count": nonzero_groups.sum().detach(),
    }
