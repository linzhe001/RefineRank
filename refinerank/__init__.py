"""RefineRank: frozen MedVLM + GroundingDINO candidates refined by RefineNet.

Core package for the ECCV 2026 MedVidU workshop submission. The pipeline is:

1. GroundingDINO tube generation (``groundingdino_runner``).
2. GT-free candidate pool + frozen MedVLM feature caches
   (``candidates``, ``features``, ``cache``).
3. RefineNet (``proposal.QueryConditionedProposalAdapter``) trained with
   ``proposal_training`` and decoded with the parameter-free ``decode``.
"""

__version__ = "1.0.0"
