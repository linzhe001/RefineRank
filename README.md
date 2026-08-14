# RefineRank: Joint Box Refinement and Ranking for Surgical Spatio-Temporal Grounding

Official code release for the ECCV 2026 MedVidU Workshop paper.

> RefineRank couples two **frozen** backbones — a MedVLM (Qwen2.5-VL
> architecture) and GroundingDINO — with a compact **1.25M-parameter**
> trainable module, **RefineNet**. RefineNet uses MedVLM language and regional
> features to predict coordinate corrections and box-quality scores for
> GroundingDINO proposals; a **parameter-free decoder** then returns the
> highest-scoring original or refined box.

- **Paper (camera-ready source)**: [`paper/main.tex`](paper/main.tex) — compile with `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
- **All checkpoints**: [huggingface.co/linzher/RefineRank](https://huggingface.co/linzher/RefineRank)
  hosts the complete `checkpoints/` tree — the frozen MedVLM (~16 GB), the
  frozen GroundingDINO SwinB weights (~895 MB), and the trained RefineNet
  (`run_iter132_submission`, the exact checkpoint behind the paper's
  MedVidBench submission). Download with
  `hf download linzher/RefineRank --local-dir .`.
- **Headline result**: 0.421 STG mIoU on the archived MedVidBench Community
  leaderboard snapshot (27 July 2026) — the best STG mIoU among the ten
  ranking metrics on that snapshot.
- **Controlled evaluation** (video-separated split, three datasets:
  CholecTrack20 / CoPESD / EgoSurgery): STG mIoU **0.2719 → 0.4534** over the
  MedVLM + GroundingDINO baseline; candidate oracle 0.6772 → 0.7302.

## Repository layout

```
RefineRank/
├── interface.py            # single user-facing entry point (see below)
├── requirements.txt
├── configs/refinerank.yaml # pipeline config (frozen backbones + RefineNet + decoder)
├── refinerank/             # core package (frozen pipeline, no ablation code)
│   ├── features.py         # FrozenMedVLMFeatureExtractor (visual layers 7/15/23/31)
│   ├── candidates.py       # GT-free candidate pool + diversity-MMR shortlist (K=48)
│   ├── proposal.py         # RefineNet: QueryConditionedProposalAdapter
│   ├── proposal_training.py# full-fit training / checkpointed prediction
│   ├── decode.py           # parameter-free decoder + MedVidBench formatting
│   ├── cache.py            # CandidateStore / FeatureCache / SpatialGridCache
│   └── ...                 # geometry, coordinates, repool, timeline, runners
├── checkpoints/            # flat folders, core files directly inside
│   ├── vlm/                            # frozen MedVLM, HF format (16 GB)
│   ├── grounding_dino/groundingdino_swinb_cogcoor.pth  # frozen detector
│   └── refinenet/                      # released RefineNet weights
│       ├── proposal_adapter_full.pt
│       └── deployment_manifest.json
├── tests/                  # 152 core unit tests (pytest)
└── paper/                  # camera-ready LaTeX source + figures
```

## Installation

```bash
cd RefineRank
pip install -r requirements.txt   # torch, transformers, numpy, Pillow, PyYAML, torchvision, pytest
```

Requirements: Python 3.10+, a CUDA GPU (feature extraction and training run
in bfloat16 on `cuda`), and a local checkout of
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) (only needed
for the `dino-tubes` step; point `model.groundingdino_repo` /
`model.groundingdino_config` in `configs/refinerank.yaml` at it).

## Checkpoints

All weights live under `checkpoints/` (see
[`checkpoints/README.md`](checkpoints/README.md)):

| Component | Role | Path |
|---|---|---|
| MedVLM (uAI-NEXUS-MedVLM-1.0a-7B-RL) | frozen, feature extraction | `checkpoints/vlm/` |
| GroundingDINO SwinB | frozen, box proposals | `checkpoints/grounding_dino/` |
| RefineNet | trainable (1.25M) | `checkpoints/refinenet/` |

The released RefineNet weights (`proposal_adapter_full.pt`, the
`run_iter132_submission` run) are the exact checkpoint
behind the paper's MedVidBench submission. The full `checkpoints/` tree
(frozen MedVLM, frozen GroundingDINO, trained RefineNet) is hosted at
[huggingface.co/linzher/RefineRank](https://huggingface.co/linzher/RefineRank);
download it into the repository root with
`hf download linzher/RefineRank --local-dir .` (see
[`checkpoints/README.md`](checkpoints/README.md) for the expected layout).
`interface.py predict` discovers the flat `checkpoints/refinenet/` weights
automatically (new training runs under `checkpoints/refinenet/run_*/` are
discovered too); override with `--checkpoint <dir or .pt>`.

## Data preparation

RefineRank expects the MedVidU-style data layout used by the paper
(train/val split JSONs, decoded frames, and the MedVidBench public test set).
Edit the `data:` section of `configs/refinerank.yaml` to point at your local
dataset root — the shipped values are the original research-machine paths
(`/mnt/f/Datasets/medvidu/...`) and **must be adapted**.

Two upstream inputs are required before the pipeline can start:

- an initial requested-time candidate store (one row per
  sample × requested timestamp), and
- a query-plan cache (per-sample detector queries).

Both are produced by the MedVidU data-preparation step of the evaluation
framework and are passed to `interface.py dino-tubes` via
`--candidate-store` / `--query-plan-cache`.

## Usage

Everything goes through `interface.py`:

```bash
python interface.py --help
python interface.py <command> --help
```

### 1. Generate GroundingDINO tubes (per phase)

```bash
python interface.py dino-tubes --phase train \
    --candidate-store /path/to/requested_time_candidates.jsonl \
    --query-plan-cache /path/to/query_plan_cache
python interface.py dino-tubes --phase public \
    --candidate-store /path/to/public_requested_time_candidates.jsonl \
    --query-plan-cache /path/to/public_query_plan_cache
```

Writes a tube JSONL (`tubes.jsonl`) under `runs/dino_tubes_<phase>_<ts>/`.
Point `data.train_tube_candidates` / `data.val_tube_candidates` in the config
at these files.

### 2. Build caches (candidates → MedVLM features → spatial grids)

```bash
python interface.py cache --phase train     # writes runs/cache_train/
python interface.py cache --phase public    # writes runs/cache_public/
```

This runs three steps sequentially: GT-free candidate pool construction
(shortlist `diversity_mmr_k48`: 16 native anchors + MMR fill to K=48), frozen
MedVLM feature caching (`q_last`, ROI final 3584-d, ROI blocks 7/15/23/31
1280-d), and raw spatial-grid caching (for exact predict-time ROI re-pooling).

### 3. Train RefineNet

```bash
python interface.py train                   # reads runs/cache_train/ by default
```

Full-fit training on all train groups (40 epochs, AdamW, lr 3e-4, weight
decay 1e-2, grad clip 1.0, seed 20260719; both backbones stay frozen and are
never touched by the optimizer). Output:

```
checkpoints/refinenet/run_<UTC>/
├── checkpoints/proposal_adapter_full.pt   # weights + adapter/train config + standardizer
├── deployment_manifest.json               # deployment contract for prediction
└── proposal_target_store.json
```

### 4. Inference

```bash
python interface.py predict                 # auto-discovers checkpoints/refinenet/
python interface.py predict --checkpoint checkpoints/refinenet/proposal_adapter_full.pt
```

Runs RefineNet over the public cache, exactly re-pools refined boxes on the
raw spatial grids, and applies the parameter-free decoder (top-1 argmax;
CholecTrack20 uses a fixed shape-DP with λ=0.2 over the top-20). Writes the
MedVidBench STG prediction rows (780-row public contract enforced by default;
disable with `--no-require-public-test`):

```json
{"source_index": 0, "id": "...", "qa_type": "stg",
 "prediction": "12.5 seconds: [x1, y1, x2, y2]; 30.0 seconds: [x1, y1, x2, y2]"}
```

### 5. Tests

```bash
python interface.py test                    # = python -m pytest tests/ -q
```

152 unit tests covering geometry/ROI pooling, cache round-trips and hashes,
RefineNet forward/loss/shortlist, re-pooling, and the decoder.

## Configuration

`configs/refinerank.yaml` has six sections: `model` (frozen backbone paths,
dtype, visual layers), `data` (splits, tubes, shortlist policy),
`proposal_adapter` (RefineNet architecture and bounded-correction limits),
`spatial_cache` (re-pooling tolerances), `train` (optimizer), and `decoder`
(parameter-free selection). Field names in `proposal_adapter` and `train` are
load-bearing — checkpoints deserialize them into `ProposalAdapterConfig` /
`OOFConfig`.

