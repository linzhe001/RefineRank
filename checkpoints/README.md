# Checkpoints

RefineRank 推理/训练所需的全部权重，目录结构固定：

```
checkpoints/
├── vlm/                                      # 冻结的 MedVLM 骨干（Qwen2.5-VL 架构，HF 格式）
│   └── uAI-NEXUS-MedVLM-1.0a-7B-RL/          # config.json / model.safetensors / tokenizer 等
├── grounding_dino/                           # 冻结的 GroundingDINO 权重
│   └── groundingdino_swinb_cogcoor.pth       # SwinB（另需 GroundingDINO repo 的 config .py，见主 README）
└── refinenet/                                # 唯一可训练模块 RefineNet（约 1.25M 参数）
    └── run_iter132_submission/               # 随论文发布的最终提交权重
        ├── checkpoints/proposal_adapter_full.pt
        └── deployment_manifest.json
```

- `vlm/` 与 `grounding_dino/` 在训练与推理中都保持冻结，仅用于特征与候选框提取。
- `refinenet/` 下每个 `run_*/` 是一次训练的产出（`python interface.py train` 默认写到这里）；
  `python interface.py predict` 不带 `--checkpoint` 时自动选择最新一个含 `deployment_manifest.json` 的 `run_*` 目录。
- 大权重文件不要提交到 git。
