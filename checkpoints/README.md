# Checkpoints

RefineRank 推理/训练所需的全部权重，平铺结构如下：

```
checkpoints/
├── vlm/                              # 冻结的 MedVLM（Qwen2.5-VL 架构，HF 格式，全部文件直接放这里）
│   ├── config.json / model-0000X-of-00004.safetensors / tokenizer.json / ...
├── grounding_dino/                   # 冻结的 GroundingDINO 权重
│   └── groundingdino_swinb_cogcoor.pth
└── refinenet/                        # 唯一可训练模块 RefineNet（约 1.25M 参数）
    ├── proposal_adapter_full.pt      # 随论文发布的最终提交权重
    └── deployment_manifest.json
```

- 从 Hugging Face 一键恢复本目录：`hf download linzher/RefineRank --local-dir .`
- `vlm/` 与 `grounding_dino/` 在训练与推理中都保持冻结，仅用于特征与候选框提取。
- `python interface.py predict` 不带 `--checkpoint` 时优先使用本目录的平铺权重；
  训练新产生的 `run_<时间戳>/` 目录（`python interface.py train` 的输出）也会被自动发现。
- 大权重文件不要提交到 git。
