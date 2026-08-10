"""Frozen MedVLM query and proposal-ROI feature extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn

from refinerank.geometry import (
    area_weighted_roi_pool,
    inverse_window_group_order,
    merged_tokens_to_grid,
    normalize_box_xyxy,
    patch_tokens_to_grid,
)
from refinerank.types import (
    CandidateGroup,
    GroupFeatures,
    SpatialGridFeatures,
)

PROMPT_TEMPLATE = (
    "At {timestamp:.3f} seconds, identify the visual target that the question "
    "asks to localize.\nQuestion: {clean_question}"
)


@dataclass(frozen=True)
class FrozenStateSignature:
    """Cheap state/version signature used around inference-only forwards."""

    entries: tuple[tuple[str, tuple[int, ...], str, int], ...]


class FrozenMedVLMFeatureExtractor:
    """Run one frozen full-frame MedVLM forward and pool candidate features."""

    def __init__(
        self,
        model: nn.Module,
        processor: Any,
        *,
        visual_layers: Sequence[int] = (23, 31),
        merge_size: int = 2,
    ) -> None:
        self.model = model
        self.processor = processor
        self.visual_layers = tuple(sorted(set(int(value) for value in visual_layers)))
        self.merge_size = int(merge_size)
        self.visual = _resolve_visual(model)
        if any(
            index < 0 or index >= len(self.visual.blocks)
            for index in self.visual_layers
        ):
            raise ValueError(
                "requested visual hook layer is outside the vision backbone"
            )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("failed to freeze all MedVLM parameters")

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        visual_layers: Sequence[int] = (23, 31),
    ) -> "FrozenMedVLMFeatureExtractor":
        """Load the fixed local MedVLM and its unmodified processor config."""

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype_value = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype)
        if dtype_value is None:
            raise ValueError(f"unsupported MedVLM dtype: {dtype}")
        processor = AutoProcessor.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=dtype_value,
            attn_implementation="sdpa",
        ).to(device)
        return cls(
            model,
            processor,
            visual_layers=visual_layers,
            merge_size=int(model.config.vision_config.spatial_merge_size),
        )

    def extract_group(self, group: CandidateGroup) -> GroupFeatures:
        """Extract query, intermediate ROI, final ROI, and structural tensors."""

        features, _spatial = self.extract_group_with_spatial(group)
        return features

    def extract_group_with_spatial(
        self,
        group: CandidateGroup,
    ) -> tuple[GroupFeatures, SpatialGridFeatures]:
        """Extract pooled features and raw l31/final grids in one frozen forward."""

        prompt = PROMPT_TEMPLATE.format(
            timestamp=group.requested_timestamp,
            clean_question=group.clean_question,
        )
        if "[BOX]" in prompt:
            raise ValueError("frozen feature prompt must not depend on [BOX]")
        with Image.open(group.frame_path) as raw_image:
            image = raw_image.convert("RGB")
            if image.size != group.frame_size:
                raise ValueError(
                    f"frame size changed for {group.group_key}: "
                    f"{image.size} != {group.frame_size}"
                )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(group.frame_path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            rendered = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs = self.processor(
                text=[rendered],
                images=[image],
                padding=True,
                return_tensors="pt",
            )

        device = _model_input_device(self.model)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        captured: dict[str, Tensor] = {}
        handles = []
        for layer_index in self.visual_layers:
            handles.append(
                self.visual.blocks[layer_index].register_forward_hook(
                    _capture_hook(captured, f"layer_{layer_index}")
                )
            )
        handles.append(
            self.visual.register_forward_hook(_capture_hook(captured, "final"))
        )
        state_before = frozen_state_signature(self.model)
        try:
            with torch.inference_mode():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        if state_before != frozen_state_signature(self.model):
            raise RuntimeError(
                "MedVLM state/version signature changed during extraction"
            )

        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("MedVLM forward did not return language hidden states")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None or attention_mask.ndim != 2:
            raise RuntimeError("processor output lacks a 2D attention_mask")
        non_padding = torch.nonzero(attention_mask[0], as_tuple=False).flatten()
        if non_padding.numel() == 0:
            raise RuntimeError("prompt contains no non-padding token")
        q_last = hidden_states[-1][0, int(non_padding[-1])].float()
        q_mean = hidden_states[-1][0, non_padding].float().mean(dim=0)

        grid_tensor = inputs.get("image_grid_thw")
        if grid_tensor is None or grid_tensor.shape != (1, 3):
            raise ValueError("feature extraction requires exactly one image grid")
        grid_thw = tuple(int(value) for value in grid_tensor[0].tolist())
        window_index, _cu_window = self.visual.get_window_index(
            grid_tensor.detach().cpu()
        )
        normalized_boxes = torch.tensor(
            [
                normalize_box_xyxy(
                    candidate.bbox_xyxy,
                    width=group.frame_size[0],
                    height=group.frame_size[1],
                )
                for candidate in group.candidates
            ],
            dtype=torch.float32,
            device=device,
        )

        intermediate: dict[int, np.ndarray] = {}
        raw_grids: dict[int, Tensor] = {}
        for layer_index in self.visual_layers:
            tokens = _required_capture(captured, f"layer_{layer_index}").float()
            restored = inverse_window_group_order(
                tokens,
                window_index,
                merge_size=self.merge_size,
            )
            grid = patch_tokens_to_grid(
                restored,
                grid_thw,
                merge_size=self.merge_size,
            )
            raw_grids[layer_index] = grid
            pooled = area_weighted_roi_pool(grid, normalized_boxes)
            intermediate[layer_index] = _normalized_half_numpy(pooled)

        final_grid = merged_tokens_to_grid(
            _required_capture(captured, "final").float(),
            grid_thw,
            merge_size=self.merge_size,
        )
        roi_final = _normalized_half_numpy(
            area_weighted_roi_pool(final_grid, normalized_boxes)
        )
        features = GroupFeatures(
            group_key=group.group_key,
            candidate_ids=tuple(item.candidate_id for item in group.candidates),
            q_last=q_last.detach().cpu().numpy().astype(np.float16),
            q_mean=q_mean.detach().cpu().numpy().astype(np.float16),
            roi_l7=intermediate.get(7),
            roi_l15=intermediate.get(15),
            roi_l23=intermediate.get(23),
            roi_l31=intermediate.get(31),
            roi_final=roi_final,
            structure=np.asarray(
                [item.structure for item in group.candidates], dtype=np.float32
            ),
            missing=np.asarray(
                [item.missing for item in group.candidates], dtype=np.float32
            ),
            frame_size=group.frame_size,
            grid_thw=grid_thw,
        )
        features.validate()
        if 31 not in raw_grids:
            raise ValueError("spatial-grid extraction requires visual layer 31")
        spatial = SpatialGridFeatures(
            group_key=group.group_key,
            grid_l31=raw_grids[31].detach().cpu().numpy().astype(np.float16),
            grid_final=final_grid.detach().cpu().numpy().astype(np.float16),
            frame_size=group.frame_size,
            grid_thw=grid_thw,
            grid_l7=(
                raw_grids[7].detach().cpu().numpy().astype(np.float16)
                if 7 in raw_grids
                else None
            ),
        )
        spatial.validate()
        return features, spatial


def frozen_state_signature(module: nn.Module) -> FrozenStateSignature:
    """Capture tensor shapes, dtypes, and in-place version counters."""

    entries = []
    for name, tensor in module.state_dict().items():
        entries.append(
            (
                name,
                tuple(int(value) for value in tensor.shape),
                str(tensor.dtype),
                int(tensor._version),
            )
        )
    return FrozenStateSignature(tuple(entries))


def assert_backbone_not_in_optimizer(
    backbone: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Fail if an optimizer contains any frozen backbone parameter."""

    backbone_ids = {id(parameter) for parameter in backbone.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    overlap = backbone_ids & optimizer_ids
    if overlap:
        raise ValueError("optimizer contains frozen backbone parameters")


def _resolve_visual(model: nn.Module) -> nn.Module:
    candidate = getattr(getattr(model, "model", None), "visual", None)
    if candidate is None:
        candidate = getattr(model, "visual", None)
    if candidate is None or not hasattr(candidate, "blocks"):
        raise TypeError("model does not expose a Qwen2.5-VL visual backbone")
    return candidate


def _model_input_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("MedVLM has no parameters") from error


def _capture_hook(captured: dict[str, Tensor], name: str):
    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, Tensor):
            raise TypeError(f"hook {name} did not return a tensor")
        captured[name] = value.detach()

    return hook


def _required_capture(captured: Mapping[str, Tensor], name: str) -> Tensor:
    value = captured.get(name)
    if value is None:
        raise RuntimeError(f"MedVLM hook did not capture {name}")
    return value


def _normalized_half_numpy(features: Tensor) -> np.ndarray:
    normalized = F.normalize(features, p=2, dim=-1, eps=1.0e-12)
    return normalized.detach().cpu().numpy().astype(np.float16)
