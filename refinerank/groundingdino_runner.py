"""Lazy, inference-only GroundingDINO runner for requested-time frames."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any


class GroundingDINORunner:
    """Load GroundingDINO once and predict source-space xyxy boxes."""

    def __init__(
        self,
        *,
        repo_path: Path,
        config_path: Path,
        checkpoint_path: Path,
        box_threshold: float,
        text_threshold: float,
        device: str,
    ) -> None:
        validated_box_threshold = _bounded_threshold(
            box_threshold, "box_threshold"
        )
        validated_text_threshold = _bounded_threshold(
            text_threshold, "text_threshold"
        )
        repo = Path(repo_path)
        config = Path(config_path)
        checkpoint = Path(checkpoint_path)
        for path in (repo, config, checkpoint):
            if not path.exists():
                raise FileNotFoundError(path)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        import numpy as np
        import torch
        from groundingdino.datasets import transforms
        from groundingdino.util.inference import load_model, preprocess_caption
        from groundingdino.util.utils import get_phrases_from_posmap
        from PIL import Image
        from torchvision.ops import box_convert

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GroundingDINO device cuda requested but unavailable")
        self.device = device
        self.box_threshold = validated_box_threshold
        self.text_threshold = validated_text_threshold
        self.preprocess_caption = preprocess_caption
        self.get_phrases_from_posmap = get_phrases_from_posmap
        self.box_convert = box_convert
        self.Image = Image
        self.np = np
        self.torch = torch
        self.transform = transforms.Compose(
            [
                transforms.RandomResize([800], max_size=1333),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        self.model = load_model(
            str(config),
            str(checkpoint),
            device=self.device,
        ).to(self.device)
        self.model.eval()
        for module in self.model.modules():
            if getattr(module, "use_checkpoint", False):
                setattr(module, "use_checkpoint", False)

    def predict_frames(
        self, frame_paths: Sequence[Path], caption: str
    ) -> list[dict[str, Any]]:
        """Predict ordered detections for each frame path."""

        normalized_caption = self.preprocess_caption(caption)
        return [
            self._predict_one(*self._load_image(path), normalized_caption)
            for path in frame_paths
        ]

    def close(self) -> None:
        """Release model references and the CUDA allocator cache."""

        model = self.model
        self.model = None
        del model
        if self.device == "cuda":
            self.torch.cuda.empty_cache()

    def _load_image(self, path: Path) -> tuple[Any, Any]:
        image_source = self.Image.open(path).convert("RGB")
        image = self.np.asarray(image_source)
        image_transformed, _ = self.transform(image_source, None)
        return image, image_transformed

    def _predict_one(
        self, source: Any, tensor: Any, caption: str
    ) -> dict[str, Any]:
        with self.torch.no_grad():
            outputs = self.model(tensor[None].to(self.device), captions=[caption])
            if self.device == "cuda":
                self.torch.cuda.synchronize()
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        mask = logits.max(dim=1)[0] > self.box_threshold
        tokenizer = self.model.tokenizer
        tokenized = tokenizer(caption)
        return self._postprocess(
            source,
            boxes[mask],
            logits[mask],
            tokenized,
            tokenizer,
        )

    def _postprocess(
        self,
        source: Any,
        boxes: Any,
        logits: Any,
        tokenized: Mapping[str, Any],
        tokenizer: Any,
    ) -> dict[str, Any]:
        height, width, _ = source.shape
        if len(boxes) == 0:
            return {
                "boxes_xyxy": [],
                "scores": [],
                "phrases": [],
                "image_size": [width, height],
                "raw_box_count": 0,
                "clipped_box_count": 0,
                "discarded_invalid_box_count": 0,
            }
        scores = logits.max(dim=1)[0]
        boxes_abs = boxes * self.torch.tensor([width, height, width, height])
        xyxy = self.box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy")
        phrases = [
            self.get_phrases_from_posmap(
                logit > self.text_threshold,
                tokenized,
                tokenizer,
            ).replace(".", "")
            for logit in logits
        ]
        order = self.torch.argsort(scores, descending=True)
        output_boxes = []
        output_scores = []
        output_phrases = []
        clipped_count = 0
        discarded_count = 0
        for index in order:
            raw_box = tuple(float(value) for value in xyxy[index].tolist())
            clipped_box = _clip_xyxy(
                raw_box,
                image_size=(width, height),
            )
            clipped_count += int(clipped_box != raw_box)
            if clipped_box[2] <= clipped_box[0] or clipped_box[3] <= clipped_box[1]:
                discarded_count += 1
                continue
            output_boxes.append(list(clipped_box))
            output_scores.append(float(scores[index].item()))
            output_phrases.append(phrases[int(index)])
        return {
            "boxes_xyxy": output_boxes,
            "scores": output_scores,
            "phrases": output_phrases,
            "image_size": [width, height],
            "raw_box_count": len(boxes),
            "clipped_box_count": clipped_count,
            "discarded_invalid_box_count": discarded_count,
        }


def refined_subboxes(
    box: Sequence[float], *, image_size: Sequence[int]
) -> tuple[tuple[str, tuple[float, float, float, float], float], ...]:
    """Return the frozen iter90 deterministic large-box refinements."""

    if len(box) != 4 or len(image_size) != 2:
        raise ValueError("box and image_size must contain 4 and 2 values")
    x1, y1, x2, y2 = (float(value) for value in box)
    image_width, image_height = (float(value) for value in image_size)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    if width < 64.0 or height < 64.0:
        return ()
    if (width * height) / max(1.0, image_width * image_height) < 0.06:
        return ()
    specs = (
        ("dino_refined_left_mid", (0.00, 0.18, 0.55, 0.52), 0.94),
        ("dino_refined_left_upper_mid", (0.00, 0.12, 0.58, 0.48), 0.91),
        ("dino_refined_center_left", (0.05, 0.18, 0.62, 0.56), 0.89),
    )
    output = []
    for name, (sx1, sy1, sx2, sy2), score_scale in specs:
        candidate = (
            min(max(x1 + sx1 * width, 0.0), image_width),
            min(max(y1 + sy1 * height, 0.0), image_height),
            min(max(x1 + sx2 * width, 0.0), image_width),
            min(max(y1 + sy2 * height, 0.0), image_height),
        )
        if candidate[2] > candidate[0] and candidate[3] > candidate[1]:
            output.append((name, candidate, score_scale))
    return tuple(output)


def _clip_xyxy(
    box: Sequence[float], *, image_size: Sequence[int]
) -> tuple[float, float, float, float]:
    if len(box) != 4 or len(image_size) != 2:
        raise ValueError("box and image_size must contain 4 and 2 values")
    x1, y1, x2, y2 = (float(value) for value in box)
    width, height = (float(value) for value in image_size)
    return (
        min(max(x1, 0.0), width),
        min(max(y1, 0.0), height),
        min(max(x2, 0.0), width),
        min(max(y2, 0.0), height),
    )


def _bounded_threshold(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a finite numeric value")
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field_name} must be finite and within [0, 1]")
    return parsed
