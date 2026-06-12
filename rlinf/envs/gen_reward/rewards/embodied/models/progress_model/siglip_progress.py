# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoImageProcessor, SiglipVisionModel



def _infer_vision_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    candidates = (
        getattr(getattr(config, "vision_config", None), "hidden_size", None),
        getattr(config, "vision_hidden_size", None),
        getattr(config, "hidden_size", None),
    )
    for value in candidates:
        if value is not None:
            return int(value)
    raise ValueError("cannot infer SigLIP vision hidden size")


class SigLIPMLPProgressModel(nn.Module):
    """Inference-only SigLIP vision backbone plus MLP progress head."""

    def __init__(
        self,
        vision_model: nn.Module,
        *,
        mlp_hidden_size: int,
        mlp_layers: int,
        mlp_dropout: float,
        output_activation: str = "sigmoid",
        v_min: float = 0.0,
        v_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.output_activation = str(output_activation)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

        hidden_size = _infer_vision_hidden_size(vision_model)
        layers: list[nn.Module] = []
        in_size = hidden_size
        for _ in range(max(0, int(mlp_layers) - 1)):
            layers.append(nn.Linear(in_size, int(mlp_hidden_size)))
            layers.append(nn.GELU())
            if float(mlp_dropout) > 0.0:
                layers.append(nn.Dropout(float(mlp_dropout)))
            in_size = int(mlp_hidden_size)
        layers.append(nn.Linear(in_size, 1))
        self.regression_head = nn.Sequential(*layers)

        for param in self.vision_model.parameters():
            param.requires_grad_(False)
        self.eval()

    @staticmethod
    def _pool_outputs(outputs: Any) -> torch.Tensor:
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise ValueError("SigLIP vision backbone did not return hidden states")
            hidden = hidden_states[-1]
        return hidden[:, 0, :]

    def _activate(self, raw: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "sigmoid":
            unit = torch.sigmoid(raw)
            return self.v_min + (self.v_max - self.v_min) * unit
        if self.output_activation in {"identity", "none"}:
            return raw
        raise ValueError(f"unsupported output_activation={self.output_activation!r}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(
                "pixel_values must have shape (batch, channels, height, width); "
                f"got {tuple(pixel_values.shape)}"
            )
        outputs = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )
        pooled = self._pool_outputs(outputs)
        raw = self.regression_head(pooled.float()).view(-1)
        return self._activate(raw)

    def load_progress_head(self, checkpoint_dir: str | Path) -> None:
        path = Path(checkpoint_dir) / "siglip_mlp_head.safetensors"
        if not path.exists():
            raise FileNotFoundError(path)
        state = load_file(str(path))
        head_state = {
            key.removeprefix("regression_head."): value
            for key, value in state.items()
            if key.startswith("regression_head.")
        }
        if not head_state:
            raise ValueError(f"no regression_head weights found in {path}")
        self.regression_head.load_state_dict(head_state, strict=True)


class SigLIPProgressPredictor:
    """Maps RGB frames to progress values with a SigLIP MLP checkpoint."""

    def __init__(
        self,
        image_processor: Any,
        model: SigLIPMLPProgressModel,
        device: torch.device,
        batch_size: int,
    ) -> None:
        self.image_processor = image_processor
        self.model = model
        self.device = device
        self.batch_size = int(batch_size)

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_path: str | Path,
        checkpoint_dir: str | Path,
        device: str | None = None,
        batch_size: int = 32,
    ) -> "SigLIPProgressPredictor":
        checkpoint_dir = Path(checkpoint_dir)
        cfg_path = checkpoint_dir / "siglip_mlp_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)
        cfg = json.loads(cfg_path.read_text())

        target_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        vision_model = SiglipVisionModel.from_pretrained(
            str(model_path),
            trust_remote_code=False,
            local_files_only=True,
        )
        model = SigLIPMLPProgressModel(
            vision_model,
            mlp_hidden_size=int(cfg["mlp_hidden_size"]),
            mlp_layers=int(cfg["mlp_layers"]),
            mlp_dropout=float(cfg["mlp_dropout"]),
            output_activation=str(cfg.get("output_activation", "sigmoid")),
            v_min=float(cfg.get("v_min", 0.0)),
            v_max=float(cfg.get("v_max", 1.0)),
        )
        model.load_progress_head(checkpoint_dir)
        model.to(target_device).eval()

        processor = AutoImageProcessor.from_pretrained(
            str(model_path),
            trust_remote_code=False,
            local_files_only=True,
        )
        return cls(
            image_processor=processor,
            model=model,
            device=target_device,
            batch_size=max(1, int(batch_size)),
        )

    @torch.inference_mode()
    def predict_video(self, video: np.ndarray) -> np.ndarray:
        video = np.asarray(video)
        if video.ndim == 3:
            video = video[None]
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"expected RGB video with shape [T,H,W,3], got {video.shape}")
        if video.dtype != np.uint8:
            video = np.rint(video).clip(0, 255).astype(np.uint8)

        predictions: list[float] = []
        for start in range(0, video.shape[0], self.batch_size):
            frames = [
                Image.fromarray(frame)
                for frame in video[start : start + self.batch_size]
            ]
            inputs = self.image_processor(images=frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)
            values = self.model(pixel_values)
            predictions.extend(float(x) for x in values.detach().cpu().tolist())
        return np.asarray(predictions, dtype=np.float32)


__all__ = ["SigLIPMLPProgressModel", "SigLIPProgressPredictor"]
