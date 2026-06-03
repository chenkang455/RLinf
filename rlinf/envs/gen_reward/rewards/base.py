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

from typing import Any

import numpy as np
import torch
from PIL import Image

class GenRewardBackend:
    def score(
        self,
        images: torch.Tensor | np.ndarray | list[Any],
        prompts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def images_to_uint8_nhwc(images: torch.Tensor | np.ndarray | list[Any]) -> np.ndarray:
    if isinstance(images, list):
        arrays = []
        for image in images:
            if isinstance(image, Image.Image):
                arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
            else:
                arrays.append(np.asarray(image))
        images = np.stack(arrays, axis=0)
    elif isinstance(images, torch.Tensor):
        images = images.detach().cpu().float().numpy()
    else:
        images = np.asarray(images)

    if images.ndim == 3:
        images = images[None]
    if images.ndim != 4:
        raise ValueError(f"Expected image batch with 4 dims, got shape {images.shape}.")
    if images.shape[1] in (1, 3, 4) and images.shape[-1] not in (1, 3, 4):
        images = np.transpose(images, (0, 2, 3, 1))
    if np.issubdtype(images.dtype, np.floating):
        images = np.clip(images, 0.0, 1.0) * 255.0
    images = np.rint(images).clip(0, 255).astype(np.uint8)
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    if images.shape[-1] == 4:
        images = images[..., :3]
    return images

