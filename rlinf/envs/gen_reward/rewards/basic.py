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

import io
from typing import Any

import numpy as np
import torch
from PIL import Image

from .base import GenRewardBackend, images_to_uint8_nhwc


class MockGenRewardBackend(GenRewardBackend):
    def score(
        self,
        images: torch.Tensor | np.ndarray | list[Any],
        prompts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        del prompts, metadatas
        image_array = images_to_uint8_nhwc(images).astype(np.float32) / 255.0
        rewards = image_array.mean(axis=(1, 2, 3)).astype(np.float32)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        return {"avg": rewards_tensor, "accuracy": rewards_tensor}


class JPEGCompressibilityRewardBackend(GenRewardBackend):
    def __init__(self, invert: bool = True, quality: int = 95, scale: float = 500.0):
        self.invert = bool(invert)
        self.quality = int(quality)
        self.scale = float(scale)

    def score(
        self,
        images: torch.Tensor | np.ndarray | list[Any],
        prompts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        del prompts, metadatas
        image_array = images_to_uint8_nhwc(images)
        sizes = []
        for image in image_array:
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="JPEG", quality=self.quality)
            sizes.append(buffer.tell() / 1000.0)
        rewards = np.asarray(sizes, dtype=np.float32)
        if self.invert:
            rewards = -rewards / self.scale
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        return {"avg": rewards_tensor, "jpeg_compressibility": rewards_tensor}
