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

import torch
from PIL import Image

from .base import GenRewardBackend, images_to_uint8_nhwc


class ImageRewardBackend(GenRewardBackend):
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        try:
            import ImageReward as RM
        except ImportError as exc:
            raise ImportError(
                "ImageReward backend requires the `ImageReward` package."
            ) from exc

        self.model = RM.load("ImageReward-v1.0", device=device).eval().to(dtype=dtype)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def score(
        self,
        images: torch.Tensor | list[Any],
        prompts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        del metadatas
        pil_images = [Image.fromarray(image) for image in images_to_uint8_nhwc(images)]
        rewards = []
        for prompt, image in zip(prompts, pil_images, strict=True):
            _, reward = self.model.inference_rank(prompt, [image])
            rewards.append(float(reward))
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        return {"avg": rewards_tensor, "imagereward": rewards_tensor}
