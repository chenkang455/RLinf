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


class PickScoreRewardBackend(GenRewardBackend):
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        processor_path: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_path: str = "yuvalkirstain/PickScore_v1",
        scale: float = 26.0,
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device(device)
        self.scale = float(scale)
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def score(
        self,
        images: torch.Tensor | list[Any],
        prompts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        del metadatas
        pil_images = [Image.fromarray(image) for image in images_to_uint8_nhwc(images)]
        image_inputs = self.processor(
            images=pil_images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}
        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)
        rewards = (scores.diag() / self.scale).detach().cpu().float()
        return {"avg": rewards, "pickscore": rewards}
