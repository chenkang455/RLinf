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

import re
from typing import Any

import numpy as np
import torch
from PIL import Image

from .base import GenRewardBackend, images_to_uint8_nhwc


class OCRRewardBackend(GenRewardBackend):
    def __init__(self, use_gpu: bool = False, lang: str = "en"):
        try:
            from Levenshtein import distance
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise ImportError(
                "OCR reward requires `paddleocr` and `python-Levenshtein`. "
                "Install those packages in the RLinf environment before using "
                "reward.type=ocr."
            ) from exc

        self.distance = distance
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang=str(lang),
            use_gpu=bool(use_gpu),
            show_log=False,
        )

    def score(
        self,
        images: torch.Tensor | np.ndarray | list[Any],
        prompts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        del metadatas
        image_array = images_to_uint8_nhwc(images)
        rewards = []
        for image, prompt in zip(image_array, prompts, strict=True):
            target = _extract_quoted_text(prompt)
            rewards.append(self._score_image(image, target))
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        return {"avg": rewards_tensor, "ocr": rewards_tensor, "accuracy": rewards_tensor}

    def _score_image(self, image: np.ndarray | Image.Image, target: str) -> float:
        if isinstance(image, Image.Image):
            image = np.asarray(image.convert("RGB"), dtype=np.uint8)
        target = _normalize_text(target)
        if not target:
            return 0.0
        try:
            result = self.ocr.ocr(image, cls=False)
            recognized = "".join(
                res[1][0] if res[1][1] > 0 else "" for res in (result[0] or [])
            )
            recognized = _normalize_text(recognized)
            dist = 0 if target in recognized else self.distance(recognized, target)
            dist = min(dist, len(target))
        except Exception as exc:
            print(f"OCR reward failed: {exc}")
            dist = len(target)
        return float(1.0 - dist / len(target))


def _extract_quoted_text(prompt: str) -> str:
    match = re.search(r'"([^"]+)"', str(prompt))
    if match:
        return match.group(1)
    return str(prompt)


def _normalize_text(text: str) -> str:
    return str(text).replace(" ", "").lower()
