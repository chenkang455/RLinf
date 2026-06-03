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
import pickle
from typing import Any

import torch
from PIL import Image

from .base import GenRewardBackend, images_to_uint8_nhwc


class GenevalRewardBackend(GenRewardBackend):
    def __init__(
        self,
        url: str,
        only_strict: bool = True,
        batch_size: int = 64,
        timeout: int = 120,
    ):
        import requests
        from requests.adapters import HTTPAdapter, Retry

        self.url = str(url)
        self.only_strict = bool(only_strict)
        self.batch_size = int(batch_size)
        self.timeout = int(timeout)
        self.session = requests.Session()
        retries = Retry(
            total=1000,
            backoff_factor=1,
            status_forcelist=[500],
            allowed_methods=False,
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def score(
        self,
        images: torch.Tensor | list[Any],
        prompts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        del prompts
        image_array = images_to_uint8_nhwc(images)
        scores: list[float] = []
        rewards: list[float] = []
        strict_rewards: list[float] = []
        group_rewards: dict[str, list[float]] = {}
        group_strict_rewards: dict[str, list[float]] = {}

        for start in range(0, len(image_array), self.batch_size):
            batch_images = image_array[start : start + self.batch_size]
            batch_metas = metadatas[start : start + self.batch_size]
            jpeg_images = []
            for image in batch_images:
                buffer = io.BytesIO()
                Image.fromarray(image).save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            payload = pickle.dumps(
                {
                    "images": jpeg_images,
                    "meta_datas": batch_metas,
                    "only_strict": self.only_strict,
                }
            )
            response = self.session.post(self.url, data=payload, timeout=self.timeout)
            response.raise_for_status()
            response_data = pickle.loads(response.content)
            scores.extend(response_data["scores"])
            rewards.extend(response_data["rewards"])
            strict_rewards.extend(response_data["strict_rewards"])
            for key, value in response_data.get("group_rewards", {}).items():
                group_rewards.setdefault(key, []).extend(value)
            for key, value in response_data.get("group_strict_rewards", {}).items():
                group_strict_rewards.setdefault(key, []).extend(value)

        output = {
            "avg": torch.as_tensor(scores, dtype=torch.float32),
            "accuracy": torch.as_tensor(rewards, dtype=torch.float32),
            "strict_accuracy": torch.as_tensor(strict_rewards, dtype=torch.float32),
        }
        for key, value in group_rewards.items():
            output[f"{key}_accuracy"] = torch.as_tensor(value, dtype=torch.float32)
        for key, value in group_strict_rewards.items():
            output[f"{key}_strict_accuracy"] = torch.as_tensor(
                value, dtype=torch.float32
            )
        return output
