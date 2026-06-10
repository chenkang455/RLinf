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

from rlinf.envs.gen_reward.rewards import frame_rewards_to_latent_rewards
from rlinf.envs.gen_reward.rewards.generation.ocr import OCRRewardBackend
from rlinf.envs.gen_reward.utils import (
    cfg_get,
    extract_quoted_text,
    media_to_uint8_nhwc,
)


class VideoOCRRewardBackend(OCRRewardBackend):
    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "en",
        frame_interval: int = -1,
    ):
        super().__init__(use_gpu=use_gpu, lang=lang)
        self.frame_interval = int(frame_interval)

    @classmethod
    def from_config(cls, cfg: Any) -> "VideoOCRRewardBackend":
        return cls(
            use_gpu=bool(cfg_get(cfg, "use_gpu", False)),
            lang=str(cfg_get(cfg, "lang", "en")),
            frame_interval=int(cfg_get(cfg, "frame_interval", -1)),
        )

    def score(
        self,
        outputs: torch.Tensor | np.ndarray | list[Any],
        records: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        task_descriptions = [record["task_description"] for record in records]
        media_array = media_to_uint8_nhwc(outputs)
        frame_rewards = []
        for media, task_description in zip(
            media_array,
            task_descriptions,
            strict=True,
        ):
            target = extract_quoted_text(task_description)
            frame_rewards.append([self._score_image(frame, target) for frame in media])

        rewards = frame_rewards_to_latent_rewards(frame_rewards, self.frame_interval)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        return {"avg": rewards_tensor, "video_ocr": rewards_tensor}


REWARD_CLS = VideoOCRRewardBackend


__all__ = ["REWARD_CLS", "VideoOCRRewardBackend"]
