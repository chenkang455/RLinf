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
from rlinf.envs.gen_reward.rewards import RewardBackend
from rlinf.envs.gen_reward.utils import (
    media_to_uint8_nhwc,
    prepare_video_pair,
)


class VideoSimilarityRewardBackend(RewardBackend):
    """Reference-video similarity reward for video generation datasets."""

    @classmethod
    def from_config(cls, cfg: Any) -> "VideoSimilarityRewardBackend":
        return cls()

    def score(
        self,
        outputs: torch.Tensor | np.ndarray | list[Any],
        records: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        output_videos = media_to_uint8_nhwc(outputs)

        frame_rewards = []
        for output_video, record in zip(output_videos, records, strict=True):
            output_video, target_video = prepare_video_pair(output_video, record)
            frame_rewards.append(self._frame_similarities(output_video, target_video))

        frame_rewards_tensor = torch.from_numpy(np.stack(frame_rewards)).float()
        return {"avg": frame_rewards_tensor, "video_similarity": frame_rewards_tensor}

    def _frame_similarities(
        self,
        output_video: np.ndarray,
        target_video: np.ndarray,
    ) -> np.ndarray:
        mse = np.mean(
            (output_video.astype(np.float32) - target_video.astype(np.float32)) ** 2,
            axis=(1, 2, 3),
        )
        return np.clip(1.0 - mse / (255.0**2), 0.0, 1.0).astype(np.float32)


REWARD_CLS = VideoSimilarityRewardBackend


__all__ = ["REWARD_CLS", "VideoSimilarityRewardBackend"]
