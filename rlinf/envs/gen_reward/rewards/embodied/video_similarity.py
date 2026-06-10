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
    record_gt_video,
    resize_video,
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

        rewards = []
        for output_video, record in zip(output_videos, records, strict=True):
            target_video = record_gt_video(record)
            rewards.append(self._video_similarity(output_video, target_video))

        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32)
        return {"avg": rewards_tensor, "video_similarity": rewards_tensor}

    def _video_similarity(
        self,
        output_video: np.ndarray,
        target_video: np.ndarray,
    ) -> float:
        if output_video.shape[0] != target_video.shape[0]:
            raise ValueError(
                "VideoSimilarityRewardBackend expects output and target videos "
                f"to have the same number of frames, got output={output_video.shape[0]} "
                f"target={target_video.shape[0]}."
            )
        if output_video.shape[0] == 0:
            return 0.0
        if output_video.shape[1:3] != target_video.shape[1:3]:
            target_video = resize_video(target_video, *output_video.shape[1:3])
        mse = np.mean(
            (output_video.astype(np.float32) - target_video.astype(np.float32)) ** 2
        )
        return float(np.clip(1.0 - mse / (255.0**2), 0.0, 1.0))


REWARD_CLS = VideoSimilarityRewardBackend


__all__ = ["REWARD_CLS", "VideoSimilarityRewardBackend"]
