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
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rlinf.envs.gen_reward.datasets import EnvRecord, ImageConditionedDataset
from rlinf.envs.gen_reward.utils import media_to_uint8_nhwc


class LeRobotImageConditionedDataset(ImageConditionedDataset):
    """LeRobot -> text-image-to-video adapter."""

    default_image_keys = ("observation.images.front",)
    default_task_key = "task"

    def __init__(
        self,
        dataset: Any,
    ):
        self.dataset = dataset

    @classmethod
    def from_config(cls, cfg: Any) -> "LeRobotImageConditionedDataset":
        future_times = [float(timestamp) for timestamp in cfg.future_times]
        dataset = LeRobotDataset(
            str(cfg.repo_id),
            root=cfg.root,
            episodes=getattr(cfg, "episodes", None),
            delta_timestamps={key: future_times for key in cls.default_image_keys},
            video_backend=getattr(cfg, "video_backend", "pyav"),
        )
        return cls(dataset=dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> EnvRecord:
        sample = self.dataset[index]
        videos = [media_to_uint8_nhwc(sample[key]) for key in self.default_image_keys]
        video = videos[0] if len(videos) == 1 else self.compose_videos(videos)
        task = sample.get(self.default_task_key, "")
        if isinstance(task, (list, tuple)):
            task = task[0] if task else ""
        return {
            "task_description": str(task),
            "main_image": video[0],
            "future_video": video[1:] if video.shape[0] > 1 else None,
        }

    def compose_videos(self, videos: list[np.ndarray]) -> np.ndarray:
        num_frames = min(video.shape[0] for video in videos)
        return np.stack(
            [
                self.compose_views([video[idx] for video in videos])
                for idx in range(num_frames)
            ],
            axis=0,
        )

    def compose_views(self, views: list[np.ndarray]) -> np.ndarray:
        return views[0]


DATASET_CLS = LeRobotImageConditionedDataset


__all__ = ["DATASET_CLS", "LeRobotImageConditionedDataset"]
