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

import numpy as np
import cv2

from rlinf.envs.gen_reward.datasets.embodied.lerobot_dataset import (
    LeRobotImageConditionedDataset,
)


class RobotwinDataset(LeRobotImageConditionedDataset):
    """RoboTwin dataset stored in standard LeRobot format."""

    default_image_keys = (
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    )

    def compose_views(self, views: list[np.ndarray]) -> np.ndarray:
        if len(views) < 3:
            return views[0]
        top, left, right = views[:3]
        height, width = top.shape[:2]
        wrist_size = (width // 2, height // 2)
        left = cv2.resize(left, wrist_size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, wrist_size, interpolation=cv2.INTER_AREA)
        bottom = np.concatenate([left, right], axis=1)
        return np.concatenate([top, bottom], axis=0)


DATASET_CLS = RobotwinDataset


__all__ = ["DATASET_CLS", "RobotwinDataset"]
