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

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.envs.gen_reward.rewards import FRAME_LEVEL, VIDEO_LEVEL, RewardBackend
from rlinf.envs.gen_reward.rewards.embodied.models.progress_model import SigLIPProgressPredictor
from rlinf.envs.gen_reward.utils import cfg_get, cfg_require, media_to_uint8_nhwc


class ProgressRewardBackend(RewardBackend):
    """SigLIP progress-model reward for RGB videos."""

    supported_reward_levels = (FRAME_LEVEL, VIDEO_LEVEL)
    support_type = FRAME_LEVEL

    def __init__(
        self,
        progress_model: SigLIPProgressPredictor,
    ) -> None:
        self.progress_model = progress_model

    @classmethod
    def from_config(cls, cfg: Any) -> "ProgressRewardBackend":
        model_path = Path(str(cfg_require(cfg, "model_path")))
        checkpoint_path = Path(str(cfg_require(cfg, "checkpoint_path")))
        progress_model = SigLIPProgressPredictor.from_pretrained(
            model_path=model_path,
            checkpoint_dir=checkpoint_path,
            device=cfg_get(cfg, "device", None),
            batch_size=int(cfg_get(cfg, "batch_size", 32)),
        )
        return cls(progress_model=progress_model)

    def score(
        self,
        outputs: torch.Tensor | np.ndarray | list[Any],
        records: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        videos = media_to_uint8_nhwc(outputs)
        progress_rows = []
        avg_rows = []
        for video in videos:
            progress = self.progress_model.predict_video(video)
            progress_tensor = torch.from_numpy(progress).float()
            progress_rows.append(progress_tensor)
            avg_rows.append(progress_tensor[-1:].clone())
        progress_rewards = torch.stack(progress_rows, dim=0)
        avg_rewards = torch.stack(avg_rows, dim=0)
        return {"avg": avg_rewards, "progress": progress_rewards}


REWARD_CLS = ProgressRewardBackend


__all__ = ["ProgressRewardBackend", "REWARD_CLS"]
