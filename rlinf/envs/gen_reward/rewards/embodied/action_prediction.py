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
import torch.nn.functional as F

from rlinf.envs.gen_reward.rewards import FRAME_LEVEL, VideoRewardBackendBase
from rlinf.envs.gen_reward.rewards.embodied.models.vidar_dim import IDM
from rlinf.envs.gen_reward.utils import cfg_get


class ActionPredictionRewardBackend(VideoRewardBackendBase):
    """Shared predicted-action base for embodied video rewards."""

    supported_reward_levels = (FRAME_LEVEL,)
    reward_type = FRAME_LEVEL

    def __init__(
        self,
        action_model: torch.nn.Module,
        device: torch.device,
        temperature: float = 1.0,
    ):
        self.action_model = action_model
        self.device = device
        self.temperature = float(temperature)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @classmethod
    def _from_config(cls, cfg: Any) -> "ActionPredictionRewardBackend":
        checkpoint_path = Path(str(cfg.checkpoint_path))
        device = torch.device(
            str(cfg_get(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        )
        action_model = IDM(model_name="mask", output_dim=14)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        action_model.load_state_dict(loaded["model_state_dict"])
        action_model.to(device=device)
        action_model.eval()
        return cls(
            action_model=action_model,
            device=device,
            temperature=float(cfg_get(cfg, "temperature", 1.0)),
        )

    def _predict_actions(self, video: np.ndarray) -> torch.Tensor:
        frames = (
            torch.from_numpy(video).to(device=self.device, dtype=torch.float32) / 255.0
        )
        frames = frames.permute(0, 3, 1, 2)
        frames = F.interpolate(
            frames,
            size=(518, 518),
            mode="bilinear",
            align_corners=False,
        )
        frames = (frames - self.mean) / self.std
        with torch.inference_mode():
            actions, _ = self.action_model(frames, return_mask=False)
        return actions.float()

    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.action_model.normalize(actions)


__all__ = ["ActionPredictionRewardBackend"]
