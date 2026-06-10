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

from rlinf.envs.gen_reward.rewards import RewardBackend
from rlinf.envs.gen_reward.rewards.embodied.models.vidar_dim import IDM
from rlinf.envs.gen_reward.utils import (
    cfg_get,
    media_to_uint8_nhwc,
    prepare_video_pair,
)


class ActionSimilarityRewardBackend(RewardBackend):
    """IDM-based action similarity reward for image-conditioned videos."""

    def __init__(
        self,
        idm: torch.nn.Module,
        device: torch.device,
        temperature: float = 1.0,
    ):
        self.idm = idm
        self.device = device
        self.temperature = float(temperature)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @classmethod
    def from_config(cls, cfg: Any) -> "ActionSimilarityRewardBackend":
        checkpoint_path = Path(str(cfg.checkpoint_path))
        device = torch.device(
            str(cfg_get(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        )
        idm = IDM(model_name="mask", output_dim=14)
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        idm.load_state_dict(loaded["model_state_dict"])
        idm.to(device=device)
        idm.eval()
        return cls(
            idm=idm,
            device=device,
            temperature=float(cfg_get(cfg, "temperature", 1.0)),
        )

    def score(
        self,
        outputs: torch.Tensor | np.ndarray | list[Any],
        records: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        output_videos = media_to_uint8_nhwc(outputs)
        frame_rewards = []
        for output_video, record in zip(output_videos, records, strict=True):
            output_video, target_video = prepare_video_pair(output_video, record)
            pred_actions = self._predict_actions(output_video)
            target_actions = self._predict_actions(target_video)
            frame_rewards.append(
                self._action_similarity(pred_actions, target_actions)
                .detach()
                .cpu()
                .numpy()
            )

        rewards_tensor = torch.from_numpy(np.stack(frame_rewards)).float()
        return {"avg": rewards_tensor, "action_similarity": rewards_tensor}

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
            actions, _ = self.idm(frames, return_mask=False)
        return actions.float()

    def _action_similarity(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        pred_actions = self.idm.normalize(pred_actions)
        target_actions = self.idm.normalize(target_actions)
        mse = (pred_actions - target_actions).pow(2).mean(dim=-1)
        return torch.exp(-mse / self.temperature).clamp(0.0, 1.0)


REWARD_CLS = ActionSimilarityRewardBackend


__all__ = ["ActionSimilarityRewardBackend", "REWARD_CLS"]
