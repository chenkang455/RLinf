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

"""Fake DreamZero components for fast policy-level debugging.

The debug path keeps ``DreamZeroPolicy`` in the call stack and only replaces
the heavyweight DreamZero backbone/action head with these lightweight
components.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature


class FakeDreamZeroDataTransforms:
    """Minimal transform contract used by ``DreamZeroPolicy`` debug runs."""

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        images = obs["video.image"]
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        images = images.permute(0, 1, 4, 2, 3).contiguous().float() / 255.0

        state = obs.get("state.state")
        if state is None:
            state = torch.zeros(images.shape[0], 1, 8, dtype=torch.float32)
        elif isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        else:
            state = state.float()

        return {
            "images": images,
            "state": state,
            "embodiment_id": torch.zeros(images.shape[0], dtype=torch.long),
        }

    def unapply(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"action.actions": data["action"]}

    def eval(self):
        return self

    def set_metadata(self, metadata):
        return None


class FakeDreamZeroBackbone(nn.Module):
    """Tiny backbone that satisfies the third-party VLA interface."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def prepare_input(self, inputs: dict[str, Any]) -> BatchFeature:
        return BatchFeature(data=inputs)

    def forward(self, inputs: BatchFeature) -> BatchFeature:
        images = inputs.get("images")
        if images is not None and torch.is_tensor(images):
            batch_size = images.shape[0]
            device = images.device
            dtype = images.dtype
        else:
            state = inputs.get("state")
            batch_size = state.shape[0]
            device = state.device
            dtype = state.dtype
        probe = torch.zeros(
            batch_size, self.proj.in_features, device=device, dtype=dtype
        )
        return BatchFeature(data={"backbone_features": self.proj(probe)[:, None, :]})


class FakeDreamZeroActionModel(nn.Module):
    """Small trainable action model used by the fake action head."""

    def __init__(self, action_dim: int = 7, hidden: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden, action_dim)

    def forward(
        self,
        noisy_latents,
        *,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, horizon, action_dim = action.shape
        probe = torch.zeros(
            batch_size,
            self.proj.in_features,
            device=action.device,
            dtype=action.dtype,
        )
        bias = self.proj(probe).view(batch_size, 1, action_dim)
        action_noise_pred = action + bias.expand(-1, horizon, -1)
        video_noise_pred = torch.zeros_like(noisy_latents)
        return video_noise_pred, action_noise_pred


class FakeDreamZeroActionHead(nn.Module):
    """Action-head replacement that avoids text/image encoder and DiT loading."""

    def __init__(
        self,
        action_dim: int = 7,
        action_horizon: int = 16,
        hidden: int = 64,
        dtype: str | torch.dtype = torch.float32,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self._dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.model = FakeDreamZeroActionModel(action_dim=action_dim, hidden=hidden)
        self.trt_engine = None

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def prepare_input(self, inputs: dict[str, Any]) -> BatchFeature:
        return BatchFeature(data=inputs)

    def lazy_joint_video_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        latent_video=None,
    ) -> BatchFeature:
        images = action_input.get("images")
        state = action_input.get("state")
        if images is not None and torch.is_tensor(images):
            batch_size = images.shape[0]
            device = images.device
            dtype = images.dtype
        else:
            batch_size = state.shape[0]
            device = state.device
            dtype = state.dtype
        features = backbone_output["backbone_features"][:, 0, :]
        action = self.model.proj(features).to(dtype=dtype)
        action = action[:, None, :].expand(batch_size, self.action_horizon, -1)
        return BatchFeature(data={"action_pred": torch.tanh(action).to(device)})

    def prepare_policy_inputs(
        self,
        data: BatchFeature,
        mode: str = "inference",
    ) -> BatchFeature:
        state = data.get("state")
        if state is None:
            first_tensor = next(v for v in data.values() if torch.is_tensor(v))
            batch_size = first_tensor.shape[0]
            device = first_tensor.device
            dtype = first_tensor.dtype
            state = torch.zeros(batch_size, 1, 8, device=device, dtype=dtype)
        else:
            batch_size = state.shape[0]
            device = state.device
            dtype = state.dtype
        return BatchFeature(
            data={
                "image_latents": torch.zeros(
                    batch_size, 4, 1, 8, 8, device=device, dtype=dtype
                ),
                "clip_feas": torch.zeros(batch_size, 1, 8, device=device, dtype=dtype),
                "ys": torch.zeros(batch_size, 1, 8, device=device, dtype=dtype),
                "prompt_embs": torch.zeros(
                    batch_size, 1, 8, device=device, dtype=dtype
                ),
                "state_features": state,
                "embodiment_id": data.get(
                    "embodiment_id",
                    torch.zeros(batch_size, device=device, dtype=torch.long),
                ),
            }
        )

    def inject_lora_after_loading(self):
        return
