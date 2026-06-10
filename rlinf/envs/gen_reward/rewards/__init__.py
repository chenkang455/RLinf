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

from typing import Any, Protocol

import numpy as np
import torch

from rlinf.envs.gen_reward.utils import cfg_get, cfg_require, normalize_type

FRAME_LEVEL = "frame_level"
VIDEO_LEVEL = "video_level"
REWARD_SUPPORT_TYPES = (FRAME_LEVEL, VIDEO_LEVEL)

RewardOutputs = torch.Tensor | np.ndarray | list[Any]
RewardRecords = list[dict[str, Any]]
RewardScores = dict[str, torch.Tensor]


class RewardBackend(Protocol):
    """Interface for generated-output reward backends.

    Args:
        outputs: Generated images/videos from the rollout side.
        records: Env records returned by `dataset.build_grouped_env_batch`.

    Returns:
        Score dict containing the configured `reward.key`, usually `avg`.
        Score tensors should have batch dimension first. `frame_level` rewards
        use shape [B, T]; `video_level` rewards use shape [B, 1].
    """

    supported_reward_levels: tuple[str, ...]
    support_type: str

    @classmethod
    def from_config(cls, cfg: Any) -> "RewardBackend":
        ...

    def score(
        self,
        outputs: RewardOutputs,
        records: RewardRecords,
    ) -> RewardScores:
        ...


def normalize_reward_support_type(value: Any) -> str:
    support_type = str(value).lower().replace("-", "_")
    if support_type not in REWARD_SUPPORT_TYPES:
        raise ValueError(f"Unknown reward support_type: {support_type}")
    return support_type


def validate_reward_support(backend: RewardBackend, cfg: Any) -> RewardBackend:
    supported = tuple(getattr(backend, "supported_reward_levels", REWARD_SUPPORT_TYPES))
    support_type = normalize_reward_support_type(
        cfg_get(cfg, "support_type", getattr(backend, "support_type", supported[0]))
    )
    if support_type not in supported:
        raise ValueError(
            f"{backend.__class__.__name__} supports support_type {supported}, "
            f"got {support_type}."
        )
    backend.support_type = support_type
    return backend


def frame_rewards_to_latent_rewards(
    frame_rewards: np.ndarray | list[float] | list[list[float]],
    frame_interval: int,
) -> np.ndarray:
    frame_rewards = np.asarray(frame_rewards, dtype=np.float32)
    if frame_interval <= 0:
        return frame_rewards
    chunks = [frame_rewards[..., :1]] + [
        frame_rewards[..., i : i + frame_interval].mean(axis=-1, keepdims=True)
        for i in range(1, frame_rewards.shape[-1], frame_interval)
    ]
    return np.concatenate(chunks, axis=-1).astype(np.float32)


class MultiRewardBackend:
    def __init__(self, reward_backends: list[tuple[str, float, RewardBackend]]):
        self.reward_backends = reward_backends

    @classmethod
    def from_config(cls, cfg: Any, build_single_reward_backend) -> "MultiRewardBackend":
        reward_backends = []
        for reward_cfg in cfg_require(cfg, "rewards"):
            reward_model = normalize_type(cfg_require(reward_cfg, "model"))
            name = str(cfg_get(reward_cfg, "name", reward_model.split(".")[-1]))
            weight = float(cfg_get(reward_cfg, "weight", 1.0))
            reward_backends.append(
                (name, weight, build_single_reward_backend(reward_cfg))
            )
        return cls(reward_backends)

    def score(
        self,
        outputs: RewardOutputs,
        records: RewardRecords,
    ) -> RewardScores:
        scores = {}
        weighted_rewards = []
        total_weight = 0.0
        for name, weight, backend in self.reward_backends:
            backend_scores = backend.score(outputs, records)
            weighted_rewards.append(backend_scores["avg"].float() * weight)
            total_weight += weight
            for key, value in backend_scores.items():
                scores[f"{name}.{key}"] = value
        scores["avg"] = sum(weighted_rewards) / total_weight
        return scores


__all__ = [
    "FRAME_LEVEL",
    "VIDEO_LEVEL",
    "MultiRewardBackend",
    "RewardBackend",
    "RewardOutputs",
    "RewardRecords",
    "RewardScores",
    "frame_rewards_to_latent_rewards",
    "normalize_reward_support_type",
    "validate_reward_support",
]
