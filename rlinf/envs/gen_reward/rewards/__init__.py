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
        Score tensors should have batch dimension first. A 1-D tensor means
        one reward per environment; a 2-D tensor means chunk rewards.
    """

    @classmethod
    def from_config(cls, cfg: Any) -> "RewardBackend":
        ...

    def score(
        self,
        outputs: RewardOutputs,
        records: RewardRecords,
    ) -> RewardScores:
        ...


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
            reward_backends.append((name, weight, build_single_reward_backend(reward_cfg)))
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
    "MultiRewardBackend",
    "RewardBackend",
    "RewardOutputs",
    "RewardRecords",
    "RewardScores",
]
