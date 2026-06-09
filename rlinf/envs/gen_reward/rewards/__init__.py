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

RewardOutputs = torch.Tensor | np.ndarray | list[Any]
RewardRecords = list[dict[str, Any]]
RewardScores = dict[str, torch.Tensor]


class RewardBackend(Protocol):
    """Interface for generated-output reward backends.

    Args:
        outputs: Generated images/videos from the rollout side.
        records: Reward records returned by `dataset.build_env_batch`.

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


__all__ = [
    "RewardBackend",
    "RewardOutputs",
    "RewardRecords",
    "RewardScores",
]
