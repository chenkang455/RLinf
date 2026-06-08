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

from importlib import import_module
from typing import Any, Protocol

import numpy as np
import torch

from rlinf.envs.gen_reward.utils import (
    cfg_require,
    metadata_from_record,
    normalize_type,
    obs_from_records,
)


class GenRewardDataset(Protocol):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> dict[str, Any]:
        ...


class GenRewardBackend(Protocol):
    def score(
        self,
        outputs: torch.Tensor | np.ndarray | list[Any],
        records: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        ...


def build_reward_dataset(dataset_cfg: Any) -> GenRewardDataset:
    dataset_type = normalize_type(cfg_require(dataset_cfg, "type"))
    module = import_module(f"rlinf.envs.gen_reward.datasets.{dataset_type}")
    return module.build_reward_dataset(dataset_cfg)


def build_reward_backend(cfg: Any) -> GenRewardBackend:
    reward_type = normalize_type(cfg_require(cfg, "type"))
    module = import_module(f"rlinf.envs.gen_reward.rewards.{reward_type}")
    return module.build_reward_backend(cfg)


from rlinf.envs.gen_reward.gen_reward_env import GenRewardEnv

__all__ = [
    "GenRewardBackend",
    "GenRewardDataset",
    "GenRewardEnv",
    "build_reward_backend",
    "build_reward_dataset",
    "metadata_from_record",
    "obs_from_records",
]
