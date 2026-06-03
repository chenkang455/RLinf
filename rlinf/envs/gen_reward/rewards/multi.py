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

import torch

from .base import GenRewardBackend


class MultiRewardBackend(GenRewardBackend):
    def __init__(self, weighted_backends: dict[str, tuple[float, GenRewardBackend]]):
        self.weighted_backends = weighted_backends

    def score(
        self,
        images: torch.Tensor | list[Any],
        prompts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        total = None
        details: dict[str, torch.Tensor] = {}
        for name, (weight, backend) in self.weighted_backends.items():
            output = backend.score(images, prompts, metadatas)
            reward = output.get("avg", output.get(name))
            if reward is None:
                raise KeyError(
                    f"Reward backend {name!r} did not return `avg` or {name!r}. "
                    f"Available keys: {sorted(output)}"
                )
            reward = reward.float()
            details[name] = reward
            for key, value in output.items():
                if key == "avg":
                    continue
                details.setdefault(key, value.float())
            weighted = float(weight) * reward
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("MultiRewardBackend requires at least one reward backend.")
        details["avg"] = total.float()
        return details
