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

from rlinf.envs.gen_reward.utils import (
    cfg_get,
    load_jsonl_records,
    load_txt_prompts,
    resolve_dataset_path,
)


class TextDataset:
    """Text records used by generation reward environments."""

    def __init__(self, records: list[dict[str, Any]]):
        if not records:
            raise ValueError("TextDataset requires at least one record.")
        self.records = records

    @classmethod
    def from_config(cls, dataset_cfg: Any) -> "TextDataset":
        prompts = cfg_get(dataset_cfg, "prompts", None)
        if prompts is not None:
            return cls([{"prompt": str(prompt)} for prompt in list(prompts)])

        path = cfg_get(dataset_cfg, "path", None)
        if path is None:
            raise ValueError("TextDataset requires dataset.path or dataset.prompts.")
        dataset_path = resolve_dataset_path(str(path), dataset_cfg)
        if dataset_path.suffix == ".txt":
            return cls(load_txt_prompts(dataset_path))
        return cls(load_jsonl_records(dataset_path))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def build_reward_dataset(dataset_cfg: Any) -> TextDataset:
    return TextDataset.from_config(dataset_cfg)
