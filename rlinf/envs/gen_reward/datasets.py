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

import json
from pathlib import Path
from typing import Any

from .rewards.base import cfg_get


class PromptDataset:
    """Prompt records used by one-step generation reward environments."""

    def __init__(self, records: list[dict[str, Any]]):
        if not records:
            raise ValueError("PromptDataset requires at least one record.")
        self.records = records

    @classmethod
    def from_config(cls, dataset_cfg: Any) -> "PromptDataset":
        prompts = cfg_get(dataset_cfg, "prompts", None)
        if prompts is not None:
            return cls([{"prompt": str(prompt)} for prompt in list(prompts)])

        path = cfg_get(dataset_cfg, "path", None)
        if path is None:
            raise ValueError("PromptDataset requires dataset.path or dataset.prompts.")
        dataset_path = _resolve_dataset_path(Path(str(path)), dataset_cfg)
        if dataset_path.suffix == ".txt":
            return cls(_load_txt_prompts(dataset_path))
        return cls(_load_jsonl_records(dataset_path))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _resolve_dataset_path(dataset_path: Path, dataset_cfg: Any) -> Path:
    if not dataset_path.is_dir():
        if not dataset_path.exists():
            raise FileNotFoundError(f"Prompt dataset not found: {dataset_path}")
        return dataset_path

    split = str(cfg_get(dataset_cfg, "split", "train"))
    candidates = [
        dataset_path / f"{split}_metadata.jsonl",
        dataset_path / f"{split}.jsonl",
        dataset_path / f"{split}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Prompt dataset not found. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _load_txt_prompts(dataset_path: Path) -> list[dict[str, Any]]:
    records = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                records.append({"prompt": prompt})
    if not records:
        raise ValueError(f"Prompt dataset is empty: {dataset_path}")
    return records


def _load_jsonl_records(dataset_path: Path) -> list[dict[str, Any]]:
    records = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"Prompt dataset is empty: {dataset_path}")
    return records
