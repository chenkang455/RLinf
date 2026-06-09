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

import numpy as np

EnvRecord = dict[str, Any]
EnvRecords = list[EnvRecord]
EnvObs = dict[str, Any]
EnvBatch = tuple[EnvObs, EnvRecords]


class TextDataset:
    """Base interface for text-conditioned generation reward datasets.

    Dataset records use a small canonical schema:

    Required:
        task_description: str

    Optional:
        task_name: str
        episode_id: str
        dataset_index: int
        metadata: dict[str, Any]

    `build_env_batch()` maps records to batched env observations with plural
    keys, for example `task_descriptions`.
    """

    records: EnvRecords

    def __init__(self, records: EnvRecords):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EnvRecord:
        """Return one canonical dataset record."""
        return self.records[index]

    def build_env_batch(
        self,
        records: EnvRecords,
    ) -> EnvBatch:
        """Return `(env_obs, env_records)` from sampled records."""
        env_obs = {
            "task_descriptions": [record["task_description"] for record in records]
        }
        return env_obs, records


class ImageConditionedDataset(TextDataset):
    """Base interface for text-plus-image-conditioned reward datasets.

    Image-conditioned records extend the text schema:

    Required:
        main_image: np.ndarray with shape [height, width, channels]

    Optional:
        main_image_path: str | None
        future_video_path: str | None
        future_video: np.ndarray with shape [time, height, width, channels]

    `build_env_batch()` maps those to `main_images` and, when present,
    `future_video_paths` / `future_videos`.
    """

    def __getitem__(self, index: int) -> EnvRecord:
        """Return one record with a loaded `main_image`."""
        record = super().__getitem__(index)
        if "main_image" not in record:
            raise KeyError(
                "ImageConditionedDataset records require `main_image`; "
                "override `__getitem__` if images should be loaded lazily."
            )
        return record

    def build_env_batch(
        self,
        records: EnvRecords,
    ) -> EnvBatch:
        """Return `(env_obs, env_records)` with image-conditioned observations."""
        env_obs, env_records = super().build_env_batch(records)
        env_obs["main_images"] = np.stack(
            [record["main_image"] for record in records],
            axis=0,
        )
        has_future_video_path = any(
            "future_video_path" in record for record in records
        )
        if has_future_video_path:
            future_video_paths = [
                record.get("future_video_path") for record in records
            ]
            env_obs["future_video_paths"] = future_video_paths
        has_future_video = records and all(
            record.get("future_video") is not None for record in records
        )
        if has_future_video:
            env_obs["future_videos"] = np.stack(
                [record["future_video"] for record in records],
                axis=0,
            )
        return env_obs, env_records


__all__ = [
    "EnvBatch",
    "EnvObs",
    "EnvRecord",
    "EnvRecords",
    "ImageConditionedDataset",
    "TextDataset",
]
