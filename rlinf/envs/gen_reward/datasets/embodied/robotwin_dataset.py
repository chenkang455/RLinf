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

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.envs.gen_reward.utils import (
    cfg_get,
    cfg_require,
    first_nonempty_text,
    load_jsonl_records,
    load_rgb_image,
    parse_image_size,
    resolve_data_path,
)


class RobotwinDataset:
    """RoboTwin records with text and image-conditioning context."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        image_size: tuple[int, int] | None,
    ):
        if not records:
            raise ValueError("RobotwinDataset requires at least one record.")
        self.records = records
        self.image_size = image_size

    @classmethod
    def from_config(cls, dataset_cfg: Any) -> "RobotwinDataset":
        data_root_value = cfg_get(dataset_cfg, "data_root", None)
        if data_root_value is None:
            data_root_value = cfg_require(dataset_cfg, "path")
        data_root = Path(str(data_root_value)).expanduser()
        metadata_path = cfg_get(dataset_cfg, "metadata_path", None)
        prompt_key = str(cfg_get(dataset_cfg, "prompt_key", "caption"))
        task_config = cfg_get(dataset_cfg, "task_config", None)
        split = cfg_get(dataset_cfg, "split", None)
        image_size = parse_image_size(cfg_get(dataset_cfg, "image_size", None))

        if metadata_path is not None:
            records = _load_robotwin_metadata(
                data_root=data_root,
                metadata_path=Path(str(metadata_path)),
                prompt_key=prompt_key,
                split=None if split is None else str(split),
            )
        else:
            records = _discover_robotwin_records(
                data_root=data_root,
                task_config=None if task_config is None else str(task_config),
            )
        return cls(records, image_size=image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        image_path = record.get("image_path")
        if image_path:
            image = load_rgb_image(Path(str(image_path)), self.image_size)
        else:
            image = _load_first_video_frame(
                Path(str(record["video_path"])),
                self.image_size,
            )
        record["main_image"] = image
        record.setdefault(
            "condition_image_path",
            str(image_path or record["video_path"]),
        )
        return record


def build_reward_dataset(dataset_cfg: Any) -> RobotwinDataset:
    return RobotwinDataset.from_config(dataset_cfg)


def _load_robotwin_metadata(
    *,
    data_root: Path,
    metadata_path: Path,
    prompt_key: str,
    split: str | None,
) -> list[dict[str, Any]]:
    metadata_file = (
        metadata_path if metadata_path.is_absolute() else data_root / metadata_path
    )
    if not metadata_file.exists():
        raise FileNotFoundError(f"RoboTwin metadata file not found: {metadata_file}")

    if metadata_file.suffix == ".jsonl":
        raw_records = load_jsonl_records(metadata_file)
    else:
        with metadata_file.open("r", newline="", encoding="utf-8") as f:
            raw_records = list(csv.DictReader(f))

    records = []
    for idx, raw in enumerate(raw_records):
        if split is not None and raw.get("split") not in (None, "", split):
            continue
        record = _normalize_robotwin_record(raw, data_root, prompt_key, idx)
        records.append(record)
    if not records:
        raise ValueError(f"No RoboTwin records found in {metadata_file}")
    return records

def _discover_robotwin_records(
    *,
    data_root: Path,
    task_config: str | None,
) -> list[dict[str, Any]]:
    pattern = f"*/{task_config}/video/*.mp4" if task_config else "*/*/video/*.mp4"
    records = []
    for idx, video_path in enumerate(sorted(data_root.glob(pattern))):
        base_dir = video_path.parent.parent
        instruction_path = base_dir / "instructions" / f"{video_path.stem}.json"
        prompt = _read_instruction(instruction_path)
        task_name = video_path.relative_to(data_root).parts[0]
        records.append(
            {
                "prompt": prompt,
                "caption": prompt,
                "video_path": str(video_path),
                "reference_video_path": str(video_path),
                "instruction_path": str(instruction_path),
                "task_name": task_name,
                "episode_id": video_path.stem,
                "dataset_index": idx,
            }
        )
    if not records:
        raise ValueError(
            f"No RoboTwin videos found under {data_root} with pattern {pattern}"
        )
    return records


def _normalize_robotwin_record(
    raw: dict[str, Any],
    data_root: Path,
    prompt_key: str,
    dataset_index: int,
) -> dict[str, Any]:
    prompt = first_nonempty_text(raw.get("prompt"))
    if not prompt:
        prompt = first_nonempty_text(raw.get(prompt_key))
    if not prompt:
        prompt = first_nonempty_text(raw.get("caption"))
    if not prompt:
        prompt = first_nonempty_text(raw.get("instruction"))

    video_value = raw.get("video_path") or raw.get("reference_video_path")
    image_value = raw.get("image_path") or raw.get("condition_image_path")
    if not video_value and not image_value:
        raise ValueError(
            "RoboTwin record requires `video_path`/`reference_video_path` or `image_path`."
        )

    record = dict(raw)
    record["prompt"] = prompt
    if video_value:
        video_path = resolve_data_path(data_root, str(video_value))
        reference_value = raw.get("reference_video_path") or video_value
        reference_video_path = resolve_data_path(data_root, str(reference_value))
        record["video_path"] = str(video_path)
        record["reference_video_path"] = str(reference_video_path)
    if image_value:
        record["image_path"] = str(resolve_data_path(data_root, str(image_value)))
    if "task_name" not in record and video_value:
        try:
            record["task_name"] = (
                resolve_data_path(data_root, str(video_value))
                .relative_to(data_root)
                .parts[0]
            )
        except (IndexError, ValueError):
            record["task_name"] = (
                Path(str(video_value)).parts[0]
                if Path(str(video_value)).parts
                else ""
            )
    if "episode_id" not in record:
        source = image_value or video_value or str(dataset_index)
        record["episode_id"] = Path(str(source)).stem
    record["dataset_index"] = dataset_index
    return record


def _load_first_video_frame(
    path: Path,
    image_size: tuple[int, int] | None,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"RoboTwin video not found: {path}")
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "RobotwinDataset needs `cv2` to read first-frame image conditions "
            "from mp4 files. Provide image_path in metadata or install opencv-python."
        ) from exc

    cap = cv2.VideoCapture(str(path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Failed to read first frame from RoboTwin video: {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        height, width = image_size
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    return frame.astype(np.uint8)


def _read_instruction(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return ""
    for key in ("seen", "unseen", "instruction", "caption", "prompt"):
        text = first_nonempty_text(data.get(key))
        if text:
            return text
    return ""
