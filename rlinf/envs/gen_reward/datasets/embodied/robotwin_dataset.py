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
from PIL import Image

from rlinf.envs.gen_reward.datasets import ImageConditionedDataset
from rlinf.envs.gen_reward.utils import cfg_get


class RobotwinDataset(ImageConditionedDataset):
    """RoboTwin text-image-to-video records."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        image_size: tuple[int, int] | None,
    ):
        super().__init__(records)
        self.image_size = image_size

    @classmethod
    def from_config(cls, dataset_cfg: Any) -> "RobotwinDataset":
        data_root = Path(
            str(cfg_get(dataset_cfg, "data_root", cfg_get(dataset_cfg, "path")))
        ).expanduser()
        metadata_path = cfg_get(dataset_cfg, "metadata_path", None)
        prompt_key = str(cfg_get(dataset_cfg, "prompt_key", "caption"))
        task_config = cfg_get(dataset_cfg, "task_config", None)
        split = cfg_get(dataset_cfg, "split", None)
        image_size = _parse_image_size(cfg_get(dataset_cfg, "image_size", None))

        if metadata_path is None:
            records = _discover_robotwin_records(
                data_root=data_root,
                task_config=None if task_config is None else str(task_config),
            )
        else:
            records = _load_robotwin_metadata(
                data_root=data_root,
                metadata_path=Path(str(metadata_path)),
                prompt_key=prompt_key,
                split=None if split is None else str(split),
            )
        return cls(records, image_size=image_size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        main_image_path = record.get("main_image_path")
        future_video_path = record.get("future_video_path")
        if main_image_path:
            image = _load_rgb_image(Path(str(main_image_path)), self.image_size)
        elif future_video_path:
            image = _load_first_video_frame(
                Path(str(future_video_path)),
                self.image_size,
            )
        else:
            raise ValueError(
                "RoboTwin record requires `main_image_path` or `future_video_path`."
            )
        record["main_image"] = image
        return record


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_data_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def _first_nonempty_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = _first_nonempty_text(item)
            if text:
                return text
    return ""


def _parse_image_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("x", ",").split(",")]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError("dataset.image_size must be [height, width].")
    return int(parts[0]), int(parts[1])


def _load_rgb_image(path: Path, image_size: tuple[int, int] | None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Condition image not found: {path}")
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        height, width = image_size
        image = image.resize((width, height), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


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
    if metadata_file.suffix == ".jsonl":
        raw_records = _load_jsonl_records(metadata_file)
    else:
        with metadata_file.open("r", newline="", encoding="utf-8") as f:
            raw_records = list(csv.DictReader(f))

    records = []
    for idx, raw in enumerate(raw_records):
        if split is not None and raw.get("split") not in (None, "", split):
            continue
        records.append(_normalize_robotwin_record(raw, data_root, prompt_key, idx))
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
                "task_description": prompt,
                "main_image_path": None,
                "future_video_path": str(video_path),
                "future_video": None,
                "task_name": task_name,
                "episode_id": video_path.stem,
                "dataset_index": idx,
                "metadata": {
                    "instruction_path": str(instruction_path),
                    "source": "robotwin_discovery",
                },
            }
        )
    return records


def _normalize_robotwin_record(
    raw: dict[str, Any],
    data_root: Path,
    prompt_key: str,
    dataset_index: int,
) -> dict[str, Any]:
    prompt = _first_nonempty_text(raw.get("task_description"))
    if not prompt:
        prompt = _first_nonempty_text(raw.get("prompt"))
    if not prompt:
        prompt = _first_nonempty_text(raw.get(prompt_key))
    if not prompt:
        prompt = _first_nonempty_text(raw.get("caption"))
    if not prompt:
        prompt = _first_nonempty_text(raw.get("instruction"))

    raw_future_video = raw.get("future_video")
    video_value = (
        raw.get("future_video_path")
        or (raw_future_video if isinstance(raw_future_video, str) else None)
        or raw.get("video_path")
        or raw.get("reference_video_path")
    )
    image_value = (
        raw.get("main_image_path")
        or raw.get("image_path")
        or raw.get("condition_image_path")
    )
    main_image_path = (
        str(_resolve_data_path(data_root, str(image_value))) if image_value else None
    )
    future_video_path = (
        str(_resolve_data_path(data_root, str(video_value))) if video_value else None
    )

    record = {
        "task_description": prompt,
        "main_image_path": main_image_path,
        "future_video_path": future_video_path,
        "future_video": (
            None if isinstance(raw_future_video, str) else raw_future_video
        ),
        "task_name": raw.get("task_name"),
        "episode_id": raw.get("episode_id"),
        "dataset_index": dataset_index,
        "metadata": dict(raw),
    }
    if not record["task_name"]:
        source = future_video_path or main_image_path or ""
        try:
            record["task_name"] = Path(str(source)).relative_to(data_root).parts[0]
        except (IndexError, ValueError):
            source_path = Path(str(source))
            record["task_name"] = source_path.parts[0] if source_path.parts else ""
    if not record["episode_id"]:
        source = main_image_path or future_video_path or str(dataset_index)
        record["episode_id"] = Path(str(source)).stem
    return record


def _load_first_video_frame(
    path: Path,
    image_size: tuple[int, int] | None,
) -> np.ndarray:
    import cv2

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
        text = _first_nonempty_text(data.get(key))
        if text:
            return text
    return ""


DATASET_CLS = RobotwinDataset


__all__ = ["DATASET_CLS", "RobotwinDataset"]
