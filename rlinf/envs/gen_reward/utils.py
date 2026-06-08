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
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def cfg_require(cfg: Any, key: str) -> Any:
    value = cfg_get(cfg, key, None)
    if value is None:
        raise KeyError(f"Missing required gen_reward config field: {key}")
    return value


def normalize_type(value: Any, prefixes: tuple[str, ...] = ()) -> str:
    type_name = str(value).lower().replace(":", ".")
    for prefix in prefixes:
        prefix = prefix if prefix.endswith(".") else f"{prefix}."
        if type_name.startswith(prefix):
            return type_name[len(prefix) :]
    return type_name


def parse_torch_dtype(value: Any) -> torch.dtype:
    dtype = str(value).lower()
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def text_condition_from_record(record: dict[str, Any]) -> str:
    for key in ("task_description", "prompt", "caption", "instruction"):
        value = record.get(key, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def metadata_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"main_image", "condition_image"}
    }


def obs_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    obs: dict[str, Any] = {
        "task_descriptions": [text_condition_from_record(record) for record in records]
    }
    if records and all("main_image" in record for record in records):
        obs["main_images"] = np.stack(
            [record["main_image"] for record in records],
            axis=0,
        )
    return obs


def media_to_uint8_nhwc(media: torch.Tensor | np.ndarray | list[Any]) -> np.ndarray:
    """Convert image/video batches to uint8 channel-last format."""
    if isinstance(media, list):
        arrays = []
        for item in media:
            if isinstance(item, Image.Image):
                arrays.append(np.asarray(item.convert("RGB"), dtype=np.uint8))
            else:
                arrays.append(np.asarray(item))
        media = np.stack(arrays, axis=0)
    elif isinstance(media, torch.Tensor):
        media = media.detach().cpu().float().numpy()
    else:
        media = np.asarray(media)

    if media.ndim == 3:
        media = media[None]
    if media.ndim not in (4, 5):
        raise ValueError(f"Expected image/video batch, got shape {media.shape}.")
    if np.issubdtype(media.dtype, np.floating):
        media = np.clip(media, 0.0, 1.0) * 255.0
    media = np.rint(media).clip(0, 255).astype(np.uint8)

    if media.ndim == 4:
        if media.shape[1] in (1, 3, 4) and media.shape[-1] not in (1, 3, 4):
            media = np.transpose(media, (0, 2, 3, 1))
    elif media.shape[-1] not in (1, 3, 4):
        if media.shape[2] in (1, 3, 4) and (
            media.shape[1] not in (1, 3, 4)
            or (media.shape[2] == 3 and media.shape[1] != 3)
        ):
            media = np.transpose(media, (0, 1, 3, 4, 2))
        elif media.shape[1] in (1, 3, 4):
            media = np.transpose(media, (0, 2, 3, 4, 1))

    if media.shape[-1] == 1:
        media = np.repeat(media, 3, axis=-1)
    if media.shape[-1] == 4:
        media = media[..., :3]
    return media


def extract_quoted_text(text: str) -> str:
    match = re.search(r'"([^"]+)"', str(text))
    return match.group(1) if match else str(text)


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_dataset_path(dataset_path: str | Path, dataset_cfg: Any) -> Path:
    dataset_path = Path(dataset_path)
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


def load_txt_prompts(dataset_path: Path) -> list[dict[str, Any]]:
    records = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                records.append({"prompt": prompt})
    if not records:
        raise ValueError(f"Prompt dataset is empty: {dataset_path}")
    return records


def resolve_data_path(data_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def first_nonempty_text(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = first_nonempty_text(item)
            if text:
                return text
    return ""


def parse_image_size(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace("x", ",").split(",")]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError("dataset.image_size must be [height, width].")
    return int(parts[0]), int(parts[1])


def load_rgb_image(path: Path, image_size: tuple[int, int] | None) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Condition image not found: {path}")
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        height, width = image_size
        image = image.resize((width, height), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


images_to_uint8_nhwc = media_to_uint8_nhwc


__all__ = [
    "cfg_get",
    "cfg_require",
    "extract_quoted_text",
    "first_nonempty_text",
    "images_to_uint8_nhwc",
    "load_txt_prompts",
    "load_jsonl_records",
    "load_rgb_image",
    "media_to_uint8_nhwc",
    "metadata_from_record",
    "normalize_type",
    "obs_from_records",
    "parse_image_size",
    "parse_torch_dtype",
    "resolve_dataset_path",
    "resolve_data_path",
    "text_condition_from_record",
]
