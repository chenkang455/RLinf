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

import re
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


__all__ = [
    "cfg_get",
    "cfg_require",
    "extract_quoted_text",
    "media_to_uint8_nhwc",
    "normalize_type",
]
