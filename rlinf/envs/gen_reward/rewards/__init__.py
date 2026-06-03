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

from .base import GenRewardBackend, cfg_get
from .basic import JPEGCompressibilityRewardBackend, MockGenRewardBackend
from .geneval import GenevalRewardBackend
from .imagereward import ImageRewardBackend
from .multi import MultiRewardBackend
from .ocr import OCRRewardBackend
from .pickscore import PickScoreRewardBackend


def build_reward_backend(cfg: Any) -> GenRewardBackend:
    reward_type = str(cfg_get(cfg, "type", "mock")).lower()
    device = str(cfg_get(cfg, "device", "cuda"))
    dtype = _parse_dtype(str(cfg_get(cfg, "dtype", "float32")))

    if reward_type in {"mock", "debug"}:
        return MockGenRewardBackend()
    if reward_type == "multi":
        return _build_multi_reward_backend(cfg)
    if reward_type in {"jpeg", "jpeg_compressibility"}:
        return JPEGCompressibilityRewardBackend(
            invert=bool(cfg_get(cfg, "invert", True)),
            quality=int(cfg_get(cfg, "quality", 95)),
            scale=float(cfg_get(cfg, "scale", 500.0)),
        )
    if reward_type == "ocr":
        return OCRRewardBackend(
            use_gpu=bool(cfg_get(cfg, "use_gpu", False)),
            lang=str(cfg_get(cfg, "lang", "en")),
        )
    if reward_type == "pickscore":
        return PickScoreRewardBackend(device=device, dtype=dtype)
    if reward_type in {"imagereward", "image_reward"}:
        return ImageRewardBackend(device=device, dtype=dtype)
    if reward_type == "geneval":
        return GenevalRewardBackend(
            url=str(cfg_get(cfg, "url", "http://127.0.0.1:18085")),
            only_strict=bool(cfg_get(cfg, "only_strict", True)),
            batch_size=int(cfg_get(cfg, "batch_size", 64)),
            timeout=int(cfg_get(cfg, "timeout", 120)),
        )
    raise ValueError(f"Unknown generation reward type: {reward_type}")


def _build_multi_reward_backend(cfg: Any) -> MultiRewardBackend:
    scores = cfg_get(cfg, "scores", cfg_get(cfg, "score_dict", {}))
    weighted_backends = {}
    for name, value in dict(scores).items():
        if hasattr(value, "items"):
            sub_cfg = dict(value)
            weight = float(sub_cfg.pop("weight", 1.0))
            sub_cfg.setdefault("type", name)
        else:
            weight = float(value)
            sub_cfg = {"type": name}
        weighted_backends[str(name)] = (weight, build_reward_backend(sub_cfg))
    return MultiRewardBackend(weighted_backends)


def _parse_dtype(dtype: str) -> torch.dtype:
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


__all__ = [
    "GenRewardBackend",
    "build_reward_backend",
]
