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

import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.envs.gen_reward import GenRewardBackend
from rlinf.envs.gen_reward.utils import (
    cfg_get,
    media_to_uint8_nhwc,
    parse_torch_dtype,
    text_condition_from_record,
)

_DEFAULT_DIFFUSION_NFT_PATH = "/mnt/public2/chenkang/wam_rl/video_rl/DiffusionNFT"


class HPSv21RewardBackend(GenRewardBackend):
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        diffusion_nft_path: str = _DEFAULT_DIFFUSION_NFT_PATH,
        checkpoint_dir: str | None = None,
    ):
        diffusion_nft_path = str(Path(diffusion_nft_path))
        if diffusion_nft_path not in sys.path:
            sys.path.insert(0, diffusion_nft_path)
        try:
            module = import_module("flow_grpo.hpsv2_scorer")
        except ImportError as exc:
            raise ImportError(
                "HPSv2.1 reward requires the DiffusionNFT scorer and "
                "`hpsv2x==1.2.0`. Follow NVlabs/DiffusionNFT reward "
                "environment setup before using reward.type=generation.hpsv2."
            ) from exc
        if checkpoint_dir is not None:
            module.CKPT_PATH = str(Path(checkpoint_dir))
        self.device = torch.device(device)
        self.dtype = dtype
        self.scorer = module.HPSv2Scorer(dtype=dtype, device=str(self.device))
        self.scorer.eval().requires_grad_(False)

    @torch.no_grad()
    def score(
        self,
        outputs: torch.Tensor | np.ndarray | list[Any],
        records: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        prompts = [text_condition_from_record(record) for record in records]
        images = media_to_uint8_nhwc(outputs)
        if images.ndim == 5:
            images = images[:, 0]
        images_tensor = (
            torch.as_tensor(images, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .div_(255.0)
        )
        rewards = self.scorer(images_tensor, prompts).detach().float().cpu()
        return {"avg": rewards, "hpsv2": rewards}


def build_reward_backend(cfg: Any) -> HPSv21RewardBackend:
    return HPSv21RewardBackend(
        device=str(cfg_get(cfg, "device", "cuda")),
        dtype=parse_torch_dtype(cfg_get(cfg, "dtype", "float32")),
        diffusion_nft_path=str(
            cfg_get(cfg, "diffusion_nft_path", _DEFAULT_DIFFUSION_NFT_PATH)
        ),
        checkpoint_dir=cfg_get(cfg, "checkpoint_dir", None),
    )


__all__ = ["HPSv21RewardBackend", "build_reward_backend"]
