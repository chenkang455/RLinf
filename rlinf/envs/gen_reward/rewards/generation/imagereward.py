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

import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from rlinf.envs.gen_reward import GenRewardBackend
from rlinf.envs.gen_reward.utils import (
    cfg_get,
    media_to_uint8_nhwc,
    parse_torch_dtype,
    text_condition_from_record,
)

_DEFAULT_DIFFUSION_NFT_PATH = "/mnt/public2/chenkang/wam_rl/video_rl/DiffusionNFT"


class ImageRewardBackend(GenRewardBackend):
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        diffusion_nft_path: str = _DEFAULT_DIFFUSION_NFT_PATH,
        hf_endpoint: str | None = "https://hf-mirror.com",
    ):
        diffusion_nft_path = str(Path(diffusion_nft_path))
        if diffusion_nft_path not in sys.path:
            sys.path.insert(0, diffusion_nft_path)
        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", str(hf_endpoint))
        try:
            import transformers.modeling_utils as modeling_utils
            import transformers.pytorch_utils as pytorch_utils

            for name in (
                "apply_chunking_to_forward",
                "find_pruneable_heads_and_indices",
                "prune_linear_layer",
            ):
                if not hasattr(modeling_utils, name):
                    setattr(modeling_utils, name, getattr(pytorch_utils, name))
            module = import_module("flow_grpo.imagereward_scorer")
        except ImportError as exc:
            raise ImportError(
                "ImageReward reward requires `image-reward` and OpenAI CLIP. "
                "Follow NVlabs/DiffusionNFT reward environment setup before "
                "using reward.type=generation.imagereward."
            ) from exc
        self.scorer = module.ImageRewardScorer(device=device, dtype=dtype)
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
        pil_images = [Image.fromarray(image) for image in images]
        _, rewards = self.scorer.model.inference_rank(prompts, pil_images)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        if rewards.numel() == len(prompts) * len(prompts):
            rewards = rewards.reshape(len(prompts), len(prompts)).diagonal()
        rewards = rewards.reshape(len(prompts)).cpu()
        return {"avg": rewards, "imagereward": rewards}


def build_reward_backend(cfg: Any) -> ImageRewardBackend:
    return ImageRewardBackend(
        device=str(cfg_get(cfg, "device", "cuda")),
        dtype=parse_torch_dtype(cfg_get(cfg, "dtype", "float32")),
        diffusion_nft_path=str(
            cfg_get(cfg, "diffusion_nft_path", _DEFAULT_DIFFUSION_NFT_PATH)
        ),
        hf_endpoint=cfg_get(cfg, "hf_endpoint", "https://hf-mirror.com"),
    )


__all__ = ["ImageRewardBackend", "build_reward_backend"]
