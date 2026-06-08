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

import copy
from typing import Any

import gym
import numpy as np
import torch

from rlinf.envs.utils import put_text_on_image

from .datasets import PromptDataset
from .rewards import GenRewardBackend, build_reward_backend, cfg_get


class GenRewardEnv(gym.Env):
    """One-step generation reward environment.

    Reset returns prompts. Step receives generated outputs and returns rewards.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info=None,
    ):
        del worker_info
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        base_seed = int(cfg_get(cfg, "seed", 42))
        self.seed = base_seed + self.seed_offset
        self.group_size = int(cfg_get(cfg, "group_size", 1))
        self.num_group = max(1, int(np.ceil(self.num_envs / max(1, self.group_size))))
        self.auto_reset = bool(cfg_get(cfg, "auto_reset", False))
        self.is_eval = bool(cfg_get(cfg, "is_eval", False))
        self._generator = np.random.default_rng(seed=self.seed)
        self._cursor = 0
        video_cfg = cfg_get(cfg, "video_cfg", {})
        self.image_frame_repeat = max(
            1, int(cfg_get(video_cfg, "image_frame_repeat", 8))
        )
        self.num_capture_samples = max(
            1, int(cfg_get(video_cfg, "num_capture_samples", 3))
        )
        self._last_capture_media: np.ndarray | None = None

        self.prompt_dataset = PromptDataset.from_config(cfg_get(cfg, "dataset", {}))
        reward_cfg = cfg_get(cfg, "reward", {})
        self.reward_key = str(cfg_get(reward_cfg, "key", "avg"))
        self.reward_backend: GenRewardBackend = build_reward_backend(reward_cfg)
        self._current_metadatas: list[dict[str, Any]] = []
        self._current_obs: dict[str, Any] | None = None

    def update_reset_state_ids(self):
        if self.is_eval:
            self._cursor = 0

    def _next_group_indices(self) -> np.ndarray:
        if self.is_eval:
            start = self._cursor + self.seed_offset * self.num_group
            self._cursor += self.num_group * self.total_num_processes
            indices = np.arange(start, start + self.num_group)
            return indices % len(self.prompt_dataset)
        return self._generator.integers(
            0,
            len(self.prompt_dataset),
            size=(self.num_group,),
        )

    def reset(self, *args, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        del args, kwargs
        self._last_capture_media = None
        group_indices = self._next_group_indices()
        records = [
            copy.deepcopy(self.prompt_dataset[int(index)]) for index in group_indices
        ]
        repeated_records = []
        for record in records:
            repeated_records.extend(copy.deepcopy(record) for _ in range(self.group_size))
        repeated_records = repeated_records[: self.num_envs]
        self._current_metadatas = repeated_records
        self._current_obs = {
            "task_descriptions": [
                str(record.get("prompt", "")) for record in repeated_records
            ]
        }
        return self._current_obs, {}

    def step(
        self, images: torch.Tensor | np.ndarray | list[Any], auto_reset: bool = True
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self._current_obs is None:
            self.reset()
        prompts = [metadata.get("prompt", "") for metadata in self._current_metadatas]
        self._last_capture_media = self._prepare_capture_media(images, prompts)
        scores = self.reward_backend.score(images, prompts, self._current_metadatas)
        if self.reward_key not in scores:
            raise KeyError(
                f"Reward key {self.reward_key!r} is missing. "
                f"Available keys: {sorted(scores)}"
            )
        rewards = scores[self.reward_key].float()
        truncations = torch.zeros_like(rewards, dtype=torch.bool)
        terminations = torch.ones_like(rewards, dtype=torch.bool)
        if rewards.ndim > 1:
            terminations[..., :-1] = False
        episode = {
            "return": rewards.detach().float(),
            "episode_len": torch.ones(self.num_envs, dtype=torch.float32),
        }
        for key, value in scores.items():
            episode[key] = value.detach().float()
        capture_media = self._last_capture_media
        final_obs = self._current_obs
        if auto_reset and self.auto_reset:
            next_obs, _ = self.reset()
            self._last_capture_media = capture_media
        else:
            next_obs = self._current_obs

        infos = {
            "episode": episode,
            "final_info": {"episode": episode},
            "final_observation": final_obs,
        }
        return next_obs, rewards, terminations, truncations, infos

    def capture_image(self) -> np.ndarray | None:
        return self._last_capture_media

    def _prepare_capture_media(
        self,
        media: torch.Tensor | np.ndarray | list[Any],
        prompts: list[str] | None = None,
    ) -> np.ndarray | None:
        if media is None:
            return None
        if isinstance(media, torch.Tensor):
            media = media.detach().cpu().numpy()
        else:
            media = np.asarray(media)

        if media.ndim not in (4, 5):
            return None
        if np.issubdtype(media.dtype, np.floating):
            media = np.clip(media, 0.0, 1.0) * 255.0
        media = np.rint(media).clip(0, 255).astype(np.uint8)

        if media.ndim == 4:
            if media.shape[1] in (1, 3, 4) and media.shape[-1] not in (1, 3, 4):
                media = np.transpose(media, (0, 2, 3, 1))
            media = np.repeat(media[:, None], self.image_frame_repeat, axis=1)
        elif media.shape[2] in (1, 3, 4) and media.shape[-1] not in (1, 3, 4):
            media = np.transpose(media, (0, 1, 3, 4, 2))

        media = media[: self.num_capture_samples]
        if prompts:
            media = self._put_prompts_on_capture_media(media, prompts)
        return media

    def _put_prompts_on_capture_media(
        self, media: np.ndarray, prompts: list[str]
    ) -> np.ndarray:
        media = media.copy()
        max_width = max(120, media.shape[3] - 20)
        for batch_idx, prompt in enumerate(prompts[: media.shape[0]]):
            lines = [f"prompt: {prompt}"]
            for frame_idx in range(media.shape[1]):
                media[batch_idx, frame_idx] = put_text_on_image(
                    media[batch_idx, frame_idx],
                    lines,
                    max_width=max_width,
                )
        return media

    def chunk_step(
        self, images: torch.Tensor | np.ndarray | list[Any]
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        obs, rewards, terminations, truncations, infos = self.step(
            images,
            auto_reset=False,
        )
        capture_media = self._last_capture_media
        if torch.logical_or(terminations, truncations).any() and self.auto_reset:
            obs, _ = self.reset()
            self._last_capture_media = capture_media
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(1)
            terminations = terminations.unsqueeze(1)
            truncations = truncations.unsqueeze(1)
        return ([obs], rewards, terminations, truncations, [infos])
