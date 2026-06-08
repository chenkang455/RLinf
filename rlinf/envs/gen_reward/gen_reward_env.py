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

from . import GenRewardBackend, build_reward_backend, build_reward_dataset
from .utils import cfg_get, cfg_require, media_to_uint8_nhwc, obs_from_records


class GenRewardEnv(gym.Env):
    """One-step generated-output reward environment.

    Reset returns dataset context. Step receives generated outputs and returns rewards.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info=None,
    ):
        # config
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        base_seed = int(cfg_get(cfg, "seed", 42))
        self.seed = base_seed + self.seed_offset
        self.group_size = int(cfg_get(cfg, "group_size", 1))
        self.num_group = max(1, int(np.ceil(self.num_envs / max(1, self.group_size))))
        self.is_eval = bool(cfg_get(cfg, "is_eval", False))
        self._generator = np.random.default_rng(seed=self.seed)
        self._cursor = 0
        # dataset and reward backend
        self.dataset = build_reward_dataset(cfg_require(cfg, "dataset"))
        reward_cfg = cfg_require(cfg, "reward")
        self.reward_key = str(cfg_get(reward_cfg, "key", "avg"))
        self.reward_backend: GenRewardBackend = build_reward_backend(reward_cfg)
        # video capture settings
        video_cfg = cfg_get(cfg, "video_cfg", {})
        self.image_frame_repeat = max(
            1, int(cfg_get(video_cfg, "image_frame_repeat", 8))
        ) # repeat the image frame to capture the video
        self.num_capture_samples = max(
            1, int(cfg_get(video_cfg, "num_capture_samples", 3))
        ) # number of samples to capture the video
        self._return_video: np.ndarray | None = None
        self._env_records: list[dict[str, Any]] = [] # metadata of the dataset
        self._env_obs: dict[str, Any] | None = None # observation of the dataset

    def update_reset_state_ids(self):
        if self.is_eval:
            self._cursor = 0

    def _next_group_indices(self) -> np.ndarray:
        if self.is_eval:
            start = self._cursor + self.seed_offset * self.num_group
            self._cursor += self.num_group * self.total_num_processes
            indices = np.arange(start, start + self.num_group)
            return indices % len(self.dataset)
        return self._generator.integers(
            0,
            len(self.dataset),
            size=(self.num_group,),
        )

    def reset(self, *args, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        self._return_video = None
        group_indices = self._next_group_indices()
        records = [copy.deepcopy(self.dataset[int(index)]) for index in group_indices]
        repeated_records = []
        for record in records:
            repeated_records.extend(copy.deepcopy(record) for _ in range(self.group_size))
        repeated_records = repeated_records[: self.num_envs]
        self._env_records = repeated_records
        self._env_obs = obs_from_records(repeated_records)
        return self._env_obs, {}

    def step(
        self, outputs: torch.Tensor | np.ndarray | list[Any]
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        task_descriptions = self._env_obs.get("task_descriptions")
        self._return_video = self._prepare_capture_media(
            outputs,
            task_descriptions,
        )
        scores = self.reward_backend.score(outputs, self._env_records)
        # return info
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
        final_obs = self._env_obs
        next_obs = self._env_obs
        infos = {
            "episode": episode,
            "final_info": {"episode": episode},
            "final_observation": final_obs,
        }
        return next_obs, rewards, terminations, truncations, infos

    def capture_image(self) -> np.ndarray | None:
        return self._return_video

    def _prepare_capture_media(
        self,
        media: torch.Tensor | np.ndarray | list[Any],
        task_descriptions: list[str] | None = None,
    ) -> np.ndarray | None:
        media = media_to_uint8_nhwc(media)
        if media.ndim == 4:
            media = np.repeat(media[:, None], self.image_frame_repeat, axis=1)
        media = media[: self.num_capture_samples]
        media = self._put_task_descriptions_on_capture_media(
            media,
            task_descriptions,
        )
        return media

    def _put_task_descriptions_on_capture_media(
        self, media: np.ndarray, task_descriptions: list[str]
    ) -> np.ndarray:
        media = media.copy()
        max_width = max(120, media.shape[3] - 20)
        for batch_idx, task_description in enumerate(
            task_descriptions[: media.shape[0]]
        ):
            lines = [f"prompt: {task_description}"]
            for frame_idx in range(media.shape[1]):
                media[batch_idx, frame_idx] = put_text_on_image(
                    media[batch_idx, frame_idx],
                    lines,
                    max_width=max_width,
                )
        return media

    def chunk_step(
        self, outputs: torch.Tensor | np.ndarray | list[Any]
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        obs, rewards, terminations, truncations, infos = self.step(outputs)
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(1)
            terminations = terminations.unsqueeze(1)
            truncations = truncations.unsqueeze(1)
        return ([obs], rewards, terminations, truncations, [infos])
