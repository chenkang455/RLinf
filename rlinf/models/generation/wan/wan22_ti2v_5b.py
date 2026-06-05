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

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.generation.sd3.utils import prompt_list


@dataclass
class Wan22_TI2V_5B_Config:
    model_path: str = ""
    resolution: int | list[int] = 480
    num_frames: int = 1
    num_steps: int = 8
    eval_num_steps: int = 20
    timestep_fraction: float = 0.99
    guidance_scale: float = 1.0
    eval_guidance_scale: float = 1.0
    cfg: bool = False
    max_sequence_length: int = 512
    output_type: str = "pt"
    rl_mode: Literal["flow-grpo", "nft"] = "nft"
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_path: str | None = None
    init_lora_weights: str = "gaussian"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            "ffn.net.0.proj",
            "ffn.net.2",
        ]
    )
    offload_auxiliary_modules: bool = True

    def update_from_dict(self, config_dict: Mapping[str, Any] | None):
        if not config_dict:
            return
        unknown_fields = sorted(set(config_dict) - set(self.__dataclass_fields__))
        if unknown_fields:
            raise ValueError(f"Unknown Wan22_TI2V_5B config fields: {unknown_fields}")
        for key, value in config_dict.items():
            setattr(self, key, value)


class Wan22_TI2V_5B(torch.nn.Module, BasePolicy):
    """Wan 2.x text-to-video policy with DiffusionNFT-style clean latent rollout."""

    def __init__(self, config: Wan22_TI2V_5B_Config, pipeline: Any):
        super().__init__()
        self.config = config
        self.model_path = str(config.model_path)
        self.pipeline = pipeline
        self.transformer = pipeline.transformer

    @property
    def _no_split_modules(self) -> list[str]:
        return list(getattr(self.transformer, "_no_split_modules", ["WanTransformerBlock"]))

    def to(self, device):
        module = super().to(device)
        device = torch.device(device)
        if device.type == "cpu":
            self.pipeline.vae.to(device=device)
            self.pipeline.text_encoder.to(device=device)
        return module

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.NFT:
            return self.nft_forward(**kwargs)
        raise NotImplementedError

    def default_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs):
        del kwargs
        batch_size = forward_inputs["nft_x0"].shape[0]
        num_train_timesteps = min(
            max(1, int(self.config.num_steps * self.config.timestep_fraction)),
            self.config.num_steps,
        )
        return {
            "logprobs": torch.zeros(
                batch_size,
                num_train_timesteps,
                device=forward_inputs["nft_x0"].device,
                dtype=forward_inputs["nft_x0"].dtype,
            ),
            "values": None,
        }

    def nft_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        nft_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        del kwargs
        device = next(self.transformer.parameters()).device
        model_dtype = next(self.transformer.parameters()).dtype
        x_t = nft_inputs["x_t"].to(device=device, dtype=model_dtype)
        timesteps = self._scheduler_to_model_timesteps(
            nft_inputs["timesteps"].to(device=device),
            x_t,
        )
        prompt_embeds = forward_inputs["prompt_embeds"].to(device=device, dtype=model_dtype)
        negative_prompt_embeds = forward_inputs.get("negative_prompt_embeds")
        if self.config.cfg:
            if negative_prompt_embeds is None:
                raise ValueError("Wan22_TI2V_5B cfg=True requires negative_prompt_embeds.")
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=device, dtype=model_dtype
            )

        v_theta = self._transformer_forward(
            x_t,
            timesteps,
            prompt_embeds,
            negative_prompt_embeds,
            self.config.guidance_scale,
        )
        return {"v_theta": v_theta}

    def obs_processor(self, env_obs: Any) -> list[str]:
        if isinstance(env_obs, Mapping):
            for key in ("prompts", "prompt", "texts", "text", "task_descriptions"):
                if key in env_obs:
                    return prompt_list(env_obs[key])
        if isinstance(env_obs, Sequence) and not isinstance(env_obs, str):
            return [str(prompt) for prompt in env_obs]
        return [str(env_obs)]

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts: str | Sequence[str],
        *,
        negative_prompts: str | Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = next(self.transformer.parameters()).device
        self.pipeline.text_encoder.to(device=device)
        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt_list(prompts),
            negative_prompt=(
                None if negative_prompts is None else prompt_list(negative_prompts)
            ),
            do_classifier_free_guidance=self.config.cfg,
            num_videos_per_prompt=1,
            max_sequence_length=self.config.max_sequence_length,
            device=device,
            dtype=next(self.transformer.parameters()).dtype,
        )
        return prompt_embeds.to(device), (
            None if negative_prompt_embeds is None else negative_prompt_embeds.to(device)
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del compute_values
        prompts = self.obs_processor(env_obs)
        negative_prompts = kwargs.get("negative_prompts") or [""] * len(prompts)
        prompt_embeds, negative_prompt_embeds = self.encode_prompts(
            prompts,
            negative_prompts=negative_prompts,
        )
        if self.config.offload_auxiliary_modules:
            self.pipeline.text_encoder.to(device="cpu")
            torch.cuda.empty_cache()

        is_eval = mode == "eval"
        num_steps = self.config.eval_num_steps if is_eval else self.config.num_steps
        guidance_scale = (
            self.config.eval_guidance_scale if is_eval else self.config.guidance_scale
        )
        images, final_latents = self._denoise(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            generator=kwargs.get("generator"),
            latents=kwargs.get("latents"),
        )
        if is_eval:
            return images, {"prev_values": None}

        num_train_timesteps = min(
            max(1, int(self.config.num_steps * self.config.timestep_fraction)),
            num_steps,
        )
        old_logprobs = torch.zeros(
            final_latents.shape[0],
            num_train_timesteps,
            device=final_latents.device,
            dtype=final_latents.dtype,
        )
        forward_inputs = {
            "nft_x0": final_latents.detach(),
            "nft_noise_level": torch.zeros(
                final_latents.shape[0],
                device=final_latents.device,
                dtype=final_latents.dtype,
            ),
            "prompt_embeds": prompt_embeds.detach(),
        }
        if negative_prompt_embeds is not None:
            forward_inputs["negative_prompt_embeds"] = negative_prompt_embeds.detach()

        return images, {
            "prev_logprobs": old_logprobs,
            "prev_values": None,
            "forward_inputs": forward_inputs,
        }

    @torch.no_grad()
    def _denoise(
        self,
        *,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_scale: float,
        num_steps: int,
        generator=None,
        latents=None,
    ):
        device = prompt_embeds.device
        model_dtype = next(self.transformer.parameters()).dtype
        self.pipeline.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.pipeline.scheduler.timesteps
        height, width = self._height_width()
        latents = self.pipeline.prepare_latents(
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=self.transformer.config.in_channels,
            height=height,
            width=width,
            num_frames=self.config.num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        self.pipeline._num_timesteps = len(timesteps)
        for t in timesteps:
            model_latents = latents.to(dtype=model_dtype)
            timestep = self._scheduler_to_model_timesteps(
                t.expand(latents.shape[0]), model_latents
            )
            noise_pred = self._transformer_forward(
                model_latents,
                timestep,
                prompt_embeds.to(dtype=model_dtype),
                (
                    None
                    if negative_prompt_embeds is None
                    else negative_prompt_embeds.to(dtype=model_dtype)
                ),
                guidance_scale,
            )
            latents = self.pipeline.scheduler.step(
                noise_pred.float(),
                t,
                latents.float(),
                return_dict=False,
            )[0]
        videos = self.decode_latents(latents, output_type=self.config.output_type)
        images = self._video_to_image_batch(videos)
        if self.config.offload_auxiliary_modules:
            self.pipeline.vae.to(device="cpu")
            torch.cuda.empty_cache()
        return images, latents

    def _video_to_image_batch(self, videos):
        if self.config.num_frames != 1:
            return videos
        if isinstance(videos, torch.Tensor) and videos.ndim == 5:
            if videos.shape[1] != 1:
                raise ValueError(f"Expected single-frame video output, got shape {videos.shape}.")
            return videos[:, 0]
        return videos

    def _transformer_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_scale: float,
    ) -> torch.Tensor:
        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        if self.config.cfg:
            noise_uncond = self.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                return_dict=False,
            )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
        return noise_pred

    def _scheduler_to_model_timesteps(
        self,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = timesteps.reshape(-1).to(device=latents.device)
        if timesteps.dtype.is_floating_point and timesteps.max() <= 1.0:
            timesteps = timesteps.to(dtype=torch.float32) * 1000.0
        if not getattr(self.pipeline.config, "expand_timesteps", False):
            return timesteps.expand(latents.shape[0])

        if timesteps.numel() == 1:
            timesteps = timesteps.expand(latents.shape[0])
        mask = torch.ones(latents.shape, dtype=torch.float32, device=latents.device)
        expanded = mask[:, 0, :, ::2, ::2] * timesteps.to(torch.float32).view(
            -1, 1, 1, 1
        )
        return expanded.flatten(1)

    def _height_width(self) -> tuple[int, int]:
        resolution = self.config.resolution
        if isinstance(resolution, Sequence) and not isinstance(resolution, str):
            if len(resolution) != 2:
                raise ValueError("Wan22_TI2V_5B resolution list must be [height, width].")
            return int(resolution[0]), int(resolution[1])
        value = int(resolution)
        return value, value

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, output_type: str = "pt"):
        self.pipeline.vae.to(device=latents.device)
        latents = latents.to(dtype=self.pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(
            self.pipeline.vae.config.latents_std
        ).view(1, self.pipeline.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.pipeline.vae.decode(latents, return_dict=False)[0]
        return self.pipeline.video_processor.postprocess_video(video, output_type)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        enable_fn = getattr(self.transformer, "gradient_checkpointing_enable", None)
        if enable_fn is None:
            return
        if gradient_checkpointing_kwargs is None:
            enable_fn()
        else:
            enable_fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        disable_fn = getattr(self.transformer, "gradient_checkpointing_disable", None)
        if disable_fn is not None:
            disable_fn()

    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def export_config(self) -> dict[str, Any]:
        config = asdict(self.config)
        wan_config = dict(config)
        wan_config.pop("model_path", None)
        return {"model_path": self.model_path, "wan22_ti2v_5b": wan_config}
