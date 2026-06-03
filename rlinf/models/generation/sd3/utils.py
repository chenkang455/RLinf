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

import math
from typing import Any

import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from diffusers.utils.torch_utils import randn_tensor


def sde_step_with_logprob(
    scheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    *,
    noise_level: float,
    prev_sample: torch.Tensor | None = None,
):
    model_output = model_output.float()
    sample = sample.float()
    prev_sample = None if prev_sample is None else prev_sample.float()

    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [idx + 1 for idx in step_index]
    sigma = scheduler.sigmas[step_index].view(-1, *([1] * (sample.ndim - 1)))
    sigma_prev = scheduler.sigmas[prev_step_index].view(
        -1, *([1] * (sample.ndim - 1))
    )
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma

    std = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))
    std = std * noise_level
    mean = sample * (1 + std**2 / (2 * sigma) * dt)
    mean += model_output * (1 + std**2 * (1 - sigma) / (2 * sigma)) * dt

    diffusion_std = std * torch.sqrt(-dt)
    if prev_sample is None:
        noise = randn_tensor(
            model_output.shape,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = mean + diffusion_std * noise

    log_prob = (
        -((prev_sample.detach() - mean) ** 2) / (2 * diffusion_std**2)
        - torch.log(diffusion_std)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob, mean, std


@torch.no_grad()
def denoise_with_logprob(
    *,
    pipeline: Any,
    transformer: torch.nn.Module,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    negative_pooled_prompt_embeds: torch.Tensor | None,
    cfg_enabled: bool,
    guidance_scale: float,
    noise_level: float,
    num_steps: int,
    resolution: int,
    output_type: str,
    generator=None,
    latents=None,
):
    pipeline._guidance_scale = guidance_scale
    pipeline._interrupt = False

    model_prompt_embeds = prompt_embeds
    model_pooled_embeds = pooled_prompt_embeds
    if cfg_enabled:
        model_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        model_pooled_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

    latents = pipeline.prepare_latents(
        prompt_embeds.shape[0],
        transformer.config.in_channels,
        resolution,
        resolution,
        prompt_embeds.dtype,
        prompt_embeds.device,
        generator,
        latents,
    ).float()
    timesteps, num_steps = retrieve_timesteps(pipeline.scheduler, num_steps, prompt_embeds.device)
    pipeline._num_timesteps = len(timesteps)

    latent_chain = [latents]
    log_probs = []
    with pipeline.progress_bar(total=num_steps) as progress_bar:
        for t in timesteps:
            model_input = torch.cat([latents] * 2) if cfg_enabled else latents
            timestep = t.expand(model_input.shape[0])
            noise_pred = transformer(
                hidden_states=model_input,
                timestep=timestep,
                encoder_hidden_states=model_prompt_embeds,
                pooled_projections=model_pooled_embeds,
                return_dict=False,
            )[0]
            if cfg_enabled:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            latents, log_prob, _, _ = sde_step_with_logprob(
                pipeline.scheduler,
                noise_pred,
                t.unsqueeze(0),
                latents,
                noise_level=noise_level,
            )
            latent_chain.append(latents)
            log_probs.append(log_prob)
            progress_bar.update()

    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    latents = latents.to(dtype=pipeline.vae.dtype)
    images = pipeline.vae.decode(latents, return_dict=False)[0]
    images = pipeline.image_processor.postprocess(images, output_type=output_type)
    pipeline.maybe_free_model_hooks()
    return images, latent_chain, log_probs
