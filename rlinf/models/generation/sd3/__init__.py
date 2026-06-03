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

import torch
from diffusers import StableDiffusion3Pipeline
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model

from rlinf.models.generation.sd3.stable_diffusion3 import (
    StableDiffusion3,
    StableDiffusion3Config,
)


def _dtype_from_precision(precision: str):
    precision = str(precision).lower()
    if precision in {"fp16", "float16", "half"}:
        return torch.float16
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def get_model(cfg: DictConfig, torch_dtype=None):
    model_config = StableDiffusion3Config(model_path=str(cfg.model_path))
    model_config.update_from_dict(
        OmegaConf.to_container(cfg.get("sd3", {}), resolve=True) or {}
    )
    if not model_config.model_path:
        raise ValueError("actor.model.model_path must point to an SD3 checkpoint.")

    inference_dtype = torch_dtype or _dtype_from_precision(cfg.get("precision", "fp32"))
    pipeline = None
    if model_config.load_pipeline_on_init:
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_config.model_path,
            torch_dtype=inference_dtype,
        )
        pipeline.safety_checker = None
        pipeline.set_progress_bar_config(disable=True)

        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)
        pipeline.text_encoder_3.requires_grad_(False)
        pipeline.transformer.requires_grad_(not model_config.use_lora)

        if model_config.use_lora:
            if model_config.lora_path:
                pipeline.transformer = PeftModel.from_pretrained(
                    pipeline.transformer,
                    str(model_config.lora_path),
                    is_trainable=True,
                )
                if hasattr(pipeline.transformer, "set_adapter"):
                    pipeline.transformer.set_adapter("default")
            else:
                lora_config = LoraConfig(
                    r=model_config.lora_rank,
                    lora_alpha=model_config.lora_alpha,
                    init_lora_weights=model_config.init_lora_weights,
                    target_modules=model_config.target_modules,
                )
                pipeline.transformer = get_peft_model(
                    pipeline.transformer,
                    lora_config,
                )

        pipeline.vae.to(dtype=torch.float32)
        pipeline.text_encoder.to(dtype=inference_dtype)
        pipeline.text_encoder_2.to(dtype=inference_dtype)
        pipeline.text_encoder_3.to(dtype=inference_dtype)
        pipeline.transformer.to(dtype=inference_dtype)

    return StableDiffusion3(model_config, pipeline=pipeline)


__all__ = ["StableDiffusion3", "StableDiffusion3Config", "get_model"]
