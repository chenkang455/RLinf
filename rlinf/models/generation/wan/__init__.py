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
from diffusers import WanPipeline
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model

from rlinf.config import torch_dtype_from_precision
from rlinf.models.generation.wan.wan22_ti2v_5b import Wan22_TI2V_5B, Wan22_TI2V_5B_Config


def get_model(cfg: DictConfig, torch_dtype=None):
    model_config = Wan22_TI2V_5B_Config(model_path=str(cfg.model_path))
    model_config.update_from_dict(
        OmegaConf.to_container(cfg.get("wan22_ti2v_5b", {}), resolve=True) or {}
    )
    inference_dtype = torch_dtype or torch_dtype_from_precision(cfg.precision)
    pipeline = WanPipeline.from_pretrained(
        model_config.model_path,
        torch_dtype=inference_dtype,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(not model_config.use_lora)

    if model_config.use_lora:
        if model_config.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(
                pipeline.transformer,
                str(model_config.lora_path),
                is_trainable=True,
            )
            pipeline.transformer.set_adapter("default")
        else:
            lora_config = LoraConfig(
                r=model_config.lora_rank,
                lora_alpha=model_config.lora_alpha,
                init_lora_weights=model_config.init_lora_weights,
                target_modules=model_config.target_modules,
            )
            pipeline.transformer = get_peft_model(pipeline.transformer, lora_config)

    pipeline.vae.to(dtype=torch.float32)
    pipeline.text_encoder.to(dtype=inference_dtype)
    pipeline.transformer.to(dtype=inference_dtype)
    return Wan22_TI2V_5B(model_config, pipeline=pipeline)


__all__ = ["Wan22_TI2V_5B", "Wan22_TI2V_5B_Config", "get_model"]
