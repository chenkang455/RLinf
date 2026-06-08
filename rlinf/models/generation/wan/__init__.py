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

import re
from pathlib import Path

import torch
from diffusers import WanPipeline
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model

from rlinf.config import torch_dtype_from_precision
from rlinf.models.generation.wan.wan22_ti2v_5b import (
    Wan22_TI2V_5B,
    Wan22_TI2V_5B_Config,
)



def _map_vidar_transformer_key(key: str) -> str | None:
    if key.endswith(".te_attention._extra_state"):
        return None

    exact_prefixes = (
        ("text_embedding.0.", "condition_embedder.text_embedder.linear_1."),
        ("text_embedding.2.", "condition_embedder.text_embedder.linear_2."),
        ("time_embedding.0.", "condition_embedder.time_embedder.linear_1."),
        ("time_embedding.2.", "condition_embedder.time_embedder.linear_2."),
        ("time_projection.1.", "condition_embedder.time_proj."),
        ("head.head.", "proj_out."),
    )
    for source, target in exact_prefixes:
        if key.startswith(source):
            return key.replace(source, target, 1)

    if key == "head.modulation":
        return "scale_shift_table"

    block_modulation_match = re.fullmatch(r"blocks\.(\d+)\.modulation", key)
    if block_modulation_match:
        return f"blocks.{block_modulation_match.group(1)}.scale_shift_table"

    replacements = (
        (".self_attn.q.", ".attn1.to_q."),
        (".self_attn.k.", ".attn1.to_k."),
        (".self_attn.v.", ".attn1.to_v."),
        (".self_attn.o.", ".attn1.to_out.0."),
        (".self_attn.norm_q.", ".attn1.norm_q."),
        (".self_attn.norm_k.", ".attn1.norm_k."),
        (".cross_attn.q.", ".attn2.to_q."),
        (".cross_attn.k.", ".attn2.to_k."),
        (".cross_attn.v.", ".attn2.to_v."),
        (".cross_attn.o.", ".attn2.to_out.0."),
        (".cross_attn.norm_q.", ".attn2.norm_q."),
        (".cross_attn.norm_k.", ".attn2.norm_k."),
        (".ffn.0.", ".ffn.net.0.proj."),
        (".ffn.2.", ".ffn.net.2."),
        (".norm3.", ".norm2."),
    )
    mapped_key = key
    for source, target in replacements:
        mapped_key = mapped_key.replace(source, target)
    return mapped_key


def _load_vidar_transformer_weights(
    transformer: torch.nn.Module, vidar_path: str | None
) -> None:
    weight_path = Path(vidar_path)
    state_dict = torch.load(
        weight_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    converted_state_dict = {}
    for key, value in state_dict.items():
        mapped_key = _map_vidar_transformer_key(key)
        if mapped_key is not None:
            converted_state_dict[mapped_key] = value

    missing_keys, unexpected_keys = transformer.load_state_dict(
        converted_state_dict, strict=True
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Failed to load converted Vidar transformer weights: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )


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
    if model_config.weight_format == "vidar":
        _load_vidar_transformer_weights(
            pipeline.transformer, model_config.vidar_path
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
