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

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class _CrossAttnKVEntry:
    text_key: torch.Tensor
    text_value: torch.Tensor
    image_key: torch.Tensor | None = None
    image_value: torch.Tensor | None = None


def _unwrap_transformer(transformer: nn.Module) -> nn.Module:
    module = transformer
    if hasattr(module, "get_base_model"):
        module = module.get_base_model()
    if hasattr(module, "model"):
        module = module.model
    return module


def _get_added_kv_projections(attn, encoder_hidden_states_img):
    if getattr(attn, "fused_projections", False):
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


def _dispatch_attention(query, key, value, *, processor: Any):
    try:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
    except ImportError:
        dispatch_attention_fn = None

    backend = getattr(processor, "_attention_backend", None)
    parallel_config = getattr(processor, "_parallel_config", None)

    if dispatch_attention_fn is not None:
        return dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=backend,
            parallel_config=parallel_config,
        )

    import torch.nn.functional as F

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2)


class WanCachedCrossAttnProcessor:
    """Use precomputed cross-attention K/V when available; otherwise fall back."""

    def __init__(self, fallback_processor: Any, transformer: nn.Module):
        self.fallback_processor = fallback_processor
        self.transformer = transformer

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None or not getattr(attn, "is_cross_attention", False):
            return self.fallback_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
            )

        branch = getattr(self.transformer, "_rlinf_cross_attn_branch", "cond")
        branch_cache = getattr(self.transformer, "_rlinf_cross_attn_kv_by_branch", None)
        layer_idx = getattr(attn, "_rlinf_cross_attn_layer_idx", None)
        if (
            branch_cache is None
            or branch not in branch_cache
            or layer_idx is None
            or layer_idx >= len(branch_cache[branch])
        ):
            return self.fallback_processor(
                attn,
                hidden_states,
                encoder_hidden_states,
                attention_mask,
                rotary_emb,
            )

        entry = branch_cache[branch][layer_idx]
        query = attn.to_q(hidden_states)
        query = attn.norm_q(query)
        query = query.unflatten(2, (attn.heads, -1))

        key = entry.text_key
        value = entry.text_value

        hidden_states_img = None
        if entry.image_key is not None and entry.image_value is not None:
            hidden_states_img = _dispatch_attention(
                query,
                entry.image_key,
                entry.image_value,
                processor=self.fallback_processor,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3).type_as(query)

        hidden_states = _dispatch_attention(
            query,
            key,
            value,
            processor=self.fallback_processor,
        )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def install_wan_cross_attn_cache_processors(transformer: nn.Module) -> None:
    if getattr(transformer, "_rlinf_cross_attn_processors_installed", False):
        return

    inner = _unwrap_transformer(transformer)
    for layer_idx, block in enumerate(inner.blocks):
        attn = block.attn2
        attn._rlinf_cross_attn_layer_idx = layer_idx
        if not isinstance(attn.processor, WanCachedCrossAttnProcessor):
            attn.set_processor(WanCachedCrossAttnProcessor(attn.processor, transformer))

    transformer._rlinf_cross_attn_processors_installed = True


@torch.no_grad()
def precompute_wan_cross_attn_kv(
    transformer: nn.Module,
    prompt_embeds: torch.Tensor,
) -> list[_CrossAttnKVEntry]:
    inner = _unwrap_transformer(transformer)
    encoder_hidden_states = inner.condition_embedder.text_embedder(prompt_embeds)

    entries: list[_CrossAttnKVEntry] = []
    for block in inner.blocks:
        attn = block.attn2
        enc = encoder_hidden_states
        enc_img = None
        if attn.add_k_proj is not None:
            image_context_length = enc.shape[1] - 512
            enc_img = enc[:, :image_context_length]
            enc = enc[:, image_context_length:]

        key = attn.norm_k(attn.to_k(enc))
        value = attn.to_v(enc)
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        key_img = value_img = None
        if enc_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, enc_img)
            key_img = attn.norm_added_k(key_img)
            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

        entries.append(
            _CrossAttnKVEntry(
                text_key=key,
                text_value=value,
                image_key=key_img,
                image_value=value_img,
            )
        )
    return entries


def set_cross_attn_cache_branch(transformer: nn.Module, branch: str) -> None:
    transformer._rlinf_cross_attn_branch = branch


def activate_cross_attn_cache(
    transformer: nn.Module,
    *,
    cond_prompt_embeds: torch.Tensor,
    uncond_prompt_embeds: torch.Tensor | None = None,
) -> None:
    branch_cache = {
        "cond": precompute_wan_cross_attn_kv(transformer, cond_prompt_embeds),
    }
    if uncond_prompt_embeds is not None:
        branch_cache["uncond"] = precompute_wan_cross_attn_kv(
            transformer, uncond_prompt_embeds
        )
    transformer._rlinf_cross_attn_kv_by_branch = branch_cache
    transformer._rlinf_cross_attn_branch = "cond"


def deactivate_cross_attn_cache(transformer: nn.Module) -> None:
    if hasattr(transformer, "_rlinf_cross_attn_kv_by_branch"):
        del transformer._rlinf_cross_attn_kv_by_branch
