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

import json
from pathlib import Path

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open


def _promote_scalar_params_to_1d(model):
    """FSDP does not support 0-d parameters, so we promote scalar Parameters to shape=[1]."""
    scalar_param_names = [name for name, p in model.named_parameters() if p.ndim == 0]
    for full_name in scalar_param_names:
        if "." in full_name:
            module_name, param_name = full_name.rsplit(".", 1)
            module = model.get_submodule(module_name)
        else:
            module = model
            param_name = full_name

        old_p = getattr(module, param_name)
        new_p = nn.Parameter(
            old_p.detach().reshape(1),
            requires_grad=old_p.requires_grad,
        )
        setattr(module, param_name, new_p)


def get_model(cfg: DictConfig, torch_dtype=None):
    """Load DreamZero policy from checkpoint."""
    # Fast-iteration path: keep DreamZeroPolicy in the call stack, but replace
    # the heavyweight backbone/action head with tiny fake components.
    if cfg.get("use_fake_model", False):
        from rlinf.models.embodiment.dreamzero.fake_dreamzero_policy import (
            FakeDreamZeroDataTransforms,
        )
        from rlinf.models.embodiment.dreamzero.dreamzero_policy import (
            DreamZeroConfig,
            DreamZeroPolicy,
        )

        action_dim = cfg.get("action_dim", 7)
        num_action_chunks = cfg.get("num_action_chunks", 16)
        hidden = cfg.get("fake_hidden_size", 64)
        dreamzero_config = DreamZeroConfig(
            backbone_cfg={
                "_target_": "rlinf.models.embodiment.dreamzero.fake_dreamzero_policy.FakeDreamZeroBackbone",
                "hidden": hidden,
            },
            action_head_cfg={
                "_target_": "rlinf.models.embodiment.dreamzero.fake_dreamzero_policy.FakeDreamZeroActionHead",
                "action_dim": action_dim,
                "action_horizon": num_action_chunks,
                "hidden": hidden,
                "dtype": str(torch_dtype).replace("torch.", "")
                if torch_dtype is not None
                else "float32",
            },
            action_horizon=num_action_chunks,
            action_dim=action_dim,
            env_action_dim=action_dim,
            num_action_chunks=num_action_chunks,
            num_steps=cfg.get("fake_num_steps", 4),
            data_transforms=FakeDreamZeroDataTransforms(),
            relative_action=False,
            relative_action_per_horizon=False,
            relative_action_keys=[],
        )

        model = DreamZeroPolicy(config=dreamzero_config)
        _promote_scalar_params_to_1d(model)
        model = model.to(dtype=torch_dtype)
        model.action_head.trt_engine = None
        return model

    from groot.vla.data.schema import DatasetMetadata
    from groot.vla.data.transform import ComposedModalityTransform
    from hydra.utils import instantiate

    from rlinf.models.embodiment.dreamzero.dreamzero_policy import (
        DreamZeroConfig,
        DreamZeroPolicy,
    )

    model_path = Path(cfg.get("model_path"))
    if not model_path.exists():
        raise FileNotFoundError(f"DreamZero model_path does not exist: {model_path}")

    tokenizer_path = cfg.get("tokenizer_path", "google/umt5-xxl")
    action_dim = cfg.get("action_dim", 7)

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Build clean DiT, load base weights, then inject LoRA with upstream defaults.
    dreamzero_config = DreamZeroConfig(**config_dict)
    dreamzero_config.action_head_cfg["_target_"] = (
        "rlinf.models.embodiment.dreamzero.dreamzero_action_head.DreamZeroActionHead"
    ) 
    action_head_cfg = dreamzero_config.action_head_cfg.get("config", {})
    # to build the num_steps for the nft worker
    dreamzero_config.num_steps = cfg.get(
        "num_steps",
        action_head_cfg.get("num_inference_steps", 4),
    )
    use_lora = cfg.get("use_lora", False)
    if use_lora:
        action_head_cfg["skip_component_loading"] = True
        action_head_cfg["train_architecture"] = "lora"
        action_head_cfg["defer_lora_injection"] = True

    dreamzero_config.env_action_dim = action_dim

    exp_cfg_dir = model_path / "experiment_cfg"
    metadata_path = exp_cfg_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadatas = json.load(f)

    embodiment_tag = cfg.get("embodiment_tag", "libero_sim")
    metadata = DatasetMetadata.model_validate(metadatas[embodiment_tag])

    train_cfg = OmegaConf.load(exp_cfg_dir / "conf.yaml")
    train_cfg.transforms[embodiment_tag].transforms[-1].tokenizer_path = tokenizer_path
    data_transforms = instantiate(train_cfg.transforms[embodiment_tag])
    assert isinstance(data_transforms, ComposedModalityTransform), f"{data_transforms=}"
    data_transforms.set_metadata(metadata)
    data_transforms.eval()

    dreamzero_config.data_transforms = data_transforms
    dreamzero_config.relative_action = train_cfg.get("relative_action", False)
    dreamzero_config.relative_action_per_horizon = train_cfg.get(
        "relative_action_per_horizon", False
    )
    dreamzero_config.relative_action_keys = train_cfg.get("relative_action_keys", [])
    model = DreamZeroPolicy(
        config=dreamzero_config,
    )
    # Stream safetensors and skip frozen encoders already loaded in
    # WANPolicyHead.__init__ (text_encoder / image_encoder / vae). Their
    # checkpoint copies are bit-exact bf16 casts of the fp32 pretrained
    # .pth files, so reloading them is redundant I/O (~15 GB).
    skip_prefixes = (
        "action_head.text_encoder.",
        "action_head.image_encoder.",
        "action_head.vae.",
    )
    st = model_path / "model.safetensors"
    st_index = model_path / "model.safetensors.index.json"
    if st_index.exists():
        with open(st_index, "r") as f:
            shard_files = sorted(set(json.load(f)["weight_map"].values()))
    elif st.exists():
        shard_files = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors weights under {model_path}")

    state_dict = {}
    for shard_file in shard_files:
        with safe_open(str(model_path / shard_file), framework="pt") as f:
            for k in f.keys():
                if k.startswith(skip_prefixes):
                    continue
                state_dict[k] = f.get_tensor(k)
    model.load_state_dict(state_dict, strict=False)

    if use_lora:
        model.action_head.inject_lora_after_loading()

    _promote_scalar_params_to_1d(model)
    model = model.to(dtype=torch_dtype)
    model.action_head.trt_engine = None
    return model
