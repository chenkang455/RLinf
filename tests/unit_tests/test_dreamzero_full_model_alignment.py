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

"""Full-model DreamZero eval alignment check against the original action head.

This test is intentionally opt-in because it loads the real DreamZero model
checkpoint and runs a causal inference step on GPU.

Environment variables:
    DREAMZERO_FULL_MODEL_PATH: Path to a full DreamZero checkpoint directory.
    DREAMZERO_FULL_TOKENIZER_PATH: Tokenizer path used by the checkpoint.

Optional environment variables:
    DREAMZERO_FULL_EMBODIMENT_TAG: Dataset metadata key. Defaults to ``libero_sim``.

If the model/tokenizer env vars are unset, the test falls back to
``examples/embodiment/config/libero_spatial_eval_dreamzero.yaml``.

Run:
    DREAMZERO_FULL_MODEL_PATH=/path/to/ckpt \
    DREAMZERO_FULL_TOKENIZER_PATH=/path/to/tokenizer \
    pytest tests/unit_tests/test_dreamzero_full_model_alignment.py -s
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

# This test validates output parity, not compiler behavior. Disable torch.compile
# before importing DreamZero modules so scheduler decorators pick it up.
os.environ.setdefault("DREAMZERO_SKIP_TORCH_COMPILE", "true")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _repo_root() -> Path:
    return REPO_ROOT


def _ensure_third_party_path() -> None:
    return None


_ensure_third_party_path()

import numpy as np
import pytest
import torch

pytest.importorskip("albumentations", reason="DreamZero full-model test requires albumentations.")
pytest.importorskip("tianshou", reason="DreamZero full-model test requires tianshou.")

from groot.vla.data.schema import DatasetMetadata
from groot.vla.data.transform import ComposedModalityTransform
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors.torch import load_file
from tianshou.data import Batch

from rlinf.models.embodiment.dreamzero.dreamzero_policy import (
    DreamZeroConfig,
    DreamZeroPolicy,
)


LOCAL_ACTION_HEAD_TARGET = (
    "rlinf.models.embodiment.dreamzero.dreamzero_action_head.DreamZeroActionHead"
)
ORIGINAL_ACTION_HEAD_TARGET = (
    "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead"
)
DEFAULT_DREAMZERO_EVAL_CONFIG = _repo_root() / "examples" / "embodiment" / "config" / "libero_spatial_eval_dreamzero.yaml"


def _path_from_default_eval_config(name: str) -> str | None:
    if not DEFAULT_DREAMZERO_EVAL_CONFIG.exists():
        return None

    cfg = OmegaConf.load(DEFAULT_DREAMZERO_EVAL_CONFIG)
    if name == "DREAMZERO_FULL_MODEL_PATH":
        value = cfg.actor.model.model_path
    elif name == "DREAMZERO_FULL_TOKENIZER_PATH":
        value = cfg.actor.model.tokenizer_path
    elif name == "DREAMZERO_FULL_EMBODIMENT_TAG":
        value = cfg.actor.model.embodiment_tag
    else:
        value = None
    return None if value in (None, "") else str(value)


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        value = _path_from_default_eval_config(name)
    if not value:
        pytest.skip(
            f"{name} is not set and no default path was found in "
            f"{DEFAULT_DREAMZERO_EVAL_CONFIG}; skipping full DreamZero alignment test."
        )
    return value


def _load_dreamzero_policy(
    model_path: Path,
    tokenizer_path: str,
    embodiment_tag: str,
    action_head_target: str,
) -> DreamZeroPolicy:
    start_time = time.perf_counter()
    print(f"[ALIGN] loading policy target={action_head_target}", flush=True)
    if not model_path.exists():
        raise FileNotFoundError(f"DreamZero model_path does not exist: {model_path}")

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    dreamzero_config = DreamZeroConfig(**config_dict)
    dreamzero_config.action_head_cfg["_target_"] = action_head_target
    if "config" in dreamzero_config.action_head_cfg and isinstance(
        dreamzero_config.action_head_cfg["config"], dict
    ):
        dreamzero_config.action_head_cfg["config"]["defer_lora_injection"] = False
        dreamzero_config.action_head_cfg["config"]["skip_component_loading"] = True

    exp_cfg_dir = model_path / "experiment_cfg"
    metadata_path = exp_cfg_dir / "metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    metadata = DatasetMetadata.model_validate(metadatas[embodiment_tag])
    train_cfg = OmegaConf.load(exp_cfg_dir / "conf.yaml")
    train_cfg.transforms[embodiment_tag].transforms[-1].tokenizer_path = tokenizer_path
    data_transforms = instantiate(train_cfg.transforms[embodiment_tag])
    assert isinstance(data_transforms, ComposedModalityTransform), f"{data_transforms=}"
    data_transforms.set_metadata(metadata)
    data_transforms.eval()

    dreamzero_config.data_transforms = data_transforms
    dreamzero_config.env_action_dim = dreamzero_config.action_dim
    dreamzero_config.relative_action = train_cfg.get("relative_action", False)
    dreamzero_config.relative_action_per_horizon = train_cfg.get(
        "relative_action_per_horizon", False
    )
    dreamzero_config.relative_action_keys = train_cfg.get("relative_action_keys", [])

    model = DreamZeroPolicy(config=dreamzero_config)

    state_dict: dict[str, Any] = {}
    st = model_path / "model.safetensors"
    st_index = model_path / "model.safetensors.index.json"
    if st_index.exists():
        with open(st_index, "r", encoding="utf-8") as f:
            index = json.load(f)
        for shard_file in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(str(model_path / shard_file)))
    elif st.exists():
        state_dict.update(load_file(str(st)))
    else:
        raise FileNotFoundError(f"No safetensors weights under {model_path}")

    if any(".base_layer." in key for key in state_dict):
        state_dict = {
            key.replace(".base_layer.", "."): value for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=False)

    prev_enable_trt = os.environ.get("ENABLE_TENSORRT")
    os.environ["ENABLE_TENSORRT"] = "true"
    try:
        if hasattr(model, "post_initialize"):
            model.post_initialize()
    finally:
        if prev_enable_trt is None:
            os.environ.pop("ENABLE_TENSORRT", None)
        else:
            os.environ["ENABLE_TENSORRT"] = prev_enable_trt

    model.eval()
    elapsed = time.perf_counter() - start_time
    print(f"[ALIGN] loaded policy target={action_head_target} in {elapsed:.2f}s", flush=True)
    return model


def _make_env_obs() -> dict[str, Any]:
    rng = np.random.default_rng(seed=0)
    return {
        "main_images": rng.integers(0, 255, size=(1, 256, 256, 3), dtype=np.uint8),
        "wrist_images": rng.integers(0, 255, size=(1, 256, 256, 3), dtype=np.uint8),
        # Match the debug script and DreamZero libero_sim metadata: the state
        # transform for this checkpoint expects an 8D proprio vector.
        "states": np.zeros((1, 8), dtype=np.float32),
        "task_descriptions": [
            "pick up the black bowl and place it on the plate"
        ],
    }


def _run_causal_step(model: DreamZeroPolicy, env_obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    start_time = time.perf_counter()
    print(
        f"[ALIGN] running causal step with action_head={type(model.action_head).__name__}",
        flush=True,
    )
    converted_obs = model._observation_convert(env_obs)
    batch = Batch(obs=converted_obs)
    normalized_input = model._process_batch(batch)
    with torch.inference_mode():
        output = model.lazy_joint_video_action_causal(normalized_input)
    action_pred = output["action_pred"].detach().float().cpu()
    video_pred = output["video_pred"].detach().float().cpu()
    elapsed = time.perf_counter() - start_time
    print(
        f"[ALIGN] finished causal step with action_head={type(model.action_head).__name__} in {elapsed:.2f}s",
        flush=True,
    )
    return action_pred, video_pred

def _normalize_video_for_compare(video_pred: torch.Tensor) -> torch.Tensor:
    # The original first-block path prepends the reference latent frame.
    if video_pred.ndim == 5 and video_pred.shape[2] > 1:
        return video_pred[:, :, 1:]
    return video_pred


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DreamZero full-model test requires CUDA.")
def test_dreamzero_full_model_matches_original_causal_first_step() -> None:
    start_time = time.perf_counter()
    model_path = Path(_required_env("DREAMZERO_FULL_MODEL_PATH"))
    tokenizer_path = _required_env("DREAMZERO_FULL_TOKENIZER_PATH")
    embodiment_tag = os.getenv("DREAMZERO_FULL_EMBODIMENT_TAG") or _path_from_default_eval_config(
        "DREAMZERO_FULL_EMBODIMENT_TAG"
    ) or "libero_sim"
    env_obs = _make_env_obs()

    local_model = _load_dreamzero_policy(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        embodiment_tag=embodiment_tag,
        action_head_target=LOCAL_ACTION_HEAD_TARGET,
    )
    local_action_pred, local_video_pred = _run_causal_step(local_model, env_obs)
    del local_model
    torch.cuda.empty_cache()
    gc.collect()

    original_model = _load_dreamzero_policy(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        embodiment_tag=embodiment_tag,
        action_head_target=ORIGINAL_ACTION_HEAD_TARGET,
    )
    original_action_pred, original_video_pred = _run_causal_step(original_model, env_obs)
    del original_model
    torch.cuda.empty_cache()
    gc.collect()

    torch.testing.assert_close(
        local_action_pred,
        original_action_pred,
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        _normalize_video_for_compare(local_video_pred),
        _normalize_video_for_compare(original_video_pred),
        rtol=1e-3,
        atol=1e-3,
    )
    elapsed = time.perf_counter() - start_time
    print(f"[ALIGN] full-model alignment test passed in {elapsed:.2f}s", flush=True)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available.")
        return 0
    test_dreamzero_full_model_matches_original_causal_first_step()
    print("OK: full DreamZero model matches original eval step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
