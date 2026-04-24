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

"""Full-model DreamZero forward alignment check against the original action head."""

from __future__ import annotations

import copy
import gc
import os
import time
from pathlib import Path

import pytest
import torch
from tianshou.data import Batch
from test_dreamzero_full_model_alignment import (
    LOCAL_ACTION_HEAD_TARGET,
    ORIGINAL_ACTION_HEAD_TARGET,
    _load_dreamzero_policy,
    _make_env_obs,
    _path_from_default_eval_config,
    _required_env,
)


def _make_training_action_input(
    model,
    env_obs: dict,
    num_frames: int = 9,
) -> dict:
    converted_obs = model._observation_convert(env_obs)
    batch = Batch(obs=converted_obs)
    normalized_input = model._process_batch(batch)
    if isinstance(normalized_input, Batch):
        normalized_input = normalized_input.__getstate__()

    training_input = {}
    for key, value in normalized_input.items():
        if torch.is_tensor(value):
            training_input[key] = value.clone()
        else:
            training_input[key] = copy.deepcopy(value)

    images = training_input["images"]
    if images.shape[1] != num_frames:
        training_input["images"] = images.repeat(1, num_frames, 1, 1, 1)

    batch_size = training_input["images"].shape[0]
    action_head = model.action_head
    state_steps = training_input["state"].shape[1]
    latent_frames = 1 + state_steps * (
        action_head.num_frame_per_block // action_head.model.num_state_per_block
    )
    action_dim = action_head.model.action_dim
    action_steps = (latent_frames - 1) * (
        action_head.model.num_action_per_block // action_head.num_frame_per_block
    )
    training_input["action"] = torch.linspace(
        -0.5,
        0.5,
        steps=batch_size * action_steps * action_dim,
        dtype=torch.float32,
    ).reshape(batch_size, action_steps, action_dim)
    training_input["has_real_action"] = torch.ones(batch_size, dtype=torch.float32)
    training_input["action_mask"] = torch.ones(
        batch_size,
        action_steps,
        action_dim,
        dtype=torch.float32,
    )
    return training_input


def _run_training_forward(model, normalized_input: dict) -> dict[str, torch.Tensor]:
    backbone_inputs, action_inputs = model.prepare_input(normalized_input)
    backbone_outputs = model.backbone(backbone_inputs)
    output = model.action_head.forward(backbone_outputs, action_inputs)
    return {key: value.detach().float().cpu() for key, value in output.items()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="DreamZero full-model test requires CUDA.")
def test_dreamzero_full_model_matches_original_forward_step() -> None:
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
    training_input = _make_training_action_input(local_model, env_obs)
    torch.manual_seed(1234)
    local_output = _run_training_forward(local_model, training_input)
    del local_model
    torch.cuda.empty_cache()
    gc.collect()

    original_model = _load_dreamzero_policy(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        embodiment_tag=embodiment_tag,
        action_head_target=ORIGINAL_ACTION_HEAD_TARGET,
    )
    torch.manual_seed(1234)
    original_output = _run_training_forward(original_model, training_input)
    del original_model
    torch.cuda.empty_cache()
    gc.collect()

    for key in ("loss", "dynamics_loss", "action_loss"):
        torch.testing.assert_close(
            local_output[key],
            original_output[key],
            rtol=1e-3,
            atol=1e-3,
        )
    elapsed = time.perf_counter() - start_time
    print(f"[ALIGN] full-model forward alignment test passed in {elapsed:.2f}s", flush=True)


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available.")
        return 0
    test_dreamzero_full_model_matches_original_forward_step()
    print("OK: full DreamZero model matches original forward step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
