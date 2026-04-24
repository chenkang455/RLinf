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

"""Lightweight consistency check for the DreamZero action-head refactor.

This test avoids constructing the full DreamZero model. It compares:

1. The original third-party ``WANPolicyHead.lazy_joint_video_action``.
2. The RLinf refactored ``DreamZeroActionHead.lazy_joint_video_action``.

Heavy components are replaced with deterministic test doubles so that we can
focus on control flow and tensor plumbing, while still asserting that the
refactored action head preserves the original first-block ``video_pred``
prefixing semantics.

Run:
    python tests/unit_tests/test_dreamzero_action_head_refactor.py
    pytest tests/unit_tests/test_dreamzero_action_head_refactor.py
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from transformers.feature_extraction_utils import BatchFeature


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeCudaEvent:
    def __init__(self, enable_timing: bool = True):
        self.enable_timing = enable_timing

    def record(self) -> None:
        return None

    def elapsed_time(self, other: "_FakeCudaEvent") -> float:
        del other
        return 0.0


class _FakeScheduler:
    def __init__(self, num_train_timesteps: int, shift: int, use_dynamic_shifting: bool):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.use_dynamic_shifting = use_dynamic_shifting
        self.timesteps = torch.arange(num_train_timesteps, dtype=torch.float32)
        self.sigmas = torch.empty(0, dtype=torch.float32)

    def set_timesteps(self, num_inference_steps: int, device: torch.device | str | None = None, shift: float | None = None) -> None:
        del device, shift
        self.timesteps = torch.arange(num_inference_steps, 0, -1, dtype=torch.int64)
        self.sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float32)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | int,
        sample: torch.Tensor,
        step_index: int,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor]:
        del timestep, return_dict
        updated = sample - 0.1 * model_output + (step_index + 1) * 0.001
        return (updated,)

    def add_noise(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        scale = timestep.to(torch.float32).reshape(-1, *([1] * (sample.ndim - 1))) / max(
            float(self.num_train_timesteps), 1.0
        )
        return sample + noise * scale.to(sample.dtype)

    def training_target(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        del sample, timestep
        return noise

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(timestep, dtype=torch.float32)


class _FakeTrainingModel:
    def __init__(self) -> None:
        self.action_dim = 3
        self.dim = 16
        self.num_layers = 2
        self.num_heads = 2
        self.local_attn_size = 16
        self.num_action_per_block = 2
        self.num_state_per_block = 1

    def __call__(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        clip_feature: torch.Tensor,
        y: torch.Tensor,
        context,
        seq_len: int,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        clean_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del seq_len, embodiment_id, clean_x
        if torch.is_tensor(context):
            prompt_term = context.float().mean()
        else:
            prompt_term = sum(prompt.float().mean() for prompt in context) / max(len(context), 1)
        video_pred = (
            noisy_latents.float() * 0.5
            + timestep.float().mean() * 0.001
            + clip_feature.float().mean() * 0.01
            + y.float().mean() * 0.01
            + state.float().mean() * 0.01
            + prompt_term * 0.02
        )
        action_pred = (
            action.float() * 0.25
            + timestep_action.float().mean() * 0.001
            + state.float().mean() * 0.01
            + prompt_term * 0.02
        )
        return video_pred.to(noisy_latents.dtype), action_pred.to(action.dtype)


class _FakeImageEncoder:
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        return torch.ones((batch_size, 1, 2), dtype=torch.bfloat16)


class _FakeVAE:
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_frames, height, width = video.shape
        return torch.ones(
            (batch_size, 48, num_frames, height // 16, width // 16),
            dtype=torch.bfloat16,
        )


def _repo_root() -> Path:
    return REPO_ROOT


def _load_modules():
    repo_root = _repo_root()

    original_module = importlib.import_module(
        "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf"
    )

    local_path = repo_root / "rlinf" / "models" / "embodiment" / "dreamzero" / "dreamzero_action_head.py"
    spec = importlib.util.spec_from_file_location("rlinf_dreamzero_action_head_local", local_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load local module from {local_path}")
    local_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_module)
    return local_module, original_module


def _install_test_doubles(local_module, original_module) -> None:
    local_module.FlowUniPCMultistepScheduler = _FakeScheduler
    original_module.FlowUniPCMultistepScheduler = _FakeScheduler
    original_module.torch.cuda.Event = _FakeCudaEvent
    original_module.torch.cuda.synchronize = lambda: None


def _build_stub_head(cls, base_cls):
    head = cls.__new__(cls)
    torch.nn.Module.__init__(head)
    head.training = False
    head.register_parameter(
        "_dtype_anchor_param",
        torch.nn.Parameter(torch.empty(0, dtype=torch.float32), requires_grad=False),
    )
    head.tiled = True
    head.tile_size_height = 34
    head.tile_size_width = 34
    head.tile_stride_height = 18
    head.tile_stride_width = 16
    head.num_frame_per_block = 1
    head.num_frames = 1
    head.hidden_size = 8
    head.action_horizon = 2
    head.num_inference_steps = 4
    head.seed = 7
    head.cfg_scale = 5.0
    head.denoising_strength = 1.0
    head.sigma_shift = 5.0
    head.scheduler = _FakeScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    head.config = SimpleNamespace(
        decouple_inference_noise=False,
        video_inference_final_noise=0.8,
        decouple_video_action_noise=False,
        use_high_noise_emphasis=False,
    )
    head.model = _FakeTrainingModel()
    head.ip_rank = 0
    head.ip_size = 1
    head.ip_group = None
    head.dynamic_cache_schedule = False
    head.dit_step_mask = [True] * head.num_inference_steps
    head.skip_countdown = 0
    head._device = "cpu"
    head.trt_engine = None
    head.language = None
    head.current_start_frame = 0
    head.clip_feas = None
    head.ys = None
    head.kv_cache1 = None
    head.kv_cache_neg = None
    head.crossattn_cache = None
    head.crossattn_cache_neg = None
    head.normalize_video = lambda videos: videos * 2.0 - 1.0
    head._debug_forward_count = 0
    head._noise_logged = False

    def set_frozen_modules_to_eval_mode(self) -> None:
        del self
        return None

    def encode_prompt(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        prompt = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 2)
        return (prompt * attention_mask.unsqueeze(-1)).to(torch.bfloat16)

    def encode_image(
        self,
        image: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del self, height, width
        batch_size = image.shape[0]
        image_input = image.transpose(1, 2).to(torch.float32)
        image_latents = image_input.mean(dim=(1, 3, 4), keepdim=True).repeat(1, 16, 1, 1, 1)
        clip_feas = image_latents.flatten(2).mean(dim=-1, keepdim=True)
        clip_feas = clip_feas.repeat(1, 1, 2)
        ys = image_latents.repeat(1, 2, num_frames, 1, 1)
        return clip_feas.to(torch.bfloat16), ys.to(torch.bfloat16), image_latents.to(torch.bfloat16)

    def generate_noise(
        self,
        shape: tuple[int, ...],
        seed: int | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        del self, device
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        return torch.randn(shape, generator=generator, dtype=dtype)

    def encode_video(
        self,
        videos: torch.Tensor,
        tiled: bool,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
    ) -> torch.Tensor:
        del self, tiled, tile_size, tile_stride
        video_btchw = videos.transpose(1, 2).to(torch.float32)
        latents = video_btchw.mean(dim=(2, 3, 4), keepdim=True).repeat(1, 1, 16, 1, 1)
        return latents.transpose(1, 2).to(torch.bfloat16)

    def _run_diffusion_steps(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        context: list[torch.Tensor],
        seq_len: int,
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        kv_caches,
        crossattn_caches,
        kv_cache_metadata,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        del self, timestep_action, embodiment_id, seq_len, kv_caches, crossattn_caches, kv_cache_metadata
        predictions = []
        base = noisy_input.to(torch.float32) * 0.5
        state_term = 0.0 if state is None else state.to(torch.float32).mean()
        action_base = None if action is None else action.to(torch.float32) * 0.25
        timestep_term = timestep.to(torch.float32).mean() * 0.001
        y_term = y.to(torch.float32).mean() * 0.01
        clip_term = clip_feature.to(torch.float32).mean() * 0.01
        for prompt_emb in context:
            prompt_term = prompt_emb.to(torch.float32).mean() * 0.02
            obs_pred = base + state_term + timestep_term + y_term + clip_term + prompt_term
            if action_base is None:
                action_pred = torch.tensor(0.0, dtype=torch.float32)
            else:
                action_pred = action_base + state_term + timestep_term + prompt_term
            predictions.append((obs_pred.to(noisy_input.dtype), action_pred.to(torch.bfloat16)))
        return predictions

    head.set_frozen_modules_to_eval_mode = MethodType(set_frozen_modules_to_eval_mode, head)
    head.encode_prompt = MethodType(encode_prompt, head)
    head.encode_image = MethodType(encode_image, head)
    head.encode_video = MethodType(encode_video, head)
    head.generate_noise = MethodType(generate_noise, head)
    head._run_diffusion_steps = MethodType(_run_diffusion_steps, head)
    head._prepare_text_inputs = MethodType(base_cls._prepare_text_inputs, head)
    head._create_kv_caches = MethodType(base_cls._create_kv_caches, head)
    head._create_crossattn_caches = MethodType(base_cls._create_crossattn_caches, head)
    head._get_caches = MethodType(base_cls._get_caches, head)
    head.should_run_model = MethodType(base_cls.should_run_model, head)
    return head


def _make_action_input() -> BatchFeature:
    images = torch.arange(1 * 1 * 8 * 8 * 3, dtype=torch.uint8).reshape(1, 1, 8, 8, 3)
    return BatchFeature(
        data={
            "images": images,
            "state": torch.tensor([[0.25, -0.5, 1.0, 2.0]], dtype=torch.float32),
            "text": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
            "text_attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            "text_negative": torch.tensor([[4, 5, 0, 0]], dtype=torch.long),
            "text_attention_mask_negative": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
            "embodiment_id": torch.tensor([2], dtype=torch.long),
        }
    )


def _make_training_action_input() -> BatchFeature:
    images = torch.arange(1 * 3 * 8 * 8 * 3, dtype=torch.uint8).reshape(1, 3, 8, 8, 3)
    return BatchFeature(
        data={
            "images": images,
            "state": torch.tensor([[0.25, -0.5]], dtype=torch.float32),
            "text": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
            "text_attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            "text_negative": torch.tensor([[4, 5, 0, 0]], dtype=torch.long),
            "text_attention_mask_negative": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
            "embodiment_id": torch.tensor([2], dtype=torch.long),
            "action": torch.linspace(-0.6, 0.6, steps=1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3),
            "has_real_action": torch.tensor([1.0], dtype=torch.float32),
            "action_mask": torch.ones((1, 4, 3), dtype=torch.float32),
        }
    )


def test_dreamzero_action_head_refactor_matches_original_first_block() -> None:
    local_module, original_module = _load_modules()
    _install_test_doubles(local_module, original_module)

    action_input = _make_action_input()
    local_head = _build_stub_head(
        local_module.DreamZeroActionHead,
        original_module.WANPolicyHead,
    )
    original_head = _build_stub_head(
        original_module.WANPolicyHead,
        original_module.WANPolicyHead,
    )

    refactored_output = local_head.lazy_joint_video_action(
        backbone_output=BatchFeature(data={}),
        action_input=action_input,
    )
    original_output = original_head.lazy_joint_video_action(
        backbone_output=BatchFeature(data={}),
        action_input=action_input,
    )

    torch.testing.assert_close(
        refactored_output["action_pred"].float(),
        original_output["action_pred"].float(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        refactored_output["video_pred"].float(),
        original_output["video_pred"].float(),
        rtol=0.0,
        atol=0.0,
    )


def test_dreamzero_action_head_forward_interface() -> None:
    local_module, original_module = _load_modules()
    _install_test_doubles(local_module, original_module)

    action_input = _make_training_action_input()
    local_head = _build_stub_head(
        local_module.DreamZeroActionHead,
        original_module.WANPolicyHead,
    )
    original_head = _build_stub_head(
        original_module.WANPolicyHead,
        original_module.WANPolicyHead,
    )

    torch.manual_seed(1234)
    refactored_output = local_head.forward(
        backbone_output=BatchFeature(data={}),
        action_input=action_input,
    )
    torch.manual_seed(1234)
    original_output = original_head.forward(
        backbone_output=BatchFeature(data={}),
        action_input=action_input,
    )

    assert set(refactored_output.keys()) == {"loss", "dynamics_loss", "action_loss"}
    for key in ("loss", "dynamics_loss", "action_loss"):
        value = refactored_output[key]
        assert torch.is_tensor(value)
        assert value.ndim == 0
        assert torch.isfinite(value)
        torch.testing.assert_close(
            refactored_output[key].float(),
            original_output[key].float(),
            rtol=0.0,
            atol=0.0,
        )
    assert torch.allclose(
        refactored_output["loss"],
        refactored_output["dynamics_loss"] + refactored_output["action_loss"],
    )


def test_dreamzero_action_head_local_5b_shape_overrides() -> None:
    local_module, original_module = _load_modules()
    _install_test_doubles(local_module, original_module)

    local_head = _build_stub_head(
        local_module.DreamZeroActionHead,
        original_module.WANPolicyHead,
    )
    local_head.model.num_heads = 24
    local_head.model.dim = 1536
    local_head.model.frame_seqlen = 50
    local_head.config.target_video_height = 160
    local_head.config.target_video_width = 320
    local_head.image_encoder = _FakeImageEncoder()
    local_head.vae = _FakeVAE()
    local_head._ensure_vae_on_device = MethodType(lambda self, ref: None, local_head)

    kv_cache1, kv_cache_neg = local_module.DreamZeroActionHead._create_kv_caches(
        local_head,
        batch_size=2,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        frame_seqlen=0,
    )
    assert kv_cache1[0].shape == (2, 2, 0, 24, 64)
    assert kv_cache_neg[0].shape == (2, 2, 0, 24, 64)

    cross_cache, cross_cache_neg = local_module.DreamZeroActionHead._create_crossattn_caches(
        local_head,
        batch_size=2,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert cross_cache[0].shape == (2, 2, 512, 24, 64)
    assert cross_cache_neg[0].shape == (2, 2, 512, 24, 64)

    videos = torch.zeros((1, 3, 2, 256, 256), dtype=torch.bfloat16)
    resized = local_module.DreamZeroActionHead._resize_videos_for_model(local_head, videos)
    assert resized.shape == (1, 3, 2, 160, 320)

    action_input = BatchFeature(
        data={
            "images": torch.zeros((1, 2, 256, 256, 3), dtype=torch.uint8),
            "state": torch.zeros((1, 2), dtype=torch.float32),
            "text": torch.tensor([[1, 2, 0]], dtype=torch.long),
            "text_attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
            "text_negative": torch.tensor([[3, 0, 0]], dtype=torch.long),
            "text_attention_mask_negative": torch.tensor([[1, 0, 0]], dtype=torch.long),
            "embodiment_id": torch.tensor([2], dtype=torch.long),
        }
    )
    policy_inputs = local_head.prepare_policy_inputs(action_input, mode="inference")
    # The current refactor keeps resize as an explicit helper; this test only
    # verifies the helper contracts without assuming it is wired into the
    # standard inference preprocessing path yet.
    assert policy_inputs["videos"].shape == (1, 3, 2, 256, 256)
    assert policy_inputs["image_latents"].shape == (1, 16, 1, 1, 1)

    local_head.encode_image = MethodType(
        lambda self, image, num_frames, height, width: (
            torch.ones((image.shape[0], 1, 2), dtype=torch.bfloat16),
            torch.ones((image.shape[0], 52, num_frames, 10, 20), dtype=torch.bfloat16),
            torch.ones((image.shape[0], 48, 1, 10, 20), dtype=torch.bfloat16),
        ),
        local_head,
    )
    policy_inputs = local_head.prepare_policy_inputs(action_input, mode="inference")
    sampling_state = local_head.prepare_noise_action_video(policy_inputs)
    assert policy_inputs["image_latents"].shape == (1, 48, 1, 10, 20)
    assert sampling_state["noise_obs"].shape == (1, 1, 48, 10, 20)


def main() -> int:
    test_dreamzero_action_head_refactor_matches_original_first_block()
    test_dreamzero_action_head_forward_interface()
    test_dreamzero_action_head_local_5b_shape_overrides()
    print("OK: DreamZero action-head refactor matches eval and forward interfaces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
