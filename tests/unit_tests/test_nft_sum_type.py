"""Unit tests for NFT loss aggregation across sum types."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy


class _FakeNFTModel(torch.nn.Module):
    """Deterministic CPU model used to exercise NFT loss logic."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.25, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.tensor(-0.35, dtype=torch.float32))
        self.offset = torch.nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.config = SimpleNamespace(action_env_dim=7, num_steps=4)

    def forward(
        self,
        *,
        forward_type: ForwardType,
        forward_inputs: dict,
        nft_inputs: dict,
        compute_values: bool,
    ) -> dict:
        """Compute a stable synthetic NFT output."""
        assert forward_type == ForwardType.NFT
        assert compute_values is False

        x_t = nft_inputs["x_t"].to(dtype=self.scale.dtype)
        timesteps = nft_inputs["timesteps"].to(dtype=x_t.dtype).view(-1, 1, 1)
        obs = forward_inputs["obs"][..., : x_t.shape[2]].to(dtype=x_t.dtype)
        ctx = forward_inputs["nested"]["ctx"][..., : x_t.shape[2]].to(dtype=x_t.dtype)
        bias_term = forward_inputs["bias_term"][..., : x_t.shape[2]].to(
            dtype=x_t.dtype
        )
        v_theta = (
            x_t * self.scale
            + timesteps * 0.5
            + obs * 0.1
            + ctx * self.bias
            + bias_term
            + self.offset
        )
        return {"v_theta": v_theta}


def _make_tensor(
    base: float, row_scale: float, chunk_scale: float, dim_scale: float
) -> torch.Tensor:
    """Create a deterministic non-zero [3, 2, 32] tensor."""
    batch_idx = torch.arange(3, dtype=torch.float32).view(3, 1, 1)
    chunk_idx = torch.arange(2, dtype=torch.float32).view(1, 2, 1)
    dim_idx = torch.arange(32, dtype=torch.float32).view(1, 1, 32)
    return base + batch_idx * row_scale + chunk_idx * chunk_scale + dim_idx * dim_scale


def _build_batch() -> dict:
    """Build a fixed NFT batch for sum-type tests."""
    return {
        "forward_inputs": {
            "obs": _make_tensor(0.2, 1.2, 0.6, 0.07),
            "nested": {"ctx": _make_tensor(0.1, 1.1, 0.5, 0.05)},
            "bias_term": _make_tensor(0.05, -0.01, -0.02, 0.01),
            "nft_xcur": _make_tensor(0.3, 1.2, 0.6, 0.09),
            "nft_v": _make_tensor(0.15, 0.6, 0.3, 0.04),
            "nft_x0": _make_tensor(0.1, 0.6, 0.3, 0.03),
            "nft_xnext": _make_tensor(0.35, 1.2, 0.6, 0.08),
            "nft_noise_level": torch.zeros(3, dtype=torch.float32),
            "nft_step_index": torch.tensor([0, 1, 3], dtype=torch.long),
        },
        "loss_mask": torch.tensor([[True], [False], [True]], dtype=torch.bool),
        "advantages": torch.tensor([[0.8], [0.4], [-0.3]], dtype=torch.float32),
    }


def _build_worker(sum_type: str) -> EmbodiedNFTFSDPPolicy:
    """Construct a minimal worker instance for direct loss-function testing."""
    worker = object.__new__(EmbodiedNFTFSDPPolicy)
    worker.cfg = SimpleNamespace(
        algorithm={
            "adv_clip_max": 1.0,
            "max_drift": 0.4,
            "nft_beta": 0.8,
            "dpo_beta": 1.3,
            "nft_weight_scale": 2.0,
            "nft_target_space": "xnext",
            "nft_weight_mode": "constant",
            "nft_loss_form": "dpo",
            "nft_sum_type": sum_type,
        }
    )
    worker.model = _FakeNFTModel()
    worker.amp_context = contextlib.nullcontext()
    return worker


@pytest.mark.parametrize("sum_type", ["action_level", "chunk_level"])
def test_nft_forward_and_loss_supports_sum_types(sum_type: str) -> None:
    """NFT loss should run for both action-level and chunk-level aggregation."""
    worker = _build_worker(sum_type)
    loss, metrics = worker.nft_forward_and_loss(_build_batch())

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["actor/nft_loss"] == pytest.approx(float(loss))
