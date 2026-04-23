#!/usr/bin/env python3
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

import argparse
import contextlib
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from rlinf.models.embodiment.base_policy import ForwardType

DEFAULT_BASELINE_COMMIT = "5419bf56e7fafc3bbf3b81dd10daf77bc9fec988"
NFT_WORKER_RELATIVE_PATH = Path("rlinf/workers/actor/fsdp_nft_policy_worker.py")


class FakeNFTModel(torch.nn.Module):
    """Deterministic CPU model used to compare NFT loss implementations."""

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
        bias_term = forward_inputs["bias_term"][..., : x_t.shape[2]].to(dtype=x_t.dtype)
        v_theta = (
            x_t * self.scale
            + timesteps * 0.5
            + obs * 0.1
            + ctx * self.bias
            + bias_term
            + self.offset
        )
        return {"v_theta": v_theta}


def _load_module_from_path(module_name: str, module_path: Path):
    """Import a Python module from an explicit file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_baseline_module(repo_root: Path, commit: str):
    """Load the baseline NFT worker module from a git commit into a temp file."""
    file_text = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo_root),
            "show",
            f"{commit}:{NFT_WORKER_RELATIVE_PATH.as_posix()}",
        ],
        text=True,
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix="_baseline_fsdp_nft_policy_worker.py", delete=False
    ) as temp_file:
        temp_file.write(file_text)
        temp_path = Path(temp_file.name)
    module = _load_module_from_path("baseline_fsdp_nft_policy_worker", temp_path)
    temp_path.unlink(missing_ok=True)
    return module


def _clone_nested(value):
    """Recursively clone tensors to avoid in-place sharing across comparisons."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    return value


def _make_tensor(base: float, row_scale: float, chunk_scale: float, dim_scale: float) -> torch.Tensor:
    """Create a deterministic non-zero [3, 2, 32] tensor for NFT regression tests."""
    batch_idx = torch.arange(3, dtype=torch.float32).view(3, 1, 1)
    chunk_idx = torch.arange(2, dtype=torch.float32).view(1, 2, 1)
    dim_idx = torch.arange(32, dtype=torch.float32).view(1, 1, 32)
    return base + batch_idx * row_scale + chunk_idx * chunk_scale + dim_idx * dim_scale


def _build_batch(*, target_space: str, noise_level: torch.Tensor) -> dict:
    """Build a fixed NFT batch for regression comparisons."""
    forward_inputs = {
        "obs": _make_tensor(0.2, 1.2, 0.6, 0.07),
        "nested": {
            "ctx": _make_tensor(0.1, 1.1, 0.5, 0.05)
        },
        "bias_term": _make_tensor(0.05, -0.01, -0.02, 0.01),
        "nft_xcur": _make_tensor(0.3, 1.2, 0.6, 0.09),
        "nft_v": _make_tensor(0.15, 0.6, 0.3, 0.04),
        "nft_x0": _make_tensor(0.1, 0.6, 0.3, 0.03),
        "nft_xnext": _make_tensor(0.35, 1.2, 0.6, 0.08),
        "nft_noise_level": noise_level,
        "nft_step_index": torch.tensor([0, 1, 3], dtype=torch.long),
    }
    if target_space not in {"x0", "xnext"}:
        raise ValueError(f"Unsupported target_space: {target_space}")

    return {
        "forward_inputs": forward_inputs,
        "loss_mask": torch.tensor([[True], [False], [True]], dtype=torch.bool),
        "advantages": torch.tensor([[0.8], [0.4], [-0.3]], dtype=torch.float32),
    }


def _build_worker(worker_cls, algorithm_cfg: dict):
    """Construct a minimal worker instance for direct loss-function testing."""
    worker = object.__new__(worker_cls)
    worker.cfg = SimpleNamespace(algorithm=algorithm_cfg)
    worker.model = FakeNFTModel()
    worker.amp_context = contextlib.nullcontext()
    return worker


def _run_case(module, algorithm_cfg: dict, batch: dict):
    """Execute nft_forward_and_loss on a worker module with cloned inputs."""
    worker = _build_worker(module.EmbodiedNFTFSDPPolicy, algorithm_cfg)
    try:
        loss, metrics = worker.nft_forward_and_loss(_clone_nested(batch))
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "type": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "status": "ok",
        "loss": float(loss.detach().cpu().item()),
        "metrics": metrics,
    }


def _assert_close(case_name: str, label: str, lhs: float, rhs: float, atol: float, rtol: float) -> None:
    """Raise a detailed assertion if two scalars differ beyond tolerance."""
    if not torch.isclose(torch.tensor(lhs), torch.tensor(rhs), atol=atol, rtol=rtol):
        raise AssertionError(
            f"{case_name}: {label} mismatch, current={lhs:.10f}, baseline={rhs:.10f}"
        )


def _compare_case(
    case_name: str,
    algorithm_cfg: dict,
    batch: dict,
    current_module,
    baseline_module,
    atol: float,
    rtol: float,
) -> str:
    """Compare current and baseline nft_forward_and_loss results for one config."""
    current_result = _run_case(current_module, algorithm_cfg, batch)
    baseline_result = _run_case(baseline_module, algorithm_cfg, batch)

    if current_result["status"] != baseline_result["status"]:
        raise AssertionError(
            f"{case_name}: status mismatch, "
            f"current={current_result['status']}, "
            f"baseline={baseline_result['status']}"
        )

    if current_result["status"] == "error":
        if (
            current_result["type"] != baseline_result["type"]
            or current_result["message"] != baseline_result["message"]
        ):
            raise AssertionError(
                f"{case_name}: exception mismatch, "
                f"current=({current_result['type']}, {current_result['message']}), "
                f"baseline=({baseline_result['type']}, {baseline_result['message']})"
            )
        return f"matching exception {current_result['type']}"

    current_loss = current_result["loss"]
    baseline_loss = baseline_result["loss"]
    current_metrics = current_result["metrics"]
    baseline_metrics = baseline_result["metrics"]

    _assert_close(case_name, "loss", current_loss, baseline_loss, atol, rtol)
    if set(current_metrics) != set(baseline_metrics):
        raise AssertionError(
            f"{case_name}: metrics keys mismatch, "
            f"current={sorted(current_metrics)}, baseline={sorted(baseline_metrics)}"
        )

    for key in sorted(current_metrics):
        _assert_close(
            case_name,
            key,
            float(current_metrics[key]),
            float(baseline_metrics[key]),
            atol,
            rtol,
        )
    return "matching numeric outputs"


def _classify_result(result_summary: str) -> tuple[str, str]:
    """Map comparison summaries to user-facing status markers."""
    if result_summary.startswith("matching exception "):
        return "⚠️", "consistent exception"
    return "✅", "matching numeric outputs"


def _build_cases() -> list[tuple[str, dict, dict]]:
    """Create a config matrix that covers NFT loss branches."""
    common_cfg = {
        "adv_clip_max": 1.0,
        "max_drift": 0.4,
        "nft_beta": 0.8,
        "dpo_beta": 1.3,
        "nft_weight_scale": 2.0,
    }
    zero_noise = torch.zeros(3, dtype=torch.float32)
    nonzero_noise = torch.tensor([0.4, 0.2, 0.1], dtype=torch.float32)

    cases = [
        (
            "x0_constant_dpo",
            {
                **common_cfg,
                "nft_target_space": "x0",
                "nft_weight_mode": "constant",
                "nft_loss_form": "dpo",
            },
            _build_batch(target_space="x0", noise_level=zero_noise),
        ),
        (
            "x0_t_mse",
            {
                **common_cfg,
                "nft_target_space": "x0",
                "nft_weight_mode": "t",
                "nft_loss_form": "mse",
                "max_drift": 0.2,
            },
            _build_batch(target_space="x0", noise_level=zero_noise),
        ),
        (
            "x0_adaptive_dpo",
            {
                **common_cfg,
                "nft_target_space": "x0",
                "nft_weight_mode": "adaptive",
                "nft_loss_form": "dpo",
            },
            _build_batch(target_space="x0", noise_level=zero_noise),
        ),
        (
            "x0_auto_dpo",
            {
                **common_cfg,
                "nft_target_space": "x0",
                "nft_weight_mode": "auto",
                "nft_loss_form": "dpo",
            },
            _build_batch(target_space="x0", noise_level=zero_noise),
        ),
        (
            "xnext_sigma_mse",
            {
                **common_cfg,
                "nft_target_space": "xnext",
                "nft_weight_mode": "sigma",
                "nft_loss_form": "mse",
            },
            _build_batch(target_space="xnext", noise_level=nonzero_noise),
        ),
        (
            "xnext_auto_mse",
            {
                **common_cfg,
                "nft_target_space": "xnext",
                "nft_weight_mode": "auto",
                "nft_loss_form": "mse",
                "max_drift": 0.15,
            },
            _build_batch(target_space="xnext", noise_level=nonzero_noise),
        ),
        (
            "xnext_constant_dpo",
            {
                **common_cfg,
                "nft_target_space": "xnext",
                "nft_weight_mode": "constant",
                "nft_loss_form": "dpo",
            },
            _build_batch(target_space="xnext", noise_level=nonzero_noise),
        ),
        (
            "xnext_t_mse",
            {
                **common_cfg,
                "nft_target_space": "xnext",
                "nft_weight_mode": "t",
                "nft_loss_form": "mse",
            },
            _build_batch(target_space="xnext", noise_level=nonzero_noise),
        ),
    ]
    return cases


def main() -> int:
    """Compare current nft_forward_and_loss results against a baseline commit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default=DEFAULT_BASELINE_COMMIT)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    current_module = _load_module_from_path(
        "current_fsdp_nft_policy_worker", repo_root / NFT_WORKER_RELATIVE_PATH
    )
    baseline_module = _load_baseline_module(repo_root, args.commit)

    cases = _build_cases()
    failures: list[tuple[str, str]] = []
    for case_name, algorithm_cfg, batch in cases:
        try:
            result_summary = _compare_case(
                case_name,
                algorithm_cfg,
                batch,
                current_module,
                baseline_module,
                args.atol,
                args.rtol,
            )
        except AssertionError as exc:
            failures.append((case_name, str(exc)))
            print(f"❌ {case_name}: {exc}")
            continue
        marker, label = _classify_result(result_summary)
        print(f"{marker} {case_name}: {label} ({result_summary})")

    passed_count = len(cases) - len(failures)
    print(
        f"Summary: {passed_count}/{len(cases)} cases matched baseline {args.commit}."
    )
    if failures:
        print("Failed cases:")
        for case_name, message in failures:
            print(f"❌ {case_name}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
