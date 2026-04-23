"""Test whether `load_state_dict(strict=False)` in fsdp_nft_policy_worker silently
drops keys due to FSDP-wrapped model emitting different key names than the fresh
model returned by `model_provider_func()`.

Uses the real OpenPi0 loader (`rlinf.models.get_model`) -- same path the worker
uses via `model_provider_func()` (fsdp_actor_worker.py:1050 -> get_model(cfg)).

Usage:
    cd /mnt/project_rlinf/chenkang/RLinf_codes/RLinf_new
    MODEL_PATH=/mnt/project_rlinf/chenkang/RLinf_codes/RLinf_new/logs/checkpoints/openpi0-libero-sft \
    torchrun --nproc_per_node=1 --master_port=29511 tests/unit_tests/test_nft_ema_statedict.py
"""
import os
import sys
import argparse

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)


def make_model_cfg(model_path: str, precision: str) -> OmegaConf:
    """Build a DictConfig matching `actor.model` from libero_object_nft_openpi yaml."""
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": model_path,
            "precision": precision,
            "num_action_chunks": 5,
            "action_dim": 7,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 4,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi0_libero",
                "num_images_in_input": 2,
                "noise_level": 0.0,
                "action_chunk": 5,
                "num_steps": 4,
                "train_expert_only": True,
                "action_env_dim": 7,
                "noise_method": "flow_ode",
                "add_value_head": False,
                "detach_critic_input": False,
                "use_dsrl": False,
                "dsrl_state_dim": 8,
                "dsrl_action_noise_dim": 32,
                "dsrl_num_q_heads": 10,
                "dsrl_agg_q": "mean",
                "dsrl_image_latent_dim": 64,
                "dsrl_state_latent_dim": 64,
                "dsrl_hidden_dims": [128, 128, 128],
                "is_nft": True,
            },
        }
    )


def _tensor_fingerprint(t: torch.Tensor) -> float:
    return t.detach().float().sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=os.environ.get(
            "MODEL_PATH",
            "/mnt/project_rlinf/chenkang/RLinf_codes/RLinf_new/logs/checkpoints/openpi0-libero-sft",
        ),
    )
    parser.add_argument("--precision", default="bf16")
    parser.add_argument(
        "--sharding",
        default="no_shard",
        choices=["no_shard", "full_shard", "shard_grad_op"],
    )
    parser.add_argument(
        "--skip-fsdp",
        action="store_true",
        help="Skip FSDP wrap; compare raw model.state_dict() keys directly.",
    )
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    is_main = local_rank == 0
    log = print if is_main else (lambda *a, **k: None)
    log(f"[cfg] model_path={args.model_path}  world_size={world_size}  sharding={args.sharding}")

    from rlinf.models import get_model

    model_cfg = make_model_cfg(args.model_path, args.precision)

    # -------------------------------------------------------------------
    # 1. Build model_a via same path used by worker (model_provider_func).
    # -------------------------------------------------------------------
    log("[1/7] building model_a via rlinf.models.get_model ...")
    model_a = get_model(model_cfg)
    model_a = model_a.to(device)
    # Force uniform dtype so FSDP1 flat_param can flatten without mixed-dtype error.
    if not args.skip_fsdp:
        log("   (forcing all params to fp32 for FSDP flat_param uniform-dtype constraint)")
        model_a = model_a.to(torch.float32)

    dtype_a = torch.float32 if not args.skip_fsdp else torch.bfloat16
    mixed_precision = MixedPrecision(
        param_dtype=dtype_a, reduce_dtype=dtype_a, buffer_dtype=dtype_a
    )
    sharding_strategy = {
        "no_shard": ShardingStrategy.NO_SHARD,
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    }[args.sharding]
    device_mesh = init_device_mesh("cuda", (world_size,))

    if args.skip_fsdp:
        log("[2/7] SKIPPING FSDP wrap; using raw model_a.")
        wrapped_a = model_a
    else:
        log("[2/7] FSDP-wrapping model_a (full module, no sub-wrap policy) ...")
        wrapped_a = FSDP(
            module=model_a,
            device_id=local_rank,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            device_mesh=device_mesh,
            use_orig_params=True,
            sync_module_states=True,
        )

    # 1a. Scale params of wrapped_a by 2.0 so we can verify transfer.
    SCALE = 2.0
    log(f"[3/7] scaling wrapped_a params by {SCALE} ...")
    with torch.no_grad():
        for p in wrapped_a.parameters():
            p.mul_(SCALE)

    # -------------------------------------------------------------------
    # 2. Extract full_state_dict. Matches the `get_model_state_dict(
    #    cpu_offload=False, full_state_dict=True)` call inside the worker.
    # -------------------------------------------------------------------
    log("[4/7] extracting full_state_dict from wrapped_a ...")
    opts = StateDictOptions(cpu_offload=False, full_state_dict=True)
    if args.skip_fsdp:
        state_dict_a = wrapped_a.state_dict()
    else:
        state_dict_a = get_model_state_dict(model=wrapped_a, options=opts)
    keys_a = set(state_dict_a.keys())
    log(f"   #keys(state_dict_a) = {len(keys_a)}")

    # -------------------------------------------------------------------
    # 3. Build model_b fresh (simulating model_provider_func() for ref_model).
    # -------------------------------------------------------------------
    log("[5/7] building fresh model_b via rlinf.models.get_model ...")
    model_b = get_model(model_cfg)
    model_b = model_b.to(device)
    keys_b = set(model_b.state_dict().keys())
    log(f"   #keys(state_dict_b) = {len(keys_b)}")

    before = {n: _tensor_fingerprint(p) for n, p in model_b.named_parameters()}

    # -------------------------------------------------------------------
    # 4. Execute the exact failing line:
    #       ref_model.load_state_dict(state_dict_from_EMA, strict=False)
    # -------------------------------------------------------------------
    log("[6/7] load_state_dict(state_dict_a, strict=False) into model_b ...")
    result = model_b.load_state_dict(state_dict_a, strict=False)

    only_in_a = keys_a - keys_b
    only_in_b = keys_b - keys_a
    log("=" * 78)
    log(f"[KEY DIFF]")
    log(f"   only_in_a (extra keys not present in fresh model_b): {len(only_in_a)}")
    for k in sorted(only_in_a)[:20]:
        log(f"     A-only: {k}")
    log(f"   only_in_b (missing from state_dict_a): {len(only_in_b)}")
    for k in sorted(only_in_b)[:20]:
        log(f"     B-only: {k}")
    log(f"[load_state_dict return]")
    log(f"   missing_keys    = {len(result.missing_keys)}")
    for k in result.missing_keys[:20]:
        log(f"     MISSING: {k}")
    log(f"   unexpected_keys = {len(result.unexpected_keys)}")
    for k in result.unexpected_keys[:20]:
        log(f"     UNEXPECTED: {k}")
    log("=" * 78)

    # -------------------------------------------------------------------
    # 5. Weight-transfer verification.
    # -------------------------------------------------------------------
    log("[7/7] verifying all params in model_b became scale*original ...")
    transferred = 0
    not_transferred = 0
    failing_names = []
    with torch.no_grad():
        for name, p in model_b.named_parameters():
            after = _tensor_fingerprint(p)
            orig = before[name]
            tol = max(1e-3, abs(orig) * 1e-3)
            if abs(after - orig * SCALE) < tol:
                transferred += 1
            else:
                not_transferred += 1
                if len(failing_names) < 10:
                    failing_names.append((name, orig, after, orig * SCALE))

    log(f"[weight transfer check]   SCALE={SCALE}")
    log(f"   transferred (after ≈ orig * SCALE):           {transferred}")
    log(f"   NOT transferred (kept fresh init or garbled): {not_transferred}")
    for name, orig, after, expected in failing_names:
        log(f"     FAIL: {name}  orig={orig:.4f}  after={after:.4f}  expected={expected:.4f}")
    log("=" * 78)

    if (
        len(result.missing_keys) == 0
        and len(result.unexpected_keys) == 0
        and not_transferred == 0
    ):
        log(">>> VERDICT: NO BUG. strict=False is not silently dropping keys.")
        log("    EMA state_dict transfer works correctly.")
        rc = 0
    else:
        log(">>> VERDICT: BUG DETECTED. ref_model retained fresh init weights.")
        rc = 1

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    sys.exit(rc)


if __name__ == "__main__":
    main()
