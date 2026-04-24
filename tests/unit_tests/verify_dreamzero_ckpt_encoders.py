"""Quick spot-check: do the text_encoder / image_encoder / vae weights inside
the DreamZero fine-tuning checkpoint equal the original pretrained .pth files?

Rationale: WANPolicyHead.__init__ unconditionally loads these three encoders
from pretrained paths. If the checkpoint also stores identical copies, we
can skip them in rlinf/models/embodiment/dreamzero/__init__.py load path.

We only sample a handful of tensors per component (enough to tell "bit-identical"
from "different") instead of loading everything.

Usage:
    python tests/unit_tests/verify_dreamzero_ckpt_encoders.py
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

CKPT = Path(
    "/mnt/project_rlinf/chenkang/RLinf_codes/RLinf_dreamzero_new/"
    ".vscode/checkpoints/checkpoint-6000"
)
PRE_DIR = Path("/mnt/project_rlinf_hs/yuanhuining/models/Wan2.1-I2V-14B-480P")

COMPONENTS = [
    # (ckpt_prefix, pretrained_pth, pth_key_prefix_to_prepend_when_looking_up_in_ckpt)
    # Rule: ckpt key == ckpt_prefix + pth_key
    ("action_head.text_encoder.",
     PRE_DIR / "models_t5_umt5-xxl-enc-bf16.pth"),
    ("action_head.image_encoder.model.",
     PRE_DIR / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ("action_head.vae.model.",
     PRE_DIR / "Wan2.1_VAE.pth"),
]

# Also diff the DiT on purpose, as a sanity "differ" baseline.
DIT_PREFIX = "action_head.model."

N_SAMPLES = 5  # per component


def load_index(ckpt_dir: Path) -> dict[str, str]:
    """key -> shard filename."""
    with open(ckpt_dir / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def open_shards(ckpt_dir: Path, keys):
    """Group keys by shard and yield (shard_path, [keys])."""
    idx = load_index(ckpt_dir)
    per_shard = {}
    for k in keys:
        if k not in idx:
            continue
        per_shard.setdefault(idx[k], []).append(k)
    return per_shard


def sample_compare(ckpt_prefix: str, pth_path: Path, n: int):
    print(f"\n=== {ckpt_prefix}  vs  {pth_path.name} ===")
    pth_sd = torch.load(pth_path, map_location="cpu", weights_only=True)
    pth_keys = list(pth_sd.keys())

    # pick first, middle, last, plus a couple evenly spaced samples
    if len(pth_keys) <= n:
        picks = pth_keys
    else:
        step = max(len(pth_keys) // n, 1)
        picks = [pth_keys[i * step] for i in range(n)]

    ckpt_keys = [ckpt_prefix + k for k in picks]
    per_shard = open_shards(CKPT, ckpt_keys)
    # flatten shard->keys into single map key->shard
    shard_of = {k: shard for shard, ks in per_shard.items() for k in ks}

    missing_in_ckpt, match, differ = [], [], []
    for pk, ck in zip(picks, ckpt_keys):
        if ck not in shard_of:
            missing_in_ckpt.append(ck)
            continue
        shard_path = str(CKPT / shard_of[ck])
        with safe_open(shard_path, framework="pt") as f:
            ckpt_tensor = f.get_tensor(ck)
        pth_tensor = pth_sd[pk]

        if ckpt_tensor.shape != pth_tensor.shape:
            differ.append((pk, "shape", tuple(pth_tensor.shape), tuple(ckpt_tensor.shape)))
            continue

        # Exact equality first; if dtypes differ, cast and use allclose.
        if ckpt_tensor.dtype == pth_tensor.dtype:
            equal = torch.equal(ckpt_tensor, pth_tensor)
        else:
            equal = torch.allclose(
                ckpt_tensor.float(), pth_tensor.float(), rtol=0, atol=0
            )

        if equal:
            match.append(pk)
        else:
            max_abs = (ckpt_tensor.float() - pth_tensor.float()).abs().max().item()
            differ.append((pk, "value", max_abs))

    print(f"  sampled {len(picks)} keys (of {len(pth_keys)} total in .pth)")
    print(f"  matched:        {len(match)}")
    print(f"  differed:       {len(differ)}")
    print(f"  missing in ckpt:{len(missing_in_ckpt)}")
    for d in differ[:3]:
        print(f"    differ: {d}")
    for m in missing_in_ckpt[:3]:
        print(f"    missing: {m}")


def sample_dit_vs_self():
    """Sanity: DiT tensors within the checkpoint are non-trivial (not all zero)
    and — to show "differ" baseline — we don't have a pretrained pth to compare
    against here, so just print a couple of norms so the reader can eyeball."""
    print(f"\n=== DiT sanity ({DIT_PREFIX}*): just a norm check ===")
    idx = load_index(CKPT)
    dit_keys = [k for k in idx if k.startswith(DIT_PREFIX)]
    picks = dit_keys[: N_SAMPLES]
    for k in picks:
        shard_path = str(CKPT / idx[k])
        with safe_open(shard_path, framework="pt") as f:
            t = f.get_tensor(k)
        print(f"  {k}  shape={tuple(t.shape)}  norm={t.float().norm().item():.3e}")


if __name__ == "__main__":
    for prefix, pth in COMPONENTS:
        sample_compare(prefix, pth, N_SAMPLES)
    sample_dit_vs_self()
