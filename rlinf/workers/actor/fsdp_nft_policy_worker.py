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

import os

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory, masked_mean
from rlinf.workers.actor.fsdp_actor_worker import (
    EmbodiedFSDPActor,
    process_nested_dict_for_train,
)


class EmbodiedNFTFSDPPolicy(EmbodiedFSDPActor):
    """Embodied FSDP policy worker for NFT with off-policy support."""

    def init_worker(self) -> None:
        """Initialize actor and lagged rollout policy state."""
        super().init_worker()
        self._init_rollout_model()

    def get_rollout_state_dict(self) -> dict:
        """Get the lagged rollout policy state dict."""
        return self.rollout_state_dict

    def _init_rollout_model(self) -> None:
        """Initialize lagged rollout policy state and model."""
        self.rollout_state_dict = self.get_model_state_dict(
            cpu_offload=True, full_state_dict=True
        )
        self.rollout_model = self.model_provider_func()
        self.rollout_model.load_state_dict(self.rollout_state_dict, strict=False)
        self.rollout_model.eval()
        self.rollout_model.requires_grad_(False)

    def _update_rollout_state_dict(self) -> None:
        """Update lagged rollout policy with NFT tau."""
        tau = float(self.cfg.algorithm.get("nft_tau", 1.0))
        student_state_dict = self.get_model_state_dict(
            cpu_offload=True, full_state_dict=True
        )

        if tau >= 1.0 or not hasattr(self, "rollout_state_dict"):
            self.rollout_state_dict = student_state_dict
            self.rollout_model.load_state_dict(self.rollout_state_dict, strict=False)
            return

        for key, target_tensor in self.rollout_state_dict.items():
            source_tensor = student_state_dict[key]
            if (
                torch.is_tensor(target_tensor)
                and torch.is_tensor(source_tensor)
                and target_tensor.is_floating_point()
                and source_tensor.is_floating_point()
            ):
                target_tensor.lerp_(source_tensor.to(target_tensor.dtype), tau)
            elif torch.is_tensor(target_tensor) and torch.is_tensor(source_tensor):
                target_tensor.copy_(source_tensor)
            else:
                self.rollout_state_dict[key] = source_tensor
        self.rollout_model.load_state_dict(self.rollout_state_dict, strict=False)

    def _precompute_nft_training_inputs(self) -> None:
        """Prepare NFT training tensors before the update loop."""
        forward_inputs = self.rollout_batch["forward_inputs"]
        target_space = self.cfg.algorithm.get("nft_target_space", "xnext")

        if target_space == "x0":
            # x0 space: resample step indices and interpolate xcur from x0
            num_steps = self.model.config.num_steps
            x0 = forward_inputs["nft_x0"]
            step_indices = torch.randint(
                0, num_steps, (x0.shape[0],), device=x0.device
            )
            schedule = torch.linspace(
                1, 0, num_steps + 1, device=x0.device, dtype=x0.dtype
            )
            t = schedule[step_indices.long()]
            xcur = (1 - t[:, None, None]) * x0 + t[:, None, None] * torch.randn_like(x0)
            forward_inputs["nft_xcur"] = xcur
            forward_inputs["nft_step_index"] = step_indices
            forward_inputs["nft_v"] = self._recompute_v_old(forward_inputs, xcur, t)
        elif target_space == "xnext":
            # xnext space: nft_xcur, nft_step_index, nft_v all come from rollout directly
            pass
        else:
            raise ValueError(f"Unsupported nft_target_space: {target_space}")

    def _recompute_v_old(
        self,
        forward_inputs: dict,
        xcur: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute old velocity with the lagged rollout model."""
        chunk_size = self.cfg.actor.micro_batch_size
        v_old_chunks = []

        training_was_on_device = not self.is_weight_offloaded
        if training_was_on_device:
            self.offload_param_and_grad()
            clear_memory()
        self.rollout_model.to(self.device)

        with torch.no_grad():
            for start in range(0, xcur.shape[0], chunk_size):
                end = min(start + chunk_size, xcur.shape[0])
                forward_inputs_slice = put_tensor_device(
                    self._slice_forward_inputs(forward_inputs, start, end),
                    self.device,
                )
                rollout_output = self.rollout_model(
                    forward_type=ForwardType.NFT,
                    forward_inputs=forward_inputs_slice,
                    nft_inputs={
                        "x_t": xcur[start:end].to(device=self.device),
                        "timesteps": t[start:end].to(device=self.device),
                    },
                    compute_values=False,
                )
                v_old_chunks.append(rollout_output["v_theta"].detach().cpu())

        self.rollout_model.to("cpu")
        if training_was_on_device:
            self.load_param_and_grad(self.device)

        return torch.cat(v_old_chunks, dim=0).to(xcur.device)

    def _slice_forward_inputs(
        self, forward_inputs: dict, start: int, end: int
    ) -> dict:
        """Slice nested forward inputs along the batch dimension."""
        ret = {}
        for key, value in forward_inputs.items():
            if isinstance(value, torch.Tensor):
                ret[key] = value[start:end]
            elif isinstance(value, dict):
                ret[key] = self._slice_forward_inputs(value, start, end)
            else:
                ret[key] = value
        return ret

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """Run NFT training with off-policy decay support."""
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )
            self._precompute_nft_training_inputs()

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    batch = put_tensor_device(
                        batch,
                        f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )

                    loss, metrics_data = self.nft_forward_and_loss(batch)

                    if self.enable_sft_co_train:
                        self._train_sft_epoch(metrics_data, loss)

                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/total_loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                self.torch_platform.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self._update_rollout_state_dict()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    def nft_forward_and_loss(self, batch):
        """NFT-specific forward and loss computation."""
        # data input
        forward_inputs = batch["forward_inputs"]
        target_space = self.cfg.algorithm.get("nft_target_space", "xnext")
        x_t_input = forward_inputs["nft_xcur"]
        step_indices = forward_inputs["nft_step_index"]
        num_steps = self.model.config.num_steps
        schedule = torch.linspace(
            1,
            0,
            num_steps + 1,
            device=x_t_input.device,
            dtype=x_t_input.dtype,
        )
        t = schedule[step_indices.long()]
        # compute v_theta
        with self.amp_context:
            output_dict = self.model(
                forward_type=ForwardType.NFT,
                forward_inputs=forward_inputs,
                nft_inputs={"x_t": x_t_input, "timesteps": t},
                compute_values=False,
            )
        # compute v_theta and v_old
        v_theta, v_old, x_t = self._slice_nft_tensors(output_dict, forward_inputs)
        # shape alignment
        sum_dims = tuple(range(2, x_t.ndim))
        batch_size, chunk_len = x_t.shape[:2]
        loss_mask = batch["loss_mask"].expand(batch_size, chunk_len)
        advantages = batch["advantages"].expand(batch_size, chunk_len)
        # preference y ∈ [-1, 1]
        adv_clip_max = float(self.cfg.algorithm.get("adv_clip_max", 1.0))
        y = (
            torch.clamp(advantages * 2.0 - 1.0, -adv_clip_max, adv_clip_max)
            / adv_clip_max
        )
        # clip delta v
        delta_v = v_theta - v_old
        delta_norm = delta_v.norm(dim=sum_dims, keepdim=True) + 1e-8
        max_drift = float(self.cfg.algorithm.get("max_drift", 0.5))
        clip_coef = (max_drift / delta_norm).clamp(max=1.0)
        beta = float(self.cfg.algorithm.get("nft_beta", 1.0))
        delta_v_clipped = delta_v * clip_coef
        # pos and neg candidate velocities
        v_pos = v_old + beta * delta_v_clipped
        v_neg = v_old - beta * delta_v_clipped
        # schedule params
        t_bc, dt_bc, sigma_i, std_t_det = self._build_schedule_params(
            schedule, step_indices, forward_inputs["nft_noise_level"], x_t
        )
        # compute var: x0 space scales with t^2, xnext space scales with dt^2
        var_scale = float(self.cfg.algorithm.get("nft_var_scale", 1.0))
        if torch.all(forward_inputs["nft_noise_level"] == 0):
            var = (t_bc**2 if target_space == "x0" else dt_bc**2) * var_scale
        else:
            var = std_t_det**2 + 1e-4
        # predict target state
        target, pred_pos = self._compute_nft_target_and_pred(
            forward_inputs, target_space, x_t, v_pos, t_bc, dt_bc, sigma_i
        )
        _, pred_neg = self._compute_nft_target_and_pred(
            forward_inputs, target_space, x_t, v_neg, t_bc, dt_bc, sigma_i
        )
        e_pos = ((pred_pos - target) ** 2 / var).sum(dim=sum_dims)
        e_neg = ((pred_neg - target) ** 2 / var).sum(dim=sum_dims)
        delta_e = e_pos - e_neg
        # dpo loss
        dpo_beta = float(self.cfg.algorithm.get("dpo_beta", 1.0))
        logit = (dpo_beta / 2.0) * y * delta_e
        loss = masked_mean(F.softplus(logit), loss_mask)
        # metrics
        with torch.no_grad():
            metrics_data = {
                "actor/nft_loss": loss.item(),
                "actor/delta_v_norm": delta_v.norm(dim=sum_dims).mean().item(),
                "actor/clip_frac": (clip_coef < 1).float().mean().item(),
                "actor/E_pos_mean": e_pos.mean().item(),
                "actor/E_neg_mean": e_neg.mean().item(),
                "actor/delta_E_mean": delta_e.mean().item(),
                "actor/logit_mean": logit.mean().item(),
                "actor/pref_acc": (logit < 0).float().mean().item(),
            }
        return loss, metrics_data

    def _slice_nft_tensors(
        self,
        output_dict: dict,
        forward_inputs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice NFT tensors used by NFT loss."""
        chunk = output_dict["v_theta"].shape[1]
        action_env_dim = self.model.config.action_env_dim
        v_theta = output_dict["v_theta"][:, :chunk, :action_env_dim]
        x_t = forward_inputs["nft_xcur"][:, :chunk, :action_env_dim]
        v_old = forward_inputs["nft_v"][:, :chunk, :action_env_dim].detach()
        return v_theta, v_old, x_t

    def _compute_nft_target_and_pred(
        self,
        forward_inputs: dict,
        target_space: str,
        x_t: torch.Tensor,
        vel: torch.Tensor,
        t_bc: torch.Tensor,
        dt_bc: torch.Tensor,
        sigma_i: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build target and predicted state for the given NFT target space."""
        # TODO: move into the model file or utils file for better reuse
        if target_space == "x0":
            target = forward_inputs["nft_x0"][:, : x_t.shape[1], : x_t.shape[2]]
            pred = x_t - vel * t_bc
        elif target_space == "xnext":
            target = forward_inputs["nft_xnext"][:, : x_t.shape[1], : x_t.shape[2]]
            x0_pred = x_t - vel * t_bc
            x1_pred = x_t + vel * (1 - t_bc)
            w0 = 1.0 - (t_bc - dt_bc)
            w1 = t_bc - dt_bc - sigma_i**2 * dt_bc / (2 * t_bc)
            pred = x0_pred * w0 + x1_pred * w1
        else:
            raise ValueError(f"Unsupported nft_target_space: {target_space}")
        return target, pred

    def _build_schedule_params(
        self,
        schedule: torch.Tensor,  # [num_steps+1] linspace 1→0
        step_indices: torch.Tensor,  # [B]
        noise_level: torch.Tensor | float,
        x_t: torch.Tensor,  # reference tensor for ndim/device/dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute timestep & noise params, broadcast to [B, 1, ..., 1] for x_t.ndim.

        Returns: (t_bc, dt_bc, sigma_i, std_t_det)
        """
        # TODO: move into the model file or utils file for better reuse
        # params
        ndim = x_t.ndim
        idx = step_indices.long()

        def pad(x):
            return x.view(-1, *([1] * (ndim - 1)))

        # timestep: t_cur and dt = t_cur - t_next
        t_bc = pad(schedule[idx])
        dt_bc = pad(schedule[idx] - schedule[idx + 1])
        # SDE noise scale: σ_i = sqrt(t / (1-t)) * noise_level
        safe_schedule = schedule.clone()
        safe_schedule[0] = safe_schedule[1]  # avoid div-by-zero at t=1
        sigma_i = pad(torch.sqrt(schedule[:-1] / (1 - safe_schedule[:-1]))[idx])
        nl = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
        sigma_i = sigma_i * (pad(nl) if nl.ndim > 0 else nl)
        # transition std
        std_t_det = (torch.sqrt(dt_bc.clamp_min(0)) * sigma_i).detach()
        return t_bc, dt_bc, sigma_i, std_t_det
