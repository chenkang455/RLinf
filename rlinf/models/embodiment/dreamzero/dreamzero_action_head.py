import torch
from einops import rearrange
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
    WANPolicyHead,
    WANPolicyHeadConfig,
)
from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import (
    FlowUniPCMultistepScheduler,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


class DreamZeroActionHead(WANPolicyHead):
    """RLinf-local override point for DreamZero lazy inference."""

    config_class = WANPolicyHeadConfig

    # ================ SFT Training ================
    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """Run the local DreamZero training forward path."""
        self.set_frozen_modules_to_eval_mode()
        breakpoint()
        policy_inputs = self.prepare_policy_inputs(data=action_input, mode="training")
        training_targets = self.build_training_noise_targets(policy_inputs)
        loss_dict = self.compute_training_losses(policy_inputs, training_targets)
        return loss_dict

    def compute_training_losses(
        self,
        policy_inputs: BatchFeature,
        training_targets: BatchFeature,
    ) -> BatchFeature:
        """Run training forward and compute DreamZero losses."""
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            video_noise_pred, action_noise_pred = self.model(
                training_targets["noisy_latents"].transpose(1, 2),
                timestep=training_targets["timestep"],
                clip_feature=policy_inputs["clip_feas"].to(self._device),
                y=policy_inputs["ys"].to(self._device),
                context=policy_inputs["prompt_embs"].to(self._device),
                seq_len=training_targets["seq_len"],
                state=policy_inputs["state_features"],
                embodiment_id=policy_inputs["embodiment_id"],
                action=training_targets["noisy_actions"],
                timestep_action=training_targets["timestep_action"],
                clean_x=policy_inputs["video_latents"],
            )
            # dynamics loss
            dynamics_loss_per_sample = torch.nn.functional.mse_loss(
                video_noise_pred.float(),
                training_targets["training_target"].float(),
                reduction="none",
            ).mean(dim=(1, 3, 4))
            weighted_dynamics_loss = self._reduce_weighted_loss(
                dynamics_loss_per_sample,
                training_targets["timestep"],
            )
            # action loss
            action_loss_per_sample = torch.nn.functional.mse_loss(
                action_noise_pred.float(),
                training_targets["training_target_action"].float(),
                reduction="none",
            ) * policy_inputs["action_mask"]
            action_loss_per_sample = (
                policy_inputs["has_real_action"][:, None, None].float()
                * action_loss_per_sample
            )
            weighted_action_loss = self._reduce_weighted_loss(
                action_loss_per_sample.mean(dim=2),
                training_targets["timestep_action"],
            )
            loss = weighted_dynamics_loss + weighted_action_loss

        return BatchFeature(
            data={
                "loss": loss,
                "dynamics_loss": weighted_dynamics_loss,
                "action_loss": weighted_action_loss,
            }
        )

    def _reduce_weighted_loss(
        self,
        loss_per_step: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Apply scheduler weights to per-step losses and reduce to a scalar."""
        weights = self.scheduler.training_weight(timestep.flatten(0, 1)).unflatten(
            0,
            (timestep.shape[0], timestep.shape[1]),
        ).to(self._device)
        return (loss_per_step * weights).mean()

    def lazy_joint_video_action(self, backbone_output: BatchFeature, action_input: BatchFeature, latent_video=None) -> BatchFeature:
        """Run local lazy inference implementation."""
        return self.sample_action_video(backbone_output=backbone_output, action_input=action_input)

    def sample_action_video(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """Sample action video from DreamZero."""
        self.set_frozen_modules_to_eval_mode()
        policy_inputs = self.prepare_policy_inputs(data=action_input, mode="inference")
        sampling_state = self.prepare_noise_action_video(policy_inputs)
        kv_caches, crossattn_caches = self.prepare_kv_cache(sampling_state["noise_obs"])
        self.warmup_kv_cache(policy_inputs, sampling_state, kv_caches, crossattn_caches)
        return self.denoise_action_video(
            policy_inputs=policy_inputs,
            sampling_state=sampling_state,
            kv_caches=kv_caches,
            crossattn_caches=crossattn_caches,
        )

    def prepare_policy_inputs(self, data: BatchFeature, mode: str = "inference") -> BatchFeature:
        """Prepare DreamZero policy inputs for inference or training."""
        # embodiment id input
        embodiment_id = data.embodiment_id
        self.current_start_frame = 0
        # state features input
        if mode == "training":
            state_features = data.state
            prompt_embs = self.encode_prompt(data["text"], data["text_attention_mask"])
        elif mode == "inference":
            state_features = data.state.to(dtype=torch.bfloat16)
            text_inputs = self._prepare_text_inputs(data)
            prompt_embs = [
                self.encode_prompt(text, attention_mask)
                for text, attention_mask in text_inputs
            ]
        # video process
        if mode == "training":
            videos = self._normalize_videos(data["images"],output_dtype=self.dtype)
        elif mode == "inference":
            videos = self._normalize_videos(data["images"],output_dtype=torch.bfloat16)
        _, _, num_frames, height, width = videos.shape
        # `encode_image` expects frame-major layout `[B, T, C, H, W]`.
        image = videos[:, :, :1].transpose(1, 2)
        if mode == "training":
            image_num_frames = num_frames
        elif mode == "inference":
            image_num_frames = self.num_frames
        clip_feas, ys, image_latents = self.encode_image(image, image_num_frames, height, width)
        clip_feas = clip_feas.to(dtype=image_latents.dtype)
        ys = ys.to(dtype=image_latents.dtype)
        # outputs
        outputs = {
                "videos": videos,
                "embodiment_id": embodiment_id,
                "state_features": state_features,
                "prompt_embs": prompt_embs,
                "clip_feas": clip_feas,
                "ys": ys,
            }
        if mode == "training":
            outputs["actions"] = data.action
            outputs["has_real_action"] = data.has_real_action
            outputs["action_mask"] = data.action_mask
            outputs["video_latents"] = self.encode_video(
                videos,
                self.tiled,
                (self.tile_size_height, self.tile_size_width),
                (self.tile_stride_height, self.tile_stride_width),
            )
        elif mode == "inference":
            outputs["image_latents"] = image_latents
        return BatchFeature(data=outputs)

    def prepare_noise_action_video(self, policy_inputs: BatchFeature) -> BatchFeature:
        """Prepare initial video/action noise and diffusion shapes."""
        # Reuse the encoded first-frame latents as the video-side reference block.
        image_latents = policy_inputs["image_latents"]
        _, latent_channels, _, latent_height, latent_width = image_latents.shape
        # Sample the initial noisy video/action states for the denoising loop.
        noise_obs = self.generate_noise(
            (
                image_latents.shape[0],
                latent_channels,
                self.num_frame_per_block,
                latent_height,
                latent_width,
            ),
            seed=self.seed,
            device="cuda",
            dtype=torch.bfloat16,
        )
        noise_action = self.generate_noise(
            (image_latents.shape[0], self.action_horizon, self.model.action_dim),
            seed=self.seed,
            device="cuda",
            dtype=torch.bfloat16,
        )
        noise_obs = noise_obs.transpose(1, 2)
        image_latents = image_latents.transpose(1, 2)
        outputs = {
            "noise_obs": noise_obs,
            "noise_action": noise_action,
            "image_latents": image_latents,
        }
        return BatchFeature(data=outputs)

    def build_training_noise_targets(
        self,
        policy_inputs: BatchFeature,
    ) -> BatchFeature:
        """Build timestep-aligned noisy inputs and training targets."""
        latents = policy_inputs["video_latents"].transpose(1, 2)
        actions = policy_inputs["actions"]
        # build timestep and block timestep
        noise = torch.randn_like(policy_inputs["video_latents"]).transpose(1, 2)
        timestep_id = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (noise.shape[0], noise.shape[1]),
        )
        timestep_id_block = timestep_id[:, 1:].reshape(
            timestep_id.shape[0], -1, self.num_frame_per_block
        )
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]
        # build action timestep
        noise_action = torch.randn_like(actions)
        timestep_action_id = timestep_id_block.repeat(
            1, 1, actions.shape[1] // (noise.shape[1] - 1)
        )
        timestep_action_id = timestep_action_id.reshape(
            timestep_action_id.shape[0], -1
        )
        # map back to timestep
        timestep_id_block = timestep_id_block.reshape(timestep_id_block.shape[0], -1)
        timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block], dim=1)
        # build latent noisy video and target 
        timestep = self.scheduler.timesteps[timestep_id].to(self._device)
        noisy_latents = self.scheduler.add_noise(
            latents.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = self.scheduler.training_target(latents, noise, timestep).transpose(1, 2)
        # build latent noisy action and target 
        _, num_frames, _, height, width = noise.shape
        frame_seqlen = (height // 2) * (width // 2)
        seq_len = num_frames * frame_seqlen
        timestep_action = self.scheduler.timesteps[timestep_action_id].to(self._device)
        noisy_actions = self.scheduler.add_noise(
            actions.flatten(0, 1),
            noise_action.flatten(0, 1),
            timestep_action.flatten(0, 1),
        ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
        training_target_action = self.scheduler.training_target(
            actions,
            noise_action,
            timestep_action,
        )
        return BatchFeature(
            data={
                "timestep_id": timestep_id,
                "timestep_action_id": timestep_action_id,
                "timestep": timestep,
                "timestep_action": timestep_action,
                "noisy_latents": noisy_latents,
                "noisy_actions": noisy_actions,
                "training_target": training_target,
                "training_target_action": training_target_action,
                "seq_len": seq_len,
            }
        )

    def prepare_kv_cache(self, noise_obs: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Initialize and fetch KV caches for the current batch."""
        # KV cache size depends on the latent spatial resolution after VAE downsampling.
        batch_size = noise_obs.shape[0]
        latent_height = noise_obs.shape[-2]
        latent_width = noise_obs.shape[-1]
        frame_seqlen = (latent_height // 2) * (latent_width // 2)
        self.kv_cache1, self.kv_cache_neg = self._create_kv_caches(
            batch_size=batch_size,
            dtype=noise_obs.dtype,
            device=noise_obs.device,
            frame_seqlen=frame_seqlen,
        )
        self.crossattn_cache, self.crossattn_cache_neg = self._create_crossattn_caches(
            batch_size=batch_size,
            dtype=noise_obs.dtype,
            device=noise_obs.device,
        )
        kv_caches = self._get_caches([self.kv_cache1, self.kv_cache_neg])
        crossattn_caches = self._get_caches([self.crossattn_cache, self.crossattn_cache_neg])
        return kv_caches, crossattn_caches

    def warmup_kv_cache(
        self,
        policy_inputs: BatchFeature,
        sampling_state: BatchFeature,
        kv_caches: list[torch.Tensor],
        crossattn_caches: list[torch.Tensor],
    ) -> None:
        """Warm up KV caches with zero-timestep reference latents."""
        noise_obs = sampling_state["noise_obs"]
        timestep = torch.zeros((noise_obs.shape[0], 1), device=noise_obs.device, dtype=torch.int64)
        seq_len = (noise_obs.shape[-2] // 2) * (noise_obs.shape[-1] // 2)
        self._run_diffusion_steps(
            # DiT forward expects channel-major latent layout `[B, C, T, H, W]`.
            noisy_input=sampling_state["image_latents"].transpose(1, 2),
            timestep=timestep,
            action=None,
            timestep_action=None,
            state=None,
            embodiment_id=None,
            context=policy_inputs["prompt_embs"],
            seq_len=seq_len,
            y=policy_inputs["ys"][:, :, 0:1],
            clip_feature=policy_inputs["clip_feas"],
            kv_caches=kv_caches,
            crossattn_caches=crossattn_caches,
            kv_cache_metadata={"start_frame": 0, "update_kv_cache": True},
        )
        self.current_start_frame += 1

    def denoise_action_video(
        self,
        policy_inputs: BatchFeature,
        sampling_state: BatchFeature,
        kv_caches: list[torch.Tensor],
        crossattn_caches: list[torch.Tensor],
    ) -> BatchFeature:
        """Run the main video/action denoising loop."""
        # input 
        noisy_input = sampling_state["noise_obs"]
        noisy_input_action = sampling_state["noise_action"]
        batch_size = noisy_input.shape[0]
        seq_len = (noisy_input.shape[-2] // 2) * (noisy_input.shape[-1] // 2) * noisy_input.shape[1]
        device = noisy_input.device
        # eval_action_sampler: "unipc" | "euler"
        use_euler = getattr(self, "eval_action_sampler", "unipc") == "euler"
        if use_euler:
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.scheduler.num_train_timesteps, shift=1.0,
            )
            sample_scheduler_action = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=self.scheduler.num_train_timesteps, shift=1.0,
            )
            sample_scheduler.set_timesteps(self.num_inference_steps, device=device)
            sample_scheduler_action.set_timesteps(self.num_inference_steps, device=device)
        else:
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.scheduler.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler_action = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.scheduler.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(self.num_inference_steps, device=device, shift=self.sigma_shift)
            sample_scheduler_action.set_timesteps(self.num_inference_steps, device=device, shift=self.sigma_shift)

        # Rescale video sigmas [sigma_max, 0] -> [sigma_max, video_inference_final_noise].
        if getattr(self.config, "decouple_inference_noise", False):
            video_final_noise = float(self.config.video_inference_final_noise)
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = (
                sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max
                + video_final_noise
            )
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64)

        prev_predictions = []
        self.skip_countdown = 0
        for index, current_timestep in enumerate(sample_scheduler.timesteps):
            video_timestep = sample_scheduler.timesteps[index]
            action_timestep = sample_scheduler_action.timesteps[index]
            timestep = torch.ones((batch_size, self.num_frame_per_block), device=device, dtype=torch.int64) * video_timestep
            timestep_action = torch.ones((batch_size, self.action_horizon), device=device, dtype=torch.int64) * action_timestep

            if self.should_run_model(index, current_timestep, prev_predictions):
                if self.current_start_frame + self.num_frame_per_block <= policy_inputs["ys"].shape[2]:
                    start_frame = self.current_start_frame
                    end_frame = self.current_start_frame + self.num_frame_per_block
                else:
                    frame_len = policy_inputs["ys"].shape[2]
                    start_frame = frame_len - self.num_frame_per_block
                    end_frame = frame_len
                y = policy_inputs["ys"][:, :, start_frame : end_frame]
                # eval_video_mode: "normal" | "clean" | "clean_noised" | "zero" | "random"
                video_mode = getattr(self, "eval_video_mode", "normal")
                if video_mode in ("clean", "clean_noised"):
                    clean = (
                        sampling_state["image_latents"]
                        .expand(-1, self.num_frame_per_block, -1, -1, -1)
                        .contiguous()
                        .to(noisy_input.dtype)
                    )
                    if video_mode == "clean":
                        noisy_input = clean
                    else:
                        # x_t = (1 - sigma) * x_0 + sigma * eps, matched to current timestep.
                        sigma = sample_scheduler.sigmas[index].to(
                            device=clean.device, dtype=clean.dtype
                        )
                        noisy_input = (1.0 - sigma) * clean + sigma * torch.randn_like(clean)
                elif video_mode == "zero":
                    noisy_input = torch.zeros_like(noisy_input)
                elif video_mode == "random":
                    noisy_input = torch.randn_like(noisy_input)
                predictions = self._run_diffusion_steps(
                    # Scheduler state is frame-major, so switch back before DiT.
                    noisy_input=noisy_input.transpose(1, 2),
                    timestep=timestep,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    state=policy_inputs["state_features"],
                    embodiment_id=policy_inputs["embodiment_id"],
                    context=policy_inputs["prompt_embs"],
                    seq_len=seq_len,
                    y=y,
                    clip_feature=policy_inputs["clip_feas"],
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    kv_cache_metadata={
                        "start_frame": self.current_start_frame,
                        "update_kv_cache": False,
                    },
                )
                flow_pred_cond, flow_pred_cond_action = predictions[0]
                flow_pred_uncond, _ = predictions[1]
                flow_pred = flow_pred_uncond + self.cfg_scale * (flow_pred_cond - flow_pred_uncond)
                prev_predictions.append((current_timestep, flow_pred, flow_pred_cond_action))
                if len(prev_predictions) > 2:
                    prev_predictions.pop(0)
            else:
                _, flow_pred, flow_pred_cond_action = prev_predictions[-1]

            noisy_input = sample_scheduler.step(
                # Scheduler updates samples in frame-major `[B, T, C, H, W]`.
                model_output=flow_pred.transpose(1, 2),
                timestep=video_timestep,
                sample=noisy_input,
                return_dict=False,
                **({} if use_euler else {"step_index": index}),
            )[0]
            noisy_input_action = sample_scheduler_action.step(
                model_output=flow_pred_cond_action,
                timestep=action_timestep,
                sample=noisy_input_action,
                return_dict=False,
                **({} if use_euler else {"step_index": index}),
            )[0]

        # output
        output = torch.cat([sampling_state["image_latents"], noisy_input], dim=1)
        return BatchFeature(data={"action_pred": noisy_input_action, "video_pred": output.transpose(1, 2)})

    def _normalize_videos(
        self,
        videos: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Convert input videos to `[B, C, T, H, W]` in `[-1, 1]`."""
        videos = rearrange(videos, "b t h w c -> b c t h w")
        if videos.dtype == torch.uint8:
            videos = videos.float().div(255.0)
            videos = videos.to(dtype=self.dtype)
            bsz, channels, num_frames, height, width = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4).reshape(bsz * num_frames, channels, height, width)
            videos = self.normalize_video(videos)
            videos = videos.reshape(bsz, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=output_dtype)
        return videos.to(dtype=output_dtype)

    def _resolve_target_video_size(self) -> tuple[int | None, int | None]:
        """Resolve an optional target video size for Wan2.2/5B checkpoints."""
        target_height = getattr(self.config, "target_video_height", None)
        target_width = getattr(self.config, "target_video_width", None)
        if target_height is not None and target_width is not None:
            return target_height, target_width

        if getattr(self.model, "frame_seqlen", None) in (50, 55):
            # Match the upstream 5B fallback when the checkpoint config does not
            # explicitly carry the resize target yet.
            return 176, 320
        return None, None

    def _resize_videos_for_model(self, videos: torch.Tensor) -> torch.Tensor:
        """Resize inputs when the backbone expects a fixed 5B resolution."""
        target_height, target_width = self._resolve_target_video_size()
        if target_height is None or target_width is None:
            return videos

        _, _, num_frames, height, width = videos.shape
        if (height, width) == (target_height, target_width):
            return videos

        batch_size, channels, _, _, _ = videos.shape
        resized = torch.nn.functional.interpolate(
            videos.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.reshape(batch_size, num_frames, channels, target_height, target_width).permute(
            0, 2, 1, 3, 4
        )
