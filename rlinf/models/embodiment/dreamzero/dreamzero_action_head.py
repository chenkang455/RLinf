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


class DreamZeroActionHead(WANPolicyHead):
    """RLinf-local override point for DreamZero lazy inference."""

    config_class = WANPolicyHeadConfig

    def lazy_joint_video_action(self, backbone_output: BatchFeature, action_input: BatchFeature, latent_video=None) -> BatchFeature:
        """Run local lazy inference implementation."""
        return self.sample_action_video(backbone_output=backbone_output, action_input=action_input)

    def sample_action_video(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """Sample action video from DreamZero."""
        del backbone_output
        self.set_frozen_modules_to_eval_mode()
        policy_inputs = self.prepare_policy_inputs(data=action_input)
        sampling_state = self.prepare_noise_action_video(policy_inputs)
        kv_caches, crossattn_caches = self.prepare_kv_cache(sampling_state["noise_obs"])
        self.warmup_kv_cache(policy_inputs, sampling_state, kv_caches, crossattn_caches)
        return self.denoise_action_video(
            policy_inputs=policy_inputs,
            sampling_state=sampling_state,
            kv_caches=kv_caches,
            crossattn_caches=crossattn_caches,
        )

    def prepare_policy_inputs(self, data: BatchFeature) -> BatchFeature:
        """Prepare single-frame DreamZero policy inputs."""
        # embodiment id input
        embodiment_id = data.embodiment_id
        # state features input
        state_features = data.state.to(dtype=torch.bfloat16)
        # language input
        text_inputs = self._prepare_text_inputs(data)
        prompt_embs = [self.encode_prompt(text, attention_mask) for text, attention_mask in text_inputs]
        # video input [B, C, T, H, W]
        videos = self._normalize_videos(data["images"])
        _, _, _, height, width = videos.shape
        image = videos[:, :, :1].transpose(1, 2)
        # [B, C, T, H, W] -> [B, T, C, H, W]
        clip_feas, ys, image_latents = self.encode_image(image, self.num_frames, height, width)
        clip_feas = clip_feas.to(dtype=image_latents.dtype)
        ys = ys.to(dtype=image_latents.dtype)
        return BatchFeature(
            data={
                "videos": videos,
                "embodiment_id": embodiment_id,
                "state_features": state_features,
                "prompt_embs": prompt_embs,
                "image_latents": image_latents,
                "clip_feas": clip_feas,
                "ys": ys,
            }
        )

    def prepare_noise_action_video(self, policy_inputs: BatchFeature) -> BatchFeature:
        """Prepare initial video/action noise and diffusion shapes."""
        # Reuse the encoded first-frame latents as the video-side reference block.
        image_latents = policy_inputs["image_latents"]
        _, _, _, height, width = policy_inputs["videos"].shape
        # Sample the initial noisy video/action states for the denoising loop.
        noise_obs = self.generate_noise(
            (image_latents.shape[0], 16, self.num_frame_per_block, height // 8, width // 8),
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
        return BatchFeature(
            data={
                "image_latents": image_latents.transpose(1, 2),
                "noise_obs": noise_obs.transpose(1, 2),
                "noise_action": noise_action,
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
        # sample scheduler
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

        prev_predictions = []
        self.skip_countdown = 0
        for index, current_timestep in enumerate(sample_scheduler.timesteps):
            video_timestep = sample_scheduler.timesteps[index]
            action_timestep = sample_scheduler_action.timesteps[index]
            timestep = torch.ones((batch_size, self.num_frame_per_block), device=device, dtype=torch.int64) * video_timestep
            timestep_action = torch.ones((batch_size, self.action_horizon), device=device, dtype=torch.int64) * action_timestep

            if self.should_run_model(index, current_timestep, prev_predictions):
                predictions = self._run_diffusion_steps(
                    noisy_input=noisy_input.transpose(1, 2),
                    timestep=timestep,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    state=policy_inputs["state_features"],
                    embodiment_id=policy_inputs["embodiment_id"],
                    context=policy_inputs["prompt_embs"],
                    seq_len=seq_len,
                    y=policy_inputs["ys"][:, :, : self.num_frame_per_block],
                    clip_feature=policy_inputs["clip_feas"],
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    kv_cache_metadata={"start_frame": 0, "update_kv_cache": False},
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
                model_output=flow_pred.transpose(1, 2),
                timestep=video_timestep,
                sample=noisy_input,
                step_index=index,
                return_dict=False,
            )[0]
            noisy_input_action = sample_scheduler_action.step(
                model_output=flow_pred_cond_action,
                timestep=action_timestep,
                sample=noisy_input_action,
                step_index=index,
                return_dict=False,
            )[0]

        return BatchFeature(data={"action_pred": noisy_input_action, "video_pred": noisy_input.transpose(1, 2)})

    def _normalize_videos(self, videos: torch.Tensor) -> torch.Tensor:
        """Convert input videos to `[B, C, T, H, W]` in `[-1, 1]`."""
        videos = rearrange(videos, "b t h w c -> b c t h w")
        if videos.dtype == torch.uint8:
            videos = videos.float().div(255.0)
            bsz, channels, num_frames, height, width = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4).reshape(bsz * num_frames, channels, height, width)
            videos = self.normalize_video(videos)
            videos = videos.reshape(bsz, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
        return videos.to(dtype=torch.bfloat16)
