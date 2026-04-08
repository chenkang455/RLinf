import torch
from einops import rearrange
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
    WANPolicyHead,
    WANPolicyHeadConfig,
)


class DreamZeroActionHead(WANPolicyHead):
    """RLinf-local override point for DreamZero lazy inference."""

    config_class = WANPolicyHeadConfig


    def lazy_joint_video_action(self, backbone_output: BatchFeature, action_input: BatchFeature, latent_video=None) -> BatchFeature:
        """Run local lazy inference implementation."""
        return self.sample_action_video(backbone_output=backbone_output, action_input=action_input)

    def sample_action_video(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """Sample action video from DreamZero."""
        self.set_frozen_modules_to_eval_mode()
        self.prepare_policy_inputs(data=action_input)
        return 

    # ===================================================================
    # ====================== Prepare policy inputs ======================
    # ===================================================================

    def prepare_policy_inputs(self, data: BatchFeature) -> BatchFeature:
        """Prepare single-frame DreamZero policy inputs."""
        # embodiment id input
        embodiment_id = data.embodiment_id
        # state features input
        state_features = data.state.to(dtype=torch.bfloat16)
        # language input
        self.language = data["text"]
        text_inputs = self._prepare_text_inputs(data)
        prompt_embs = [self.encode_prompt(text, attention_mask) for text, attention_mask in text_inputs]
        # video input
        videos = self._normalize_videos(data["images"])
        _, _, _, height, width = videos.shape
        image = videos[:, :, :1].transpose(1, 2)
        clip_feas, ys, image_latents = self.encode_image(image, self.num_frames, height, width)
        self.clip_feas = clip_feas.to(dtype=image_latents.dtype)
        self.ys = ys.to(dtype=image_latents.dtype)
        return BatchFeature(
            data={
                "videos": videos,
                "embodiment_id": embodiment_id,
                "state_features": state_features,
                "prompt_embs": prompt_embs,
                "image_latents": image_latents,
                "clip_feas": self.clip_feas,
                "ys": self.ys,
                "height": height,
                "width": width,
            }
        )

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
