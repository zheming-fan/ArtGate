"""ArtGate detector model."""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.nn import functional as F
from transformers import CLIPConfig, CLIPModel

from .artifact_branch import freq_resnet50
from .clip_vision import encode_clip_image

def build_clip_vit_large_patch14() -> CLIPModel:
    """Create the released model's CLIP structure without pretrained weights.

    These are the architecture fields from ``openai/clip-vit-large-patch14``.
    Keeping them locally means evaluation never contacts Hugging Face and does
    not require a separate CLIP directory. All parameters are subsequently
    replaced by the final ArtGate checkpoint.
    """
    config = CLIPConfig(
        projection_dim=768,
        logit_scale_init_value=2.6592,
        text_config={
            "vocab_size": 49408,
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "max_position_embeddings": 77,
            "hidden_act": "quick_gelu",
            "layer_norm_eps": 1e-5,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "eos_token_id": 2,
            "pad_token_id": 1,
        },
        vision_config={
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "image_size": 224,
            "patch_size": 14,
            "hidden_act": "quick_gelu",
            "layer_norm_eps": 1e-5,
            "attention_dropout": 0.0,
        },
    )
    return CLIPModel(config)


class ArtGateCLIP(nn.Module):
    """CLIP detector augmented with frequency-domain artifact tokens."""

    def __init__(
        self,
        *,
        num_classes: int = 1,
        artifact_token_count: int = 32,
        injection_layer: int = 23,
    ):
        super().__init__()
        self.injection_layer = injection_layer
        self.artifact_token_count = artifact_token_count

        self.model = build_clip_vit_large_patch14()
        self.resnetmodel = freq_resnet50()

        hidden_size = self.model.config.vision_config.hidden_size
        projection_size = self.model.config.projection_dim
        self.fc0 = nn.Linear(512, artifact_token_count * hidden_size)
        self.fc = nn.Linear(projection_size, num_classes)

        # Retained as an alias for compatibility with the released checkpoints.
        self.logit_scale = self.model.logit_scale

        prefix = f"vision_model.encoder.layers.{injection_layer}"
        lora_config = LoraConfig(
            # The final checkpoint replaces all adapter parameters, so avoid
            # the expensive PiSSA SVD initialization used during training.
            init_lora_weights=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.v_proj",
                f"{prefix}.self_attn.q_proj",
                f"{prefix}.self_attn.out_proj",
                f"{prefix}.mlp.fc1",
                f"{prefix}.mlp.fc2",
                "visual_projection",
                "text_projection",
            ],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        self.model = get_peft_model(self.model, lora_config)
        self.lora_parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if "lora" in name
        ]

    def _artifact_tokens(self, artifact_images: torch.Tensor):
        features = self.resnetmodel.get_features(artifact_images)
        tokens = self.fc0(features).reshape(
            features.shape[0], self.artifact_token_count, -1
        )
        enabled = self.resnetmodel(artifact_images).reshape(-1).sigmoid() > 0.5
        return tokens, enabled

    def encode_image(
        self, images: torch.Tensor, artifact_images: torch.Tensor
    ) -> torch.Tensor:
        artifact_tokens, enabled = self._artifact_tokens(artifact_images)

        if bool(enabled.all()):
            return encode_clip_image(
                self.model,
                images,
                artifact_tokens,
                injection_layer=self.injection_layer,
            )
        if not bool(enabled.any()):
            return encode_clip_image(
                self.model, images, injection_layer=self.injection_layer
            )

        # A batch can contain both real and generated images. Encode each group
        # with the correct sequence length, then restore the original ordering.
        plain = encode_clip_image(
            self.model, images[~enabled], injection_layer=self.injection_layer
        )
        augmented = encode_clip_image(
            self.model,
            images[enabled],
            artifact_tokens[enabled],
            injection_layer=self.injection_layer,
        )
        output = images.new_empty((images.shape[0], plain.shape[-1]))
        output[~enabled] = plain
        output[enabled] = augmented
        return output

    def forward(
        self, images: torch.Tensor, artifact_images: torch.Tensor
    ) -> torch.Tensor:
        image_features = self.encode_image(images, artifact_images)
        return self.fc(F.normalize(image_features, p=2, dim=-1))


# Backward-compatible import/class name used by the released evaluation code.
ArtGate_CLIP = ArtGateCLIP
