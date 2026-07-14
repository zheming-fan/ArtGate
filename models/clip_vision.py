"""ArtGate's artifact-token extension for an upstream Hugging Face CLIP model.

The project previously carried a complete, modified copy of ``transformers``
only to add this behavior to CLIP.  Keeping the extension here makes the
ownership boundary explicit and lets the rest of Transformers come from the
normal third-party dependency.
"""

from __future__ import annotations

from typing import Optional

import torch


def _forward_layer_with_artifacts(
    layer,
    hidden_states: torch.Tensor,
    artifact_tokens: torch.Tensor,
    *,
    output_attentions: bool = False,
):
    """Run one CLIP encoder layer, appending artifact tokens to its residual.

    This intentionally preserves the checkpoint's original fusion semantics:
    artifact tokens enter through attention while their initial residual is
    zero.  No Transformers source code needs to be patched.
    """
    if hidden_states.ndim != 3 or artifact_tokens.ndim != 3:
        raise ValueError("hidden states and artifact tokens must both be rank-3 tensors")
    if hidden_states.shape[0] != artifact_tokens.shape[0]:
        raise ValueError("hidden states and artifact tokens must have the same batch size")
    if hidden_states.shape[-1] != artifact_tokens.shape[-1]:
        raise ValueError(
            "artifact token width must match CLIP hidden width: "
            f"{artifact_tokens.shape[-1]} != {hidden_states.shape[-1]}"
        )

    residual = torch.cat(
        (hidden_states, torch.zeros_like(artifact_tokens)), dim=1
    )
    hidden_states = torch.cat((hidden_states, artifact_tokens), dim=1)
    hidden_states = layer.layer_norm1(hidden_states)
    hidden_states, attention = layer.self_attn(
        hidden_states=hidden_states,
        attention_mask=None,
        causal_attention_mask=None,
        output_attentions=output_attentions,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = layer.layer_norm2(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return (hidden_states, attention) if output_attentions else (hidden_states,)


def encode_clip_image(
    clip_model,
    pixel_values: torch.Tensor,
    artifact_tokens: Optional[torch.Tensor] = None,
    *,
    injection_layer: int = 23,
) -> torch.Tensor:
    """Encode images with upstream CLIP and optionally inject artifact tokens."""
    vision = clip_model.vision_model
    layers = vision.encoder.layers
    if not 0 <= injection_layer < len(layers):
        raise ValueError(
            f"injection layer {injection_layer} is outside CLIP's {len(layers)} layers"
        )

    hidden_states = vision.embeddings(pixel_values)
    hidden_states = vision.pre_layrnorm(hidden_states)

    for index, layer in enumerate(layers):
        if index == injection_layer and artifact_tokens is not None:
            hidden_states = _forward_layer_with_artifacts(
                layer, hidden_states, artifact_tokens
            )[0]
        else:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False,
            )[0]

    pooled_output = vision.post_layernorm(hidden_states[:, 0, :])
    return clip_model.visual_projection(pooled_output)
