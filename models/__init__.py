"""Model definitions owned by ArtGate."""

from .artgate import ArtGateCLIP, ArtGate_CLIP, build_clip_vit_large_patch14
from .artifact_branch import freq_resnet50

__all__ = [
    "ArtGateCLIP",
    "ArtGate_CLIP",
    "build_clip_vit_large_patch14",
    "freq_resnet50",
]
