"""Feature fusion transformers for combining multiscale descriptors."""

from adaptive_cgr_vision.fusion.fixed import FixedFusionTransformer
from adaptive_cgr_vision.fusion.adaptive import AdaptiveFusionTransformer

__all__ = [
    "FixedFusionTransformer",
    "AdaptiveFusionTransformer",
]
