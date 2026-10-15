"""Deep learning model architectures for hyperspectral remote sensing."""

from terraspectra.models.cnn3d import (
    Hyperspectral3DCNN,
    Hyperspectral3DCNNConfig,
    build_3d_cnn,
)

__all__ = [
    "Hyperspectral3DCNN",
    "Hyperspectral3DCNNConfig",
    "build_3d_cnn",
]
