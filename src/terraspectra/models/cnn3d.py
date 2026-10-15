"""3D-CNN baseline model architecture for hyperspectral spatial-spectral classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class Hyperspectral3DCNNConfig:
    """Configuration hyperparameters for Hyperspectral3DCNN."""
    in_channels: int = 1                     # Typically 1 for raw hyperspectral volumes
    spectral_depth: int = 204               # Number of spectral channels (e.g. 204 raw, 15 PCA)
    spatial_size: int = 7                   # Spatial window height & width (e.g. 5, 7, 9)
    num_classes: int = 17                   # Number of output classes (Salinas: 0..16)
    conv_channels: Tuple[int, ...] = (8, 16, 32)  # Feature channels per 3D conv block
    kernel_spectral: Tuple[int, ...] = (7, 5, 3)  # Spectral kernel sizes across blocks
    kernel_spatial: int = 3                 # Spatial kernel diameter
    dropout_rate: float = 0.3               # Dropout probability in classification head
    fc_hidden: int = 64                     # Hidden units in classification head

    def __post_init__(self):
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {self.in_channels}")
        if self.spectral_depth < 1:
            raise ValueError(f"spectral_depth must be >= 1, got {self.spectral_depth}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if len(self.conv_channels) != len(self.kernel_spectral):
            raise ValueError(
                f"conv_channels length ({len(self.conv_channels)}) must match kernel_spectral length ({len(self.kernel_spectral)})"
            )
        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {self.dropout_rate}")



class Hyperspectral3DCNN(nn.Module):
    """
    3D Convolutional Neural Network baseline for joint spatial-spectral feature extraction.

    Expected input tensor layout:
        (batch, channels, depth, height, width)
        where:
        - channels: feature channels (typically 1)
        - depth: spectral dimension (contiguous wavelengths or PCA components)
        - height: spatial patch height (odd integer, e.g. 7)
        - width: spatial patch width (odd integer, e.g. 7)

    Produces output logits:
        (batch, num_classes)
    """

    def __init__(self, config: Optional[Hyperspectral3DCNNConfig] = None, **kwargs):
        super().__init__()

        if config is None:
            # Allow kwargs override of config defaults
            config = Hyperspectral3DCNNConfig(**kwargs)
        elif kwargs:
            # If both config and kwargs provided, override config attributes
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        self._validate_config(config)
        self.config = config

        # Construct sequential 3D Convolution blocks
        # Each block performs Conv3D -> BatchNorm3d -> ReLU
        layers = []
        current_in = config.in_channels

        for i, (out_ch, k_spec) in enumerate(zip(config.conv_channels, config.kernel_spectral)):
            # Spatial padding=1 preserves (H, W) for kernel=3
            # Spectral padding is 0 to progressively distill spectral features
            spatial_pad = config.kernel_spatial // 2
            layers.append(
                nn.Conv3d(
                    in_channels=current_in,
                    out_channels=out_ch,
                    kernel_size=(k_spec, config.kernel_spatial, config.kernel_spatial),
                    stride=1,
                    padding=(0, spatial_pad, spatial_pad),
                    bias=False,  # BatchNorm handles bias
                )
            )
            layers.append(nn.BatchNorm3d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            current_in = out_ch

        self.features = nn.Sequential(*layers)

        # Global Spatial-Spectral Adaptive Pooling
        # Reduces arbitrary (B, C_last, D_remaining, H, W) -> (B, C_last, 1, 1, 1)
        # This makes the classification head invariant to the input spectral depth (204 vs 15)
        # and spatial patch size (5, 7, 9) without requiring hardcoded dense weight matrices.
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Classification Head: Flatten -> Linear -> ReLU -> Dropout -> Linear -> Logits
        final_conv_ch = config.conv_channels[-1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_conv_ch, config.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.fc_hidden, config.num_classes),
        )

    @staticmethod
    def _validate_config(config: Hyperspectral3DCNNConfig) -> None:
        """Validate model architectural hyperparameters."""
        if config.in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {config.in_channels}")
        if config.spectral_depth < 1:
            raise ValueError(f"spectral_depth must be >= 1, got {config.spectral_depth}")
        if config.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {config.num_classes}")
        if len(config.conv_channels) != len(config.kernel_spectral):
            raise ValueError(
                f"conv_channels length ({len(config.conv_channels)}) must match kernel_spectral length ({len(config.kernel_spectral)})"
            )
        if not (0.0 <= config.dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {config.dropout_rate}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing raw classification logits.

        Parameters
        ----------
        x : torch.Tensor
            5D tensor of shape (batch, channels, depth, height, width).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")

        if x.ndim != 5:
            raise ValueError(
                f"3D-CNN expects a 5-dimensional tensor (batch, channels, depth, height, width), "
                f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
            )

        # Feature extraction through 3D convolutions
        feat = self.features(x)

        # Global spatial-spectral pooling: (B, C, D', H', W') -> (B, C, 1, 1, 1)
        pooled = self.global_pool(feat)

        # Classification head: (B, C, 1, 1, 1) -> (B, num_classes)
        logits = self.classifier(pooled)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Calculate total, trainable, and non-trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
        }


def build_3d_cnn(
    spectral_depth: int = 204,
    spatial_size: int = 7,
    num_classes: int = 17,
    in_channels: int = 1,
    conv_channels: Tuple[int, ...] = (8, 16, 32),
    kernel_spectral: Tuple[int, ...] = (7, 5, 3),
    dropout_rate: float = 0.3,
) -> Hyperspectral3DCNN:
    """Convenience factory function to instantiate Hyperspectral3DCNN with custom parameters."""
    config = Hyperspectral3DCNNConfig(
        in_channels=in_channels,
        spectral_depth=spectral_depth,
        spatial_size=spatial_size,
        num_classes=num_classes,
        conv_channels=conv_channels,
        kernel_spectral=kernel_spectral,
        dropout_rate=dropout_rate,
    )
    return Hyperspectral3DCNN(config)
