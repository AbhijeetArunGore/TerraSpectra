"""CNN–ViT Hybrid Fusion Network for hyperspectral spatial-spectral classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from terraspectra.models.cnn3d import Hyperspectral3DCNN, Hyperspectral3DCNNConfig
from terraspectra.models.vit import HyperspectralViT, HyperspectralViTConfig

logger = logging.getLogger(__name__)


@dataclass
class HyperspectralHybridConfig:
    """
    Configuration parameters for the CNN-ViT Hybrid Fusion model.

    Parameters
    ----------
    in_channels : int
        Number of channel dimensions (typically 1 for hyperspectral volumes).
    spectral_depth : int
        Number of spectral bands / components (e.g., 15 for PCA-reduced, 204 for raw).
    spatial_size : int
        Spatial window height and width (e.g., 7 for a 7x7 patch).
    num_classes : int
        Number of output classification classes (e.g., 16 for Salinas land-cover classes).
    cnn_conv_channels : Tuple[int, ...]
        Output channels for 3D convolution stages in the CNN branch (default: (8, 16, 32)).
    cnn_kernel_spectral : Tuple[int, ...]
        Spectral kernel sizes for 3D convolution stages (default: (7, 5, 3)).
    vit_embed_dim : int
        Latent token embedding dimension for the ViT branch (default: 64).
    vit_num_heads : int
        Number of multi-head self-attention heads (default: 4).
    vit_num_layers : int
        Number of Transformer encoder blocks in the ViT branch (default: 2).
    vit_mlp_dim : int
        Hidden dimension of feed-forward MLP inside Transformer blocks (default: 128).
    vit_pooling : str
        ViT token pooling method: 'mean' (default) or 'cls'.
    fusion_dim : int
        Dimensionality of the projected unified representation after concatenation (default: 64).
    dropout_rate : float
        Dropout probability applied across the fusion projection and dense head (default: 0.2).
    """

    in_channels: int = 1
    spectral_depth: int = 15
    spatial_size: int = 7
    num_classes: int = 16
    cnn_conv_channels: Tuple[int, ...] = (8, 16, 32)
    cnn_kernel_spectral: Tuple[int, ...] = (7, 5, 3)
    vit_embed_dim: int = 64
    vit_num_heads: int = 4
    vit_num_layers: int = 2
    vit_mlp_dim: int = 128
    vit_pooling: str = "mean"
    fusion_dim: int = 64
    dropout_rate: float = 0.2

    def __post_init__(self) -> None:
        """Validate architectural hyperparameters."""
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {self.in_channels}")
        if self.spectral_depth < 1:
            raise ValueError(f"spectral_depth must be >= 1, got {self.spectral_depth}")
        if self.spatial_size < 3 or self.spatial_size % 2 == 0:
            raise ValueError(
                f"spatial_size must be a positive odd integer >= 3 (e.g. 5, 7, 9), got {self.spatial_size}"
            )
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if len(self.cnn_conv_channels) != len(self.cnn_kernel_spectral):
            raise ValueError(
                f"cnn_conv_channels length ({len(self.cnn_conv_channels)}) must match "
                f"cnn_kernel_spectral length ({len(self.cnn_kernel_spectral)})"
            )
        if self.vit_embed_dim < 1:
            raise ValueError(f"vit_embed_dim must be >= 1, got {self.vit_embed_dim}")
        if self.vit_num_heads < 1:
            raise ValueError(f"vit_num_heads must be >= 1, got {self.vit_num_heads}")
        if self.vit_embed_dim % self.vit_num_heads != 0:
            raise ValueError(
                f"vit_embed_dim ({self.vit_embed_dim}) must be divisible by "
                f"vit_num_heads ({self.vit_num_heads})"
            )
        if self.vit_num_layers < 1:
            raise ValueError(f"vit_num_layers must be >= 1, got {self.vit_num_layers}")
        if self.vit_mlp_dim < 1:
            raise ValueError(f"vit_mlp_dim must be >= 1, got {self.vit_mlp_dim}")
        if self.fusion_dim < 1:
            raise ValueError(f"fusion_dim must be >= 1, got {self.fusion_dim}")
        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {self.dropout_rate}")
        if self.vit_pooling not in ("mean", "cls"):
            raise ValueError(f"vit_pooling must be 'mean' or 'cls', got '{self.vit_pooling}'")


class HyperspectralHybridCNNViT(nn.Module):
    """
    CNN–ViT Hybrid Fusion Network for joint spatial-spectral hyperspectral classification.

    Architectural Flow:
        Input Patch X: (B, C, D, H, W)
             │
             ├──► 3D-CNN Branch: extract_features(X) ──────────► CNN Feature Vector (B, D_cnn)
             │                                                         │
             └──► Vision Transformer Branch: extract_features(X) ──► ViT Feature Vector (B, D_vit)
                                                                       │
             ┌─────────────────────────────────────────────────────────┘
             ▼
        Feature Concatenation: (B, D_cnn + D_vit)
             │
             ▼
        Fusion Projection: Linear(D_cnn + D_vit -> fusion_dim)
                           -> LayerNorm(fusion_dim)
                           -> GELU()
                           -> Dropout(dropout_rate)
             │
             ▼ Unified Fused Representation (B, fusion_dim)
             │
             ▼
        Classification Head: Linear(fusion_dim -> num_classes)
             │
             ▼ Output Logits: (B, num_classes)

    Branch Roles:
        - 3D-CNN Branch: Captures local spatial textures, continuous spectral profiles,
          and translation-equivariant fine-grained edge transitions via 3D convolutions.
        - ViT Branch: Captures global pairwise interactions and long-range spatial context
          across all 49 patch pixels via multi-head self-attention.
        - Fusion Layer: Harmonizes both modalities into a dense, unified latent vector
          prior to class logit prediction.
    """

    def __init__(
        self,
        config: Optional[HyperspectralHybridConfig] = None,
        cnn_branch: Optional[Hyperspectral3DCNN] = None,
        vit_branch: Optional[HyperspectralViT] = None,
        **kwargs,
    ):
        super().__init__()

        if config is None:
            config = HyperspectralHybridConfig(**kwargs)
        elif kwargs:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        self.config = config

        # 1. CNN Branch
        if cnn_branch is not None:
            self.cnn_branch = cnn_branch
        else:
            cnn_cfg = Hyperspectral3DCNNConfig(
                in_channels=config.in_channels,
                spectral_depth=config.spectral_depth,
                spatial_size=config.spatial_size,
                num_classes=config.num_classes,
                conv_channels=config.cnn_conv_channels,
                kernel_spectral=config.cnn_kernel_spectral,
                dropout_rate=config.dropout_rate,
            )
            self.cnn_branch = Hyperspectral3DCNN(cnn_cfg)

        # 2. ViT Branch
        if vit_branch is not None:
            self.vit_branch = vit_branch
        else:
            vit_cfg = HyperspectralViTConfig(
                in_channels=config.in_channels,
                spectral_depth=config.spectral_depth,
                spatial_size=config.spatial_size,
                num_classes=config.num_classes,
                embed_dim=config.vit_embed_dim,
                num_heads=config.vit_num_heads,
                num_layers=config.vit_num_layers,
                mlp_dim=config.vit_mlp_dim,
                dropout_rate=config.dropout_rate,
                pooling=config.vit_pooling,
            )
            self.vit_branch = HyperspectralViT(vit_cfg)

        # Branch feature dimensions
        self.cnn_feature_dim: int = self.cnn_branch.feature_dim
        self.vit_feature_dim: int = self.vit_branch.feature_dim
        self.raw_fusion_dim: int = self.cnn_feature_dim + self.vit_feature_dim

        # 3. Fusion Projection Layer
        # Uses LayerNorm instead of BatchNorm1d to ensure consistent execution
        # across all batch sizes, including batch_size=1 during inference.
        self.fusion_projection = nn.Sequential(
            nn.Linear(self.raw_fusion_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(p=config.dropout_rate),
        )

        # 4. Final Classification Head: maps fused latent vector to unnormalized class logits
        self.classifier = nn.Linear(config.fusion_dim, config.num_classes)

        self._init_fusion_weights()

    def _init_fusion_weights(self) -> None:
        """Initialize fusion projection and classifier weights."""
        linear_proj = self.fusion_projection[0]
        if isinstance(linear_proj, nn.Linear):
            nn.init.xavier_uniform_(linear_proj.weight)
            if linear_proj.bias is not None:
                nn.init.zeros_(linear_proj.bias)

        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def _validate_input_tensor(self, x: torch.Tensor) -> None:
        """Validate input tensor dimensionality, rank, channels, depth, and spatial bounds."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be torch.Tensor, got {type(x)}")

        if x.ndim != 5:
            raise ValueError(
                f"HyperspectralHybridCNNViT expects a 5D tensor (batch, channels, depth, height, width), "
                f"got {x.ndim}D tensor with shape {tuple(x.shape)}"
            )

        B, C, D, H, W = x.shape

        if C != self.config.in_channels:
            raise ValueError(
                f"Input channel dimension C={C} does not match configured in_channels={self.config.in_channels}"
            )

        if D != self.config.spectral_depth:
            raise ValueError(
                f"Input spectral depth D={D} does not match configured spectral_depth={self.config.spectral_depth}"
            )

        if H != self.config.spatial_size or W != self.config.spatial_size:
            raise ValueError(
                f"Input spatial dimensions ({H}, {W}) do not match configured spatial_size={self.config.spatial_size}"
            )

    def extract_branch_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract feature representations independently from both CNN and ViT branches.

        Parameters
        ----------
        x : torch.Tensor
            5D tensor of shape (batch, channels, depth, height, width).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            - CNN feature tensor of shape (batch, cnn_feature_dim)
            - ViT feature tensor of shape (batch, vit_feature_dim)
        """
        self._validate_input_tensor(x)
        cnn_feat = self.cnn_branch.extract_features(x)
        vit_feat = self.vit_branch.extract_features(x)
        return cnn_feat, vit_feat

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Execute forward pass producing raw classification logits.

        Parameters
        ----------
        x : torch.Tensor
            5D tensor of shape (batch, channels, depth, height, width).
        return_features : bool
            If True, additionally returns a dictionary containing intermediate
            branch and fused feature representations for introspection.

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            If return_features is False (standard model contract):
                Logits of shape (batch, num_classes).
            If return_features is True:
                Tuple of (Logits, feature_dict) where feature_dict contains:
                    - 'cnn_features': (batch, cnn_feature_dim)
                    - 'vit_features': (batch, vit_feature_dim)
                    - 'fused_features': (batch, fusion_dim)
        """
        # 1. Extract branch representations
        cnn_feat, vit_feat = self.extract_branch_features(x)

        # 2. Concatenate along feature dimension: (B, D_cnn) + (B, D_vit) -> (B, D_cnn + D_vit)
        concat_feat = torch.cat([cnn_feat, vit_feat], dim=-1)

        # 3. Fusion Projection: (B, D_cnn + D_vit) -> (B, fusion_dim)
        fused_feat = self.fusion_projection(concat_feat)

        # 4. Classification Head: (B, fusion_dim) -> (B, num_classes)
        logits = self.classifier(fused_feat)

        if return_features:
            features = {
                "cnn_features": cnn_feat,
                "vit_features": vit_feat,
                "fused_features": fused_feat,
            }
            return logits, features
        return logits

    def count_parameters(self) -> Dict[str, Any]:
        """
        Compute total, trainable, and per-component parameter breakdowns.

        Returns
        -------
        Dict[str, Any]
            Dictionary detailing parameter counts for CNN branch, ViT branch,
            fusion projection, classifier head, and total network.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable

        cnn_total = sum(p.numel() for p in self.cnn_branch.parameters())
        vit_total = sum(p.numel() for p in self.vit_branch.parameters())
        fusion_total = sum(p.numel() for p in self.fusion_projection.parameters())
        classifier_total = sum(p.numel() for p in self.classifier.parameters())

        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
            "component_breakdown": {
                "cnn_branch_parameters": cnn_total,
                "vit_branch_parameters": vit_total,
                "fusion_projection_parameters": fusion_total,
                "classifier_parameters": classifier_total,
            },
        }


def build_hybrid(
    spectral_depth: int = 15,
    spatial_size: int = 7,
    num_classes: int = 16,
    in_channels: int = 1,
    cnn_conv_channels: Tuple[int, ...] = (8, 16, 32),
    cnn_kernel_spectral: Tuple[int, ...] = (7, 5, 3),
    vit_embed_dim: int = 64,
    vit_num_heads: int = 4,
    vit_num_layers: int = 2,
    vit_mlp_dim: int = 128,
    vit_pooling: str = "mean",
    fusion_dim: int = 64,
    dropout_rate: float = 0.2,
) -> HyperspectralHybridCNNViT:
    """Convenience factory function to instantiate HyperspectralHybridCNNViT."""
    config = HyperspectralHybridConfig(
        in_channels=in_channels,
        spectral_depth=spectral_depth,
        spatial_size=spatial_size,
        num_classes=num_classes,
        cnn_conv_channels=cnn_conv_channels,
        cnn_kernel_spectral=cnn_kernel_spectral,
        vit_embed_dim=vit_embed_dim,
        vit_num_heads=vit_num_heads,
        vit_num_layers=vit_num_layers,
        vit_mlp_dim=vit_mlp_dim,
        vit_pooling=vit_pooling,
        fusion_dim=fusion_dim,
        dropout_rate=dropout_rate,
    )
    return HyperspectralHybridCNNViT(config)
