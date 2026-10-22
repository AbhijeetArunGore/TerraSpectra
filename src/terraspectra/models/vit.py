"""Vision Transformer (ViT) architecture, spatial tokenization, and self-attention for hyperspectral classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class HyperspectralViTConfig:
    """
    Configuration parameters for HyperspectralViT.

    Parameters
    ----------
    in_channels : int
        Number of channel dimensions (typically 1 for hyperspectral volumes).
    spectral_depth : int
        Number of spectral bands / components (e.g., 15 for PCA-reduced, 204 for raw).
    spatial_size : int
        Spatial window height and width (e.g., 7 for a 7x7 patch).
    num_classes : int
        Number of output classification classes (e.g., 16 for Salinas labeled classes).
    embed_dim : int
        Dimensionality of projected token embeddings (e.g., 64).
    num_heads : int
        Number of parallel attention heads (embed_dim must be divisible by num_heads).
    num_layers : int
        Number of stacked Transformer encoder blocks.
    mlp_dim : int
        Hidden dimension of the feed-forward MLP inside each encoder block.
    dropout_rate : float
        Dropout probability applied across attention, projections, and MLP.
    pooling : str
        Token aggregation strategy for classification: 'mean' (default) or 'cls'.
    """

    in_channels: int = 1
    spectral_depth: int = 15
    spatial_size: int = 7
    num_classes: int = 16
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 128
    dropout_rate: float = 0.1
    pooling: str = "mean"

    def __post_init__(self) -> None:
        """Validate architectural hyperparameters."""
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {self.in_channels}")
        if self.spectral_depth < 1:
            raise ValueError(f"spectral_depth must be >= 1, got {self.spectral_depth}")
        if self.spatial_size < 1 or self.spatial_size % 2 == 0:
            raise ValueError(
                f"spatial_size must be a positive odd integer (e.g. 3, 5, 7), got {self.spatial_size}"
            )
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if self.embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {self.embed_dim}")
        if self.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {self.num_heads}")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.mlp_dim < 1:
            raise ValueError(f"mlp_dim must be >= 1, got {self.mlp_dim}")
        if not (0.0 <= self.dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {self.dropout_rate}")
        if self.pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got '{self.pooling}'")


class PatchTokenizer(nn.Module):
    """
    Tokenizes a 5D hyperspectral patch into a sequence of spatial tokens.

    Input tensor shape:
        (B, C, D, H, W)
    where:
        - B: Batch size
        - C: In-channels (typically 1)
        - D: Spectral depth (e.g., 15 PCA components)
        - H: Spatial height (e.g., 7)
        - W: Spatial width (e.g., 7)

    Tokenization Strategy:
        Treats each spatial coordinate (h, w) as an individual token whose input
        feature vector comprises all C * D spectral values at that location.
        A linear projection maps each spectral profile of dimension (C * D)
        into the latent embedding dimension E.

    Output token sequence shape:
        (B, N, E) where N = H * W (e.g., 7 * 7 = 49 tokens)

    Design Trade-off:
        This spatial tokenization preserves 2D localized pixel coordinates while
        allowing multi-head self-attention to model long-range spatial dependencies
        and canopy context across the patch. It avoids quadratic explosion in token
        count that would occur if every individual wavelength were treated as a token.
    """

    def __init__(self, in_channels: int, spectral_depth: int, embed_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.spectral_depth = spectral_depth
        self.embed_dim = embed_dim
        self.in_features = in_channels * spectral_depth

        self.projection = nn.Linear(self.in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project 5D hyperspectral volume into a sequence of token embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (B, C, D, H, W).

        Returns
        -------
        torch.Tensor
            Token sequence of shape (B, H * W, embed_dim).
        """
        B, C, D, H, W = x.shape
        # Permute spatial dimensions to sequence positions and combine spectral features:
        # (B, C, D, H, W) -> (B, H, W, C, D) -> (B, H * W, C * D)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x = x.view(B, H * W, C * D)
        # Linear projection to latent embedding dimension E
        tokens = self.projection(x)
        return tokens


class TransformerEncoderBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Encoder Block with Multi-Head Self-Attention.

    Architecture:
        x -> LayerNorm -> MultiHeadAttention -> (+ Residual) ->
             LayerNorm -> FeedForward MLP   -> (+ Residual) -> Output

    Self-Attention Formulation:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
        where:
            - Q: Query projection (what a token is looking for)
            - K: Key projection (what a token offers / advertises)
            - V: Value projection (the actual content to be aggregated)
            - sqrt(d_k): Scaling factor preventing softmax saturation for large d_k
            - softmax: Probability distribution over key tokens summing to 1.0
            - Weighted sum: Content aggregated from all tokens proportional to affinity
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Pre-LayerNorm for attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        # Pre-LayerNorm for feed-forward network
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Execute forward pass through the encoder block.

        Parameters
        ----------
        x : torch.Tensor
            Input tokens of shape (B, N, embed_dim).
        need_weights : bool
            If True, returns per-head self-attention weight matrices.

        Returns
        -------
        Tuple[torch.Tensor, Optional[torch.Tensor]]
            - Output tokens of shape (B, N, embed_dim)
            - Attention weights of shape (B, num_heads, N, N) if need_weights=True, else None.
        """
        # Pre-LN attention branch
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm,
            x_norm,
            x_norm,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        x = x + self.dropout1(attn_out)

        # Pre-LN MLP branch
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights


class HyperspectralViT(nn.Module):
    """
    Vision Transformer (ViT) model for hyperspectral patch land-cover classification.

    Compatible with standard 5D hyperspectral input batches:
        (batch_size, in_channels, spectral_depth, height, width)

    Produces unnormalized raw logits:
        (batch_size, num_classes)
    """

    def __init__(self, config: Optional[HyperspectralViTConfig] = None, **kwargs):
        super().__init__()

        if config is None:
            config = HyperspectralViTConfig(**kwargs)
        elif kwargs:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)

        self.config = config
        self.num_spatial_tokens = config.spatial_size * config.spatial_size

        # 1. Hyperspectral Tokenizer
        self.tokenizer = PatchTokenizer(
            in_channels=config.in_channels,
            spectral_depth=config.spectral_depth,
            embed_dim=config.embed_dim,
        )

        # 2. Optional CLS token and Positional Embeddings
        if config.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            total_tokens = self.num_spatial_tokens + 1
        else:
            self.cls_token = None
            total_tokens = self.num_spatial_tokens

        self.pos_embedding = nn.Parameter(torch.randn(1, total_tokens, config.embed_dim) * 0.02)
        self.pos_dropout = nn.Dropout(config.dropout_rate)

        # 3. Stack of Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_dim=config.mlp_dim,
                dropout_rate=config.dropout_rate,
            )
            for _ in range(config.num_layers)
        ])

        # 4. Final LayerNorm & Classification Head
        self.norm = nn.LayerNorm(config.embed_dim)
        self.classifier = nn.Linear(config.embed_dim, config.num_classes)

        # Storage for educational inspection of attention weights
        self.last_attention_weights: Optional[List[torch.Tensor]] = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with standard truncated normal / Xavier distributions."""
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def _validate_input_tensor(self, x: torch.Tensor) -> None:
        """Validate input tensor shape, rank, and dimensionality."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be torch.Tensor, got {type(x)}")

        if x.ndim != 5:
            raise ValueError(
                f"HyperspectralViT expects a 5D tensor (batch, channels, depth, height, width), "
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
                f"Input spatial dimensions ({H}, {W}) do not match configured spatial_size={self.config.spatial_size}. "
                f"Positional embeddings require fixed spatial dimension."
            )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Execute forward pass producing raw classification logits.

        Parameters
        ----------
        x : torch.Tensor
            5D tensor of shape (B, C, D, H, W).
        return_attention : bool
            If True, additionally returns list of attention weight matrices from each encoder layer.

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, List[torch.Tensor]]
            If return_attention is False (standard model contract):
                Logits of shape (B, num_classes).
            If return_attention is True:
                Tuple of (Logits, attention_weights_per_layer) where each element
                has shape (B, num_heads, seq_len, seq_len).
        """
        self._validate_input_tensor(x)

        B = x.shape[0]

        # 1. Hyperspectral Tokenization: (B, C, D, H, W) -> (B, N, embed_dim)
        tokens = self.tokenizer(x)

        # 2. Append CLS token if configured
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        # 3. Add Positional Embeddings
        x_seq = self.pos_dropout(tokens + self.pos_embedding)

        # 4. Pass through Transformer Encoder Blocks
        attn_weights_list: List[torch.Tensor] = []
        for block in self.blocks:
            x_seq, w = block(x_seq, need_weights=return_attention)
            if return_attention and w is not None:
                attn_weights_list.append(w)

        # Cache last attention weights for inspection
        self.last_attention_weights = attn_weights_list if return_attention else None

        # 5. Final LayerNorm
        x_seq = self.norm(x_seq)

        # 6. Token Pooling
        if self.config.pooling == "cls":
            pooled = x_seq[:, 0]  # CLS token representation
        else:
            pooled = x_seq.mean(dim=1)  # Mean pooling over spatial tokens

        # 7. Classification Head: produces unnormalized logits
        logits = self.classifier(pooled)

        if return_attention:
            return logits, attn_weights_list
        return logits

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Convenience method to retrieve per-layer attention matrices without altering model state."""
        was_training = self.training
        self.eval()
        with torch.inference_mode():
            _, attn_weights = self.forward(x, return_attention=True)
        if was_training:
            self.train()
        return attn_weights

    def count_parameters(self) -> Dict[str, int]:
        """Compute total, trainable, and non-trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable = total - trainable
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "non_trainable_parameters": non_trainable,
        }


def build_vit(
    spectral_depth: int = 15,
    spatial_size: int = 7,
    num_classes: int = 16,
    in_channels: int = 1,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    mlp_dim: int = 128,
    dropout_rate: float = 0.1,
    pooling: str = "mean",
) -> HyperspectralViT:
    """Convenience factory function to instantiate HyperspectralViT."""
    config = HyperspectralViTConfig(
        in_channels=in_channels,
        spectral_depth=spectral_depth,
        spatial_size=spatial_size,
        num_classes=num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        dropout_rate=dropout_rate,
        pooling=pooling,
    )
    return HyperspectralViT(config)
