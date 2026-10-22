"""Unit tests for HyperspectralViT, PatchTokenizer, and self-attention operations."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from terraspectra.models.vit import (
    HyperspectralViT,
    HyperspectralViTConfig,
    PatchTokenizer,
    TransformerEncoderBlock,
    build_vit,
)


@pytest.fixture
def default_config() -> HyperspectralViTConfig:
    return HyperspectralViTConfig(
        in_channels=1,
        spectral_depth=15,
        spatial_size=7,
        num_classes=16,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_dim=128,
        dropout_rate=0.1,
        pooling="mean",
    )


def test_vit_initialization_defaults(default_config: HyperspectralViTConfig) -> None:
    """Verify model instantiates with expected submodules and parameter count."""
    model = HyperspectralViT(default_config)

    assert isinstance(model.tokenizer, PatchTokenizer)
    assert len(model.blocks) == 2
    assert isinstance(model.blocks[0], TransformerEncoderBlock)
    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.out_features == 16
    assert model.cls_token is None

    counts = model.count_parameters()
    assert counts["total_parameters"] > 0
    assert counts["trainable_parameters"] == counts["total_parameters"]
    assert counts["non_trainable_parameters"] == 0


def test_vit_forward_pass_shapes_and_values(default_config: HyperspectralViTConfig) -> None:
    """Verify forward pass produces correctly shaped finite logits."""
    model = HyperspectralViT(default_config)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 1, 15, 7, 7)

    with torch.inference_mode():
        logits = model(x)

    assert logits.shape == (batch_size, 16)
    assert torch.isfinite(logits).all(), "Logits must contain only finite real numbers"


def test_vit_raw_spectral_depth_204() -> None:
    """Verify ViT functions with raw 204 AVIRIS spectral channels."""
    config = HyperspectralViTConfig(
        in_channels=1,
        spectral_depth=204,
        spatial_size=7,
        num_classes=16,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
    )
    model = HyperspectralViT(config)
    x = torch.randn(2, 1, 204, 7, 7)

    with torch.inference_mode():
        logits = model(x)

    assert logits.shape == (2, 16)
    assert torch.isfinite(logits).all()


def test_patch_tokenizer_dimensions() -> None:
    """Verify PatchTokenizer flattens spatial patch to H*W sequence with E features."""
    tokenizer = PatchTokenizer(in_channels=1, spectral_depth=15, embed_dim=64)
    x = torch.randn(3, 1, 15, 7, 7)

    tokens = tokenizer(x)
    # Expected: (B, H*W, embed_dim) -> (3, 49, 64)
    assert tokens.shape == (3, 49, 64)
    assert torch.isfinite(tokens).all()


def test_vit_attention_weights_contract(default_config: HyperspectralViTConfig) -> None:
    """Verify attention weights inspection contract and normalization."""
    model = HyperspectralViT(default_config)
    model.eval()

    x = torch.randn(2, 1, 15, 7, 7)

    # 1. Normal forward pass should return only a Tensor, preserving standard contract
    logits = model(x)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 16)

    # 2. Forward with return_attention=True returns (logits, attention_list)
    logits2, attn_weights = model(x, return_attention=True)
    assert torch.allclose(logits, logits2, atol=1e-5)
    assert len(attn_weights) == default_config.num_layers

    for w in attn_weights:
        # Expected shape: (B, num_heads, seq_len, seq_len) -> (2, 4, 49, 49)
        assert w.shape == (2, 4, 49, 49)
        # Softmax rows must sum to 1.0
        row_sums = w.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    # 3. Test get_attention_weights helper
    attn_weights_helper = model.get_attention_weights(x)
    assert len(attn_weights_helper) == default_config.num_layers
    assert attn_weights_helper[0].shape == (2, 4, 49, 49)


def test_vit_pooling_modes() -> None:
    """Verify both mean pooling and CLS token pooling produce valid output logits."""
    # Mean pooling
    cfg_mean = HyperspectralViTConfig(pooling="mean")
    model_mean = HyperspectralViT(cfg_mean)
    assert model_mean.cls_token is None

    # CLS pooling
    cfg_cls = HyperspectralViTConfig(pooling="cls")
    model_cls = HyperspectralViT(cfg_cls)
    assert model_cls.cls_token is not None
    assert model_cls.pos_embedding.shape[1] == (7 * 7 + 1)  # 50 tokens

    x = torch.randn(2, 1, 15, 7, 7)
    with torch.inference_mode():
        out_mean = model_mean(x)
        out_cls = model_cls(x)

    assert out_mean.shape == (2, 16)
    assert out_cls.shape == (2, 16)
    assert torch.isfinite(out_mean).all()
    assert torch.isfinite(out_cls).all()


def test_vit_rejects_invalid_inputs(default_config: HyperspectralViTConfig) -> None:
    """Verify clear error messages on malformed input tensors."""
    model = HyperspectralViT(default_config)

    # Non-tensor
    with pytest.raises(TypeError, match="Expected input to be torch.Tensor"):
        model("not a tensor")  # type: ignore

    # Incorrect rank (4D instead of 5D)
    with pytest.raises(ValueError, match="expects a 5D tensor"):
        model(torch.randn(2, 15, 7, 7))

    # Incompatible in_channels
    with pytest.raises(ValueError, match="Input channel dimension C=3"):
        model(torch.randn(2, 3, 15, 7, 7))

    # Incompatible spectral_depth
    with pytest.raises(ValueError, match="Input spectral depth D=30"):
        model(torch.randn(2, 1, 30, 7, 7))

    # Incompatible spatial dimensions
    with pytest.raises(ValueError, match="Input spatial dimensions"):
        model(torch.randn(2, 1, 15, 5, 5))


def test_vit_rejects_invalid_configurations() -> None:
    """Verify architectural hyperparameter validation."""
    # embed_dim not divisible by num_heads
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        HyperspectralViTConfig(embed_dim=65, num_heads=4)

    # Even spatial size
    with pytest.raises(ValueError, match="must be a positive odd integer"):
        HyperspectralViTConfig(spatial_size=6)

    # num_classes < 2
    with pytest.raises(ValueError, match="num_classes must be >= 2"):
        HyperspectralViTConfig(num_classes=1)

    # Invalid pooling
    with pytest.raises(ValueError, match="pooling must be 'mean' or 'cls'"):
        HyperspectralViTConfig(pooling="max")

    # Invalid dropout rate
    with pytest.raises(ValueError, match="dropout_rate must be in"):
        HyperspectralViTConfig(dropout_rate=1.5)


def test_vit_gradient_flow(default_config: HyperspectralViTConfig) -> None:
    """Verify gradients propagate through all layers during backpropagation."""
    model = HyperspectralViT(default_config)
    model.train()

    x = torch.randn(2, 1, 15, 7, 7)
    targets = torch.tensor([3, 7], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    logits = model(x)
    loss = criterion(logits, targets)
    loss.backward()

    # Check that projection, encoder, and head received gradients
    assert model.tokenizer.projection.weight.grad is not None
    assert model.pos_embedding.grad is not None
    assert model.blocks[0].attn.in_proj_weight.grad is not None
    assert model.classifier.weight.grad is not None


def test_vit_cpu_execution_explicitly(default_config: HyperspectralViTConfig) -> None:
    """Verify CPU execution is supported without GPU assumptions."""
    device = torch.device("cpu")
    model = HyperspectralViT(default_config).to(device)
    x = torch.randn(2, 1, 15, 7, 7, device=device)

    with torch.inference_mode():
        logits = model(x)

    assert logits.device == device
    assert logits.shape == (2, 16)


def test_build_vit_factory() -> None:
    """Verify build_vit convenience function creates configured model."""
    model = build_vit(
        spectral_depth=15,
        spatial_size=7,
        num_classes=16,
        embed_dim=32,
        num_heads=2,
        num_layers=1,
    )
    assert model.config.embed_dim == 32
    assert model.config.num_heads == 2
    assert len(model.blocks) == 1

    x = torch.randn(1, 1, 15, 7, 7)
    with torch.inference_mode():
        logits = model(x)
    assert logits.shape == (1, 16)
