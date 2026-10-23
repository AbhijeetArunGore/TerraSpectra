"""Unit tests for the CNN-ViT Hybrid Fusion Network architecture."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from terraspectra.models.cnn3d import Hyperspectral3DCNN, Hyperspectral3DCNNConfig
from terraspectra.models.hybrid import (
    HyperspectralHybridCNNViT,
    HyperspectralHybridConfig,
    build_hybrid,
)
from terraspectra.models.vit import HyperspectralViT, HyperspectralViTConfig


@pytest.fixture
def default_config() -> HyperspectralHybridConfig:
    return HyperspectralHybridConfig(
        in_channels=1,
        spectral_depth=15,
        spatial_size=7,
        num_classes=16,
        cnn_conv_channels=(8, 16, 32),
        cnn_kernel_spectral=(7, 5, 3),
        vit_embed_dim=64,
        vit_num_heads=4,
        vit_num_layers=2,
        vit_mlp_dim=128,
        vit_pooling="mean",
        fusion_dim=64,
        dropout_rate=0.2,
    )


def test_hybrid_construction_and_defaults(default_config: HyperspectralHybridConfig) -> None:
    """Verify hybrid model instantiates with expected submodules and feature dimensions."""
    model = HyperspectralHybridCNNViT(default_config)

    assert isinstance(model.cnn_branch, Hyperspectral3DCNN)
    assert isinstance(model.vit_branch, HyperspectralViT)
    assert isinstance(model.fusion_projection, nn.Sequential)
    assert isinstance(model.classifier, nn.Linear)

    # Check feature dimension compatibility
    assert model.cnn_feature_dim == 32
    assert model.vit_feature_dim == 64
    assert model.raw_fusion_dim == 96
    assert model.classifier.out_features == 16

    counts = model.count_parameters()
    assert counts["total_parameters"] > 0
    assert counts["trainable_parameters"] == counts["total_parameters"]
    assert counts["non_trainable_parameters"] == 0
    assert "cnn_branch_parameters" in counts["component_breakdown"]
    assert "vit_branch_parameters" in counts["component_breakdown"]
    assert "fusion_projection_parameters" in counts["component_breakdown"]


def test_hybrid_forward_pass_batch_size_1(default_config: HyperspectralHybridConfig) -> None:
    """Verify forward pass operates cleanly with batch size 1 (single-sample inference)."""
    model = HyperspectralHybridCNNViT(default_config)
    model.eval()

    x = torch.randn(1, 1, 15, 7, 7)
    with torch.inference_mode():
        logits = model(x)

    assert logits.shape == (1, 16)
    assert torch.isfinite(logits).all(), "Logits must contain only finite real numbers"


def test_hybrid_forward_pass_batch_size_greater_than_1(default_config: HyperspectralHybridConfig) -> None:
    """Verify forward pass produces correctly shaped finite logits with batches > 1."""
    model = HyperspectralHybridCNNViT(default_config)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 1, 15, 7, 7)
    with torch.inference_mode():
        logits = model(x)

    assert logits.shape == (batch_size, 16)
    assert torch.isfinite(logits).all()


def test_hybrid_feature_extraction_and_shapes(default_config: HyperspectralHybridConfig) -> None:
    """Verify intermediate branch and fused feature extraction shapes."""
    model = HyperspectralHybridCNNViT(default_config)
    model.eval()

    batch_size = 3
    x = torch.randn(batch_size, 1, 15, 7, 7)

    # 1. extract_branch_features
    with torch.inference_mode():
        cnn_feat, vit_feat = model.extract_branch_features(x)

    assert cnn_feat.shape == (batch_size, 32)
    assert vit_feat.shape == (batch_size, 64)
    assert torch.isfinite(cnn_feat).all()
    assert torch.isfinite(vit_feat).all()

    # 2. forward with return_features=True
    with torch.inference_mode():
        logits, features = model(x, return_features=True)

    assert logits.shape == (batch_size, 16)
    assert features["cnn_features"].shape == (batch_size, 32)
    assert features["vit_features"].shape == (batch_size, 64)
    assert features["fused_features"].shape == (batch_size, 64)

    # 3. Standard forward returns only logits tensor
    with torch.inference_mode():
        logits_standard = model(x)
    assert isinstance(logits_standard, torch.Tensor)
    assert torch.equal(logits, logits_standard)


def test_hybrid_backpropagation_gradients(default_config: HyperspectralHybridConfig) -> None:
    """Verify gradients propagate to both CNN, ViT, and fusion projection branches."""
    model = HyperspectralHybridCNNViT(default_config)
    model.train()

    batch_size = 2
    x = torch.randn(batch_size, 1, 15, 7, 7)
    targets = torch.tensor([2, 9], dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    logits = model(x)
    loss = criterion(logits, targets)
    loss.backward()

    # CNN branch receives gradients
    assert model.cnn_branch.features[0].weight.grad is not None
    # ViT branch receives gradients
    assert model.vit_branch.tokenizer.projection.weight.grad is not None
    assert model.vit_branch.pos_embedding.grad is not None
    # Fusion projection receives gradients
    assert model.fusion_projection[0].weight.grad is not None
    # Classifier receives gradients
    assert model.classifier.weight.grad is not None


def test_hybrid_rejects_invalid_inputs(default_config: HyperspectralHybridConfig) -> None:
    """Verify clear error handling for non-tensor, invalid rank, and shape mismatches."""
    model = HyperspectralHybridCNNViT(default_config)

    # Non-tensor
    with pytest.raises(TypeError, match="Expected input to be torch.Tensor"):
        model("not a tensor")  # type: ignore

    # Rank mismatch (4D instead of 5D)
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


def test_hybrid_rejects_invalid_configurations() -> None:
    """Verify hyperparameter validation during config initialization."""
    # Even spatial size
    with pytest.raises(ValueError, match="spatial_size must be a positive odd integer"):
        HyperspectralHybridConfig(spatial_size=6)

    # Spatial size < 3
    with pytest.raises(ValueError, match="spatial_size must be a positive odd integer"):
        HyperspectralHybridConfig(spatial_size=1)

    # num_classes < 2
    with pytest.raises(ValueError, match="num_classes must be >= 2"):
        HyperspectralHybridConfig(num_classes=1)

    # Mismatched CNN conv channels and kernels
    with pytest.raises(ValueError, match="cnn_conv_channels length"):
        HyperspectralHybridConfig(
            cnn_conv_channels=(8, 16),
            cnn_kernel_spectral=(7, 5, 3),
        )

    # vit_embed_dim not divisible by vit_num_heads
    with pytest.raises(ValueError, match="must be divisible by vit_num_heads"):
        HyperspectralHybridConfig(vit_embed_dim=65, vit_num_heads=4)

    # Invalid fusion dimension
    with pytest.raises(ValueError, match="fusion_dim must be >= 1"):
        HyperspectralHybridConfig(fusion_dim=0)

    # Invalid dropout rate
    with pytest.raises(ValueError, match="dropout_rate must be in"):
        HyperspectralHybridConfig(dropout_rate=1.5)


def test_standalone_cnn_and_vit_compatibility() -> None:
    """Verify backward compatibility: standalone CNN and ViT extract_features and feature_dim."""
    # Standalone CNN
    cnn = Hyperspectral3DCNN(Hyperspectral3DCNNConfig(spectral_depth=15, spatial_size=7, num_classes=16))
    assert cnn.feature_dim == 32
    x = torch.randn(2, 1, 15, 7, 7)
    with torch.inference_mode():
        cnn_feat = cnn.extract_features(x)
        cnn_logits = cnn(x)
    assert cnn_feat.shape == (2, 32)
    assert cnn_logits.shape == (2, 16)

    # Standalone ViT
    vit = HyperspectralViT(HyperspectralViTConfig(spectral_depth=15, spatial_size=7, num_classes=16, embed_dim=64))
    assert vit.feature_dim == 64
    with torch.inference_mode():
        vit_feat = vit.extract_features(x)
        vit_logits = vit(x)
    assert vit_feat.shape == (2, 64)
    assert vit_logits.shape == (2, 16)


def test_hybrid_with_custom_branches() -> None:
    """Verify hybrid model can be instantiated with pre-existing branch instances."""
    cnn = Hyperspectral3DCNN(Hyperspectral3DCNNConfig(spectral_depth=15, spatial_size=7, num_classes=16))
    vit = HyperspectralViT(HyperspectralViTConfig(spectral_depth=15, spatial_size=7, num_classes=16, embed_dim=64))

    hybrid = HyperspectralHybridCNNViT(cnn_branch=cnn, vit_branch=vit)
    assert hybrid.cnn_branch is cnn
    assert hybrid.vit_branch is vit

    x = torch.randn(2, 1, 15, 7, 7)
    with torch.inference_mode():
        logits = hybrid(x)
    assert logits.shape == (2, 16)


def test_build_hybrid_factory() -> None:
    """Verify build_hybrid convenience factory function instantiates model properly."""
    model = build_hybrid(
        spectral_depth=15,
        spatial_size=7,
        num_classes=16,
        fusion_dim=32,
    )
    assert model.config.fusion_dim == 32
    x = torch.randn(1, 1, 15, 7, 7)
    with torch.inference_mode():
        logits = model(x)
    assert logits.shape == (1, 16)


def test_hybrid_cpu_execution_explicitly(default_config: HyperspectralHybridConfig) -> None:
    """Verify CPU execution is supported without GPU assumptions."""
    device = torch.device("cpu")
    model = HyperspectralHybridCNNViT(default_config).to(device)
    x = torch.randn(2, 1, 15, 7, 7, device=device)

    with torch.inference_mode():
        logits = model(x)

    assert logits.device == device
    assert logits.shape == (2, 16)
