"""Automated tests for PyTorch 3D-CNN baseline model architecture."""

import pytest
import torch
import torch.nn.functional as F

from terraspectra.models.cnn3d import (
    Hyperspectral3DCNN,
    Hyperspectral3DCNNConfig,
    build_3d_cnn,
)


def test_model_initialization_defaults():
    model = build_3d_cnn(spectral_depth=204, spatial_size=7, num_classes=17)
    assert isinstance(model, Hyperspectral3DCNN)
    assert model.config.spectral_depth == 204
    assert model.config.spatial_size == 7
    assert model.config.num_classes == 17

    params = model.count_parameters()
    assert params["total_parameters"] > 0
    assert params["trainable_parameters"] == params["total_parameters"]
    assert params["non_trainable_parameters"] == 0


def test_forward_pass_raw_spectral_depth():
    # Batch of 4 samples, 1 channel, 204 spectral bands, 7x7 spatial patch
    batch_size = 4
    spectral_depth = 204
    spatial_size = 7
    num_classes = 17

    model = build_3d_cnn(spectral_depth=spectral_depth, spatial_size=spatial_size, num_classes=num_classes)
    model.eval()

    x = torch.randn(batch_size, 1, spectral_depth, spatial_size, spatial_size, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch_size, num_classes)
    assert torch.all(torch.isfinite(logits)), "Logits must be finite"


def test_forward_pass_pca_reduced_depth():
    # Model should adapt flexibly to PCA-reduced spectral depth (e.g. 15 bands) and 5x5 patch
    batch_size = 2
    spectral_depth = 15
    spatial_size = 5
    num_classes = 16

    model = build_3d_cnn(
        spectral_depth=spectral_depth,
        spatial_size=spatial_size,
        num_classes=num_classes,
        kernel_spectral=(5, 3, 3),  # Kernel sizes adapted to smaller depth
    )
    x = torch.randn(batch_size, 1, spectral_depth, spatial_size, spatial_size, dtype=torch.float32)
    logits = model(x)

    assert logits.shape == (batch_size, num_classes)
    assert torch.all(torch.isfinite(logits))


def test_backward_pass_gradients():
    # Test gradient computation with CrossEntropyLoss
    torch.manual_seed(42)
    batch_size = 3
    num_classes = 17
    model = build_3d_cnn(spectral_depth=30, spatial_size=7, num_classes=num_classes)
    model.train()

    x = torch.randn(batch_size, 1, 30, 7, 7, dtype=torch.float32)
    targets = torch.tensor([0, 5, 16], dtype=torch.long)

    logits = model(x)
    loss = F.cross_entropy(logits, targets)

    assert torch.isfinite(loss)
    loss.backward()

    # Verify that gradients are computed and non-zero
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} was not computed"
            assert torch.all(torch.isfinite(param.grad)), f"Gradient for {name} contains NaN or Inf"


def test_rejects_invalid_tensor_dimensionality():
    model = build_3d_cnn()
    # 4D tensor (Batch, Channels, Height, Width) instead of 5D
    invalid_4d = torch.randn(2, 204, 7, 7)
    with pytest.raises(ValueError) as exc:
        model(invalid_4d)
    assert "expects a 5-dimensional tensor" in str(exc.value)

    # Non-tensor input
    with pytest.raises(TypeError):
        model("not_a_tensor")  # type: ignore


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        # num_classes < 2
        Hyperspectral3DCNNConfig(num_classes=1)

    with pytest.raises(ValueError):
        # in_channels < 1
        Hyperspectral3DCNNConfig(in_channels=0)

    with pytest.raises(ValueError):
        # Mismatched conv_channels and kernel_spectral length
        Hyperspectral3DCNN(
            Hyperspectral3DCNNConfig(
                conv_channels=(8, 16),
                kernel_spectral=(7, 5, 3),
            )
        )

    with pytest.raises(ValueError):
        # Invalid dropout rate
        Hyperspectral3DCNNConfig(dropout_rate=1.5)


def test_cpu_execution_explicitly():
    cpu_device = torch.device("cpu")
    model = build_3d_cnn(spectral_depth=20, spatial_size=5, num_classes=5).to(cpu_device)
    x = torch.randn(1, 1, 20, 5, 5, device=cpu_device)
    out = model(x)
    assert out.device == cpu_device
    assert out.shape == (1, 5)
