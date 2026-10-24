"""Automated unit tests for Day 14: CNN-ViT Hybrid model training and checkpointing."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from terraspectra.models.hybrid import HyperspectralHybridCNNViT, HyperspectralHybridConfig
from terraspectra.models.train import (
    TrainingConfig,
    TrainingHistory,
    build_model_for_training,
    load_checkpoint,
    save_checkpoint,
    select_device,
    set_seed,
    train_model,
    train_one_epoch,
    validate_one_epoch,
)


@pytest.fixture
def synthetic_hybrid_loaders():
    """Create small synthetic DataLoaders tailored for fast hybrid model testing."""
    batch_size = 4
    channels, depth, height, width = 1, 15, 5, 5
    num_classes = 4
    num_samples = 14  # Non-multiple of batch_size to test uneven final batch

    rng = torch.Generator().manual_seed(42)
    x_train = torch.randn(num_samples, channels, depth, height, width, generator=rng)
    y_train = torch.randint(0, num_classes, (num_samples,), generator=rng)

    x_val = torch.randn(8, channels, depth, height, width, generator=rng)
    y_val = torch.randint(0, num_classes, (8,), generator=rng)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, depth, height, num_classes


def test_build_model_for_training_factory():
    """Verify build_model_for_training constructs correct model instances."""
    cnn = build_model_for_training("3d_cnn", spectral_depth=15, spatial_size=5, num_classes=3)
    assert cnn.__class__.__name__ == "Hyperspectral3DCNN"

    hybrid = build_model_for_training("hybrid", spectral_depth=15, spatial_size=5, num_classes=3)
    assert hybrid.__class__.__name__ == "HyperspectralHybridCNNViT"

    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_model_for_training("unknown_type")


def test_hybrid_train_one_epoch_updates_both_branches(synthetic_hybrid_loaders):
    """Verify train_one_epoch updates parameters in both CNN and ViT branches."""
    train_loader, _, depth, height, num_classes = synthetic_hybrid_loaders
    device = torch.device("cpu")

    config = HyperspectralHybridConfig(
        spectral_depth=depth,
        spatial_size=height,
        num_classes=num_classes,
        cnn_conv_channels=(8, 16),
        cnn_kernel_spectral=(5, 3),
        vit_embed_dim=16,
        vit_num_heads=2,
        vit_num_layers=1,
        vit_mlp_dim=32,
        fusion_dim=16,
        dropout_rate=0.2,
    )
    model = HyperspectralHybridCNNViT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Capture initial weights of CNN branch, ViT branch, and fusion head
    cnn_init = [p.clone() for p in model.cnn_branch.parameters() if p.requires_grad]
    vit_init = [p.clone() for p in model.vit_branch.parameters() if p.requires_grad]
    fusion_init = [p.clone() for p in model.fusion_projection.parameters() if p.requires_grad]

    loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert loss > 0.0
    assert 0.0 <= acc <= 1.0

    # Ensure backprop updated both branches and fusion projection
    cnn_changed = any(not torch.allclose(w0, w1) for w0, w1 in zip(cnn_init, model.cnn_branch.parameters()))
    vit_changed = any(not torch.allclose(w0, w1) for w0, w1 in zip(vit_init, model.vit_branch.parameters()))
    fusion_changed = any(not torch.allclose(w0, w1) for w0, w1 in zip(fusion_init, model.fusion_projection.parameters()))

    assert cnn_changed, "CNN branch parameters did not update during training"
    assert vit_changed, "ViT branch parameters did not update during training"
    assert fusion_changed, "Fusion projection parameters did not update during training"


def test_hybrid_validate_one_epoch_preserves_weights(synthetic_hybrid_loaders):
    """Verify validation runs without tracking gradients or mutating model weights."""
    _, val_loader, depth, height, num_classes = synthetic_hybrid_loaders
    device = torch.device("cpu")

    config = HyperspectralHybridConfig(
        spectral_depth=depth,
        spatial_size=height,
        num_classes=num_classes,
        cnn_conv_channels=(8, 16),
        cnn_kernel_spectral=(5, 3),
        vit_embed_dim=16,
        vit_num_heads=2,
        vit_num_layers=1,
        vit_mlp_dim=32,
        fusion_dim=16,
    )
    model = HyperspectralHybridCNNViT(config).to(device)
    criterion = nn.CrossEntropyLoss()

    initial_weights = [p.clone() for p in model.parameters()]

    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

    assert isinstance(val_loss, float)
    assert isinstance(val_acc, float)
    assert val_loss > 0.0
    assert 0.0 <= val_acc <= 1.0

    # Ensure weights are strictly unmodified
    for w0, w1 in zip(initial_weights, model.parameters()):
        assert torch.allclose(w0, w1), "Validation unexpectedly altered hybrid model parameters"


def test_hybrid_checkpoint_saving_and_reloading(tmp_path: Path):
    """Verify saving and restoring hybrid model checkpoints on disk."""
    config = HyperspectralHybridConfig(
        spectral_depth=15,
        spatial_size=5,
        num_classes=4,
        cnn_conv_channels=(8, 16),
        cnn_kernel_spectral=(5, 3),
        vit_embed_dim=16,
        vit_num_heads=2,
        vit_num_layers=1,
        vit_mlp_dim=32,
        fusion_dim=16,
    )
    model = HyperspectralHybridCNNViT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    checkpoint_path = tmp_path / "best_hybrid_cnn_vit.pt"

    saved_path = save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        val_loss=0.314,
        val_acc=0.925,
        config=config,
    )

    assert saved_path.is_file()
    assert saved_path.name == "best_hybrid_cnn_vit.pt"

    # Reload into freshly initialized model instance
    fresh_model = HyperspectralHybridCNNViT(config)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    payload = load_checkpoint(saved_path, model=fresh_model, optimizer=fresh_opt, device="cpu")

    assert payload["epoch"] == 2
    assert payload["model_architecture"] == "HyperspectralHybridCNNViT"
    assert np.isclose(payload["val_loss"], 0.314)
    assert np.isclose(payload["val_acc"], 0.925)
    assert payload["model_config"]["spectral_depth"] == 15
    assert payload["model_config"]["vit_embed_dim"] == 16

    # Verify identical weight values
    for p0, p1 in zip(model.parameters(), fresh_model.parameters()):
        assert torch.allclose(p0, p1)

    # Verify loaded model can execute forward inference cleanly
    sample = torch.randn(2, 1, 15, 5, 5)
    fresh_model.eval()
    with torch.no_grad():
        out = fresh_model(sample)
    assert out.shape == (2, 4)


def test_hybrid_train_model_pipeline(tmp_path: Path, synthetic_hybrid_loaders):
    """End-to-end smoke test for train_model with HyperspectralHybridCNNViT."""
    train_loader, val_loader, depth, height, num_classes = synthetic_hybrid_loaders

    config = HyperspectralHybridConfig(
        spectral_depth=depth,
        spatial_size=height,
        num_classes=num_classes,
        cnn_conv_channels=(8, 16),
        cnn_kernel_spectral=(5, 3),
        vit_embed_dim=16,
        vit_num_heads=2,
        vit_num_layers=1,
        vit_mlp_dim=32,
        fusion_dim=16,
        dropout_rate=0.2,
    )
    model = HyperspectralHybridCNNViT(config)

    train_config = TrainingConfig(
        epochs=2,
        lr=1e-2,
        weight_decay=1e-4,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_filename="best_hybrid_cnn_vit.pt",
        model_type="hybrid",
        seed=42,
    )

    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        model_config=config,
    )

    assert isinstance(history, TrainingHistory)
    assert len(history.epochs) == 2
    assert history.best_epoch in (1, 2)
    assert history.checkpoint_path is not None
    assert Path(history.checkpoint_path).is_file()
    assert Path(history.checkpoint_path).name == "best_hybrid_cnn_vit.pt"

    # Forward pass on trained model produces valid logits
    x_test = torch.randn(3, 1, depth, height, height)
    with torch.no_grad():
        logits = trained_model(x_test)
    assert logits.shape == (3, num_classes)
