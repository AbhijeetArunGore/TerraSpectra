"""Automated unit tests for baseline 3D-CNN training, validation, and checkpointing."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from terraspectra.models.cnn3d import Hyperspectral3DCNN, Hyperspectral3DCNNConfig
from terraspectra.models.train import (
    EpochMetrics,
    TrainingConfig,
    TrainingHistory,
    load_checkpoint,
    save_checkpoint,
    select_device,
    set_seed,
    train_model,
    train_one_epoch,
    validate_one_epoch,
)


@pytest.fixture
def synthetic_batch_loaders():
    """Create small synthetic DataLoaders for deterministic training tests."""
    batch_size = 4
    channels, depth, height, width = 1, 15, 5, 5
    num_classes = 3
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


def test_select_device_and_set_seed():
    device_cpu = select_device("cpu")
    assert device_cpu.type == "cpu"

    device_auto = select_device("auto")
    assert device_auto.type in ("cpu", "cuda")

    # Seed reproducibility check
    set_seed(123)
    t1 = torch.randn(5)
    set_seed(123)
    t2 = torch.randn(5)
    assert torch.allclose(t1, t2)


def test_train_one_epoch_updates_parameters(synthetic_batch_loaders):
    train_loader, _, depth, height, num_classes = synthetic_batch_loaders
    device = torch.device("cpu")

    model = Hyperspectral3DCNN(
        Hyperspectral3DCNNConfig(
            spectral_depth=depth,
            spatial_size=height,
            num_classes=num_classes,
            conv_channels=(8, 16),
            kernel_spectral=(5, 3),
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    # Capture initial weights
    initial_weights = [p.clone() for p in model.parameters() if p.requires_grad]

    loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

    assert isinstance(loss, float)
    assert isinstance(acc, float)
    assert loss > 0.0
    assert 0.0 <= acc <= 1.0

    # Assert at least some weights were updated by backpropagation
    updated_weights = [p for p in model.parameters() if p.requires_grad]
    has_changed = any(not torch.allclose(w0, w1) for w0, w1 in zip(initial_weights, updated_weights))
    assert has_changed, "Model parameters were not updated during train_one_epoch"


def test_validate_one_epoch_does_not_update_parameters(synthetic_batch_loaders):
    _, val_loader, depth, height, num_classes = synthetic_batch_loaders
    device = torch.device("cpu")

    model = Hyperspectral3DCNN(
        Hyperspectral3DCNNConfig(
            spectral_depth=depth,
            spatial_size=height,
            num_classes=num_classes,
            conv_channels=(8, 16),
            kernel_spectral=(5, 3),
        )
    )
    criterion = nn.CrossEntropyLoss()

    initial_weights = [p.clone() for p in model.parameters()]

    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

    assert isinstance(val_loss, float)
    assert isinstance(val_acc, float)
    assert val_loss > 0.0
    assert 0.0 <= val_acc <= 1.0

    # Ensure parameters are strictly identical
    current_weights = [p for p in model.parameters()]
    for w0, w1 in zip(initial_weights, current_weights):
        assert torch.allclose(w0, w1), "Validation unexpectedly modified model parameters"


def test_metric_aggregation_handles_uneven_batches():
    """Verify that loss and accuracy are sample-weighted across batches."""
    device = torch.device("cpu")

    # Construct two batches: batch 1 has 10 samples (perfect accuracy), batch 2 has 2 samples (0 accuracy)
    # Total samples = 12. Correct = 10. Expected accuracy = 10 / 12 = 0.8333...
    class DummyModel(nn.Module):
        def forward(self, x):
            return x

    model = DummyModel()
    criterion = nn.CrossEntropyLoss()

    x1 = torch.tensor([[10.0, 0.0]] * 10)  # Predicts class 0
    y1 = torch.tensor([0] * 10)

    x2 = torch.tensor([[0.0, 10.0]] * 2)   # Predicts class 1
    y2 = torch.tensor([0] * 2)             # True class is 0 (both wrong)

    class CustomLoader:
        def __iter__(self):
            return iter([(x1, y1), (x2, y2)])

    loader = CustomLoader()
    avg_loss, acc = validate_one_epoch(model, loader, criterion, device)

    assert np.isclose(acc, 10.0 / 12.0, atol=1e-4)


def test_checkpoint_save_and_load(tmp_path: Path):
    model = Hyperspectral3DCNN(
        Hyperspectral3DCNNConfig(
            spectral_depth=15,
            spatial_size=5,
            num_classes=4,
            conv_channels=(8, 16),
            kernel_spectral=(5, 3),
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint_file = tmp_path / "test_checkpoint.pt"

    saved_path = save_checkpoint(
        checkpoint_path=checkpoint_file,
        model=model,
        optimizer=optimizer,
        epoch=3,
        val_loss=0.456,
        val_acc=0.885,
        config=model.config,
    )
    assert saved_path.is_file()

    # Create fresh uninitialized model
    new_model = Hyperspectral3DCNN(
        Hyperspectral3DCNNConfig(
            spectral_depth=15,
            spatial_size=5,
            num_classes=4,
            conv_channels=(8, 16),
            kernel_spectral=(5, 3),
        )
    )
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-3)

    payload = load_checkpoint(checkpoint_file, model=new_model, optimizer=new_opt, device="cpu")

    assert payload["epoch"] == 3
    assert np.isclose(payload["val_loss"], 0.456)
    assert np.isclose(payload["val_acc"], 0.885)

    # Verify weights match exactly
    for p0, p1 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p0, p1)

    # Verify new model forward pass works
    sample_x = torch.randn(2, 1, 15, 5, 5)
    out = new_model(sample_x)
    assert out.shape == (2, 4)


def test_train_model_synthetic_smoke_test(tmp_path: Path, synthetic_batch_loaders):
    train_loader, val_loader, depth, height, num_classes = synthetic_batch_loaders

    config = TrainingConfig(
        epochs=3,
        lr=1e-2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        checkpoint_filename="test_best.pt",
        device="cpu",
        seed=42,
    )

    model = Hyperspectral3DCNN(
        Hyperspectral3DCNNConfig(
            spectral_depth=depth,
            spatial_size=height,
            num_classes=num_classes,
            conv_channels=(8, 16),
            kernel_spectral=(5, 3),
        )
    )

    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    assert isinstance(history, TrainingHistory)
    assert len(history.epochs) == 3
    assert history.best_epoch in (1, 2, 3)
    assert history.checkpoint_path is not None
    assert Path(history.checkpoint_path).is_file()

    # Forward pass on trained model
    x_test = torch.randn(3, 1, depth, height, height)
    with torch.no_grad():
        out = trained_model(x_test)
    assert out.shape == (3, num_classes)


def test_training_config_validation():
    with pytest.raises(ValueError, match="epochs must be >= 1"):
        TrainingConfig(epochs=0)

    with pytest.raises(ValueError, match="lr must be positive"):
        TrainingConfig(lr=-0.01)

    with pytest.raises(ValueError, match="weight_decay cannot be negative"):
        TrainingConfig(weight_decay=-1e-4)

    with pytest.raises(ValueError, match="monitor_metric must be"):
        TrainingConfig(monitor_metric="invalid_metric")


def test_load_checkpoint_missing_file_raises():
    model = nn.Linear(10, 2)
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint("non_existent_checkpoint_file.pt", model)
