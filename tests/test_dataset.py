"""Automated tests for PyTorch HyperspectralPatchDataset and DataLoader pipeline."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from terraspectra.data.dataset import (
    DataLoaderMetadata,
    HyperspectralPatchDataset,
    create_dataloaders,
    split_hyperspectral_samples,
)
from terraspectra.data.loader import DatasetMetadata, SalinasDataset
from terraspectra.data.patches import extract_spatial_patch
from terraspectra.models.cnn3d import Hyperspectral3DCNN, Hyperspectral3DCNNConfig


@pytest.fixture
def synthetic_scene():
    """Create a small deterministic hyperspectral scene for fast unit testing."""
    rng = np.random.default_rng(42)
    height, width, bands = 24, 20, 15
    cube = rng.standard_normal((height, width, bands)).astype(np.float32)

    # Ground truth with background (0) and 3 active classes (1, 2, 3)
    gt = np.zeros((height, width), dtype=np.uint8)
    gt[2:10, 2:10] = 1
    gt[10:18, 2:10] = 2
    gt[2:18, 10:18] = 3

    return cube, gt


def test_dataset_initialization_and_sample_retrieval(synthetic_scene):
    cube, gt = synthetic_scene
    patch_size = 5
    dataset = HyperspectralPatchDataset(cube, gt, patch_size=patch_size, exclude_unlabeled=True)

    n_labeled = int(np.sum(gt != 0))
    assert len(dataset) == n_labeled
    assert dataset.num_classes == 3

    # Retrieve first sample
    patch, target = dataset[0]

    assert isinstance(patch, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert patch.dtype == torch.float32
    assert target.dtype == torch.int64

    # Expected shape: (1, spectral_depth, patch_h, patch_w)
    assert patch.shape == (1, 15, patch_size, patch_size)
    assert target.ndim == 0
    assert 0 <= target.item() < dataset.num_classes


def test_dataset_center_pixel_label_alignment(synthetic_scene):
    cube, gt = synthetic_scene
    dataset = HyperspectralPatchDataset(cube, gt, patch_size=5, exclude_unlabeled=True)

    for i in range(min(len(dataset), 25)):
        meta = dataset.get_sample_metadata(i)
        r, c = meta["center_row"], meta["center_col"]
        expected_orig_label = int(gt[r, c])

        assert meta["original_label"] == expected_orig_label
        assert meta["mapped_target"] == dataset.label_to_idx[expected_orig_label]

        _, target_tensor = dataset[i]
        assert target_tensor.item() == dataset.label_to_idx[expected_orig_label]


def test_dataset_exclude_and_include_unlabeled(synthetic_scene):
    cube, gt = synthetic_scene

    # With unlabeled excluded
    ds_no_bg = HyperspectralPatchDataset(cube, gt, patch_size=5, exclude_unlabeled=True)
    assert len(ds_no_bg) == int(np.sum(gt > 0))
    assert 0 not in ds_no_bg.label_to_idx

    # With unlabeled included
    ds_with_bg = HyperspectralPatchDataset(cube, gt, patch_size=5, exclude_unlabeled=False)
    assert len(ds_with_bg) == gt.shape[0] * gt.shape[1]
    assert 0 in ds_with_bg.label_to_idx
    assert ds_with_bg.label_to_idx[0] == 0


def test_dataset_custom_label_mapping(synthetic_scene):
    cube, gt = synthetic_scene
    # Original classes in synthetic_scene are 1, 2, 3
    custom_mapping = {1: 10, 2: 20, 3: 30}
    dataset = HyperspectralPatchDataset(
        cube, gt, patch_size=5, exclude_unlabeled=True, label_to_idx=custom_mapping
    )

    assert dataset.label_to_idx == custom_mapping
    assert dataset.idx_to_label == {10: 1, 20: 2, 30: 3}
    _, target = dataset[0]
    assert target.item() in (10, 20, 30)

    # Incomplete mapping must raise ValueError
    with pytest.raises(ValueError, match="missing mappings"):
        HyperspectralPatchDataset(
            cube, gt, patch_size=5, exclude_unlabeled=True, label_to_idx={1: 0, 2: 1}
        )


def test_dataset_slice_matches_patches_utility(synthetic_scene):
    """Verify that dataset pre-padded O(1) slicing yields identical values to extract_spatial_patch."""
    cube, gt = synthetic_scene
    patch_size = 7
    dataset = HyperspectralPatchDataset(cube, gt, patch_size=patch_size, pad_mode="reflect")

    # Test an interior pixel and a border pixel
    test_coords = [(0, 0), (10, 10), (cube.shape[0] - 1, cube.shape[1] - 1)]
    for r, c in test_coords:
        patch_sample = extract_spatial_patch(
            cube, r, c, patch_size=patch_size, gt=gt, pad_mode="reflect", add_batch_dim=False
        )
        # Find index in dataset
        match_idx = np.where((dataset.coords[:, 0] == r) & (dataset.coords[:, 1] == c))[0]
        if len(match_idx) > 0:
            ds_patch, ds_target = dataset[match_idx[0]]
            np.testing.assert_allclose(
                ds_patch.numpy(),
                patch_sample.tensor.numpy(),
                err_msg=f"Mismatch at coordinate ({r}, {c})",
            )
            assert ds_target.item() == dataset.label_to_idx[patch_sample.label]


def test_dataset_invalid_inputs_and_edge_cases(synthetic_scene):
    cube, gt = synthetic_scene

    # Non-3D cube
    with pytest.raises(ValueError, match="cube must have 3 dimensions"):
        HyperspectralPatchDataset(cube[:, :, 0], gt, patch_size=5)

    # Non-2D gt
    with pytest.raises(ValueError, match="gt must have 2 dimensions"):
        HyperspectralPatchDataset(cube, gt[:, :, np.newaxis], patch_size=5)

    # Spatial mismatch
    with pytest.raises(ValueError, match="Spatial dimensions mismatch"):
        HyperspectralPatchDataset(cube[:10, :10, :], gt, patch_size=5)

    # Even patch size
    with pytest.raises(ValueError, match="must be an odd integer"):
        HyperspectralPatchDataset(cube, gt, patch_size=6)

    # Non-positive patch size
    with pytest.raises(ValueError, match="positive integer"):
        HyperspectralPatchDataset(cube, gt, patch_size=0)

    # Empty coordinates
    with pytest.raises(ValueError, match="coords cannot be an empty sequence"):
        HyperspectralPatchDataset(cube, gt, patch_size=5, coords=[])

    # Out-of-bounds coordinates
    with pytest.raises(IndexError, match="Coordinate out of bounds"):
        HyperspectralPatchDataset(cube, gt, patch_size=5, coords=[(999, 999)])

    # All-zero gt with exclude_unlabeled=True
    zero_gt = np.zeros_like(gt)
    with pytest.raises(ValueError, match="No eligible labeled pixels"):
        HyperspectralPatchDataset(cube, zero_gt, patch_size=5, exclude_unlabeled=True)

    # Dataset index out of range
    ds = HyperspectralPatchDataset(cube, gt, patch_size=5)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]
    with pytest.raises(IndexError):
        _ = ds[-1]


def test_split_hyperspectral_samples_reproducibility():
    coords = np.column_stack([np.arange(100), np.arange(100)])
    labels = np.array([1] * 50 + [2] * 30 + [3] * 20)

    train1, val1, test1 = split_hyperspectral_samples(
        coords, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, stratify=True
    )
    train2, val2, test2 = split_hyperspectral_samples(
        coords, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, stratify=True
    )

    np.testing.assert_array_equal(train1, train2)
    np.testing.assert_array_equal(val1, val2)
    np.testing.assert_array_equal(test1, test2)

    # Total sample count conservation
    assert len(train1) + len(val1) + len(test1) == len(coords)

    # No coordinate leakage between splits
    set_train = set(map(tuple, train1))
    set_val = set(map(tuple, val1))
    set_test = set(map(tuple, test1))
    assert len(set_train.intersection(set_val)) == 0
    assert len(set_train.intersection(set_test)) == 0
    assert len(set_val.intersection(set_test)) == 0

    # Different seed yields different split order/indices
    train3, _, _ = split_hyperspectral_samples(
        coords, labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=99, stratify=True
    )
    assert not np.array_equal(train1, train3)


def test_split_hyperspectral_samples_rare_classes():
    """Verify that split handles rare classes (1 or 2 samples) without crashing."""
    coords = np.column_stack([np.arange(20), np.arange(20)])
    # Class 4 has only 1 sample, class 5 has only 2 samples
    labels = np.array([1] * 10 + [2] * 7 + [4] * 1 + [5] * 2)

    train, val, test = split_hyperspectral_samples(
        coords, labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42, stratify=True
    )

    assert len(train) + len(val) + len(test) == 20
    assert len(train) > 0


def test_create_dataloaders_shapes_and_types(synthetic_scene):
    cube, gt = synthetic_scene
    batch_size = 8
    patch_size = 5

    train_loader, val_loader, test_loader, meta = create_dataloaders(
        cube,
        gt,
        patch_size=patch_size,
        batch_size=batch_size,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=42,
    )

    assert isinstance(meta, DataLoaderMetadata)
    assert meta.total_samples == meta.train_samples + meta.val_samples + meta.test_samples
    assert meta.num_classes == 3

    # Inspect first batch of train_loader
    batch_x, batch_y = next(iter(train_loader))

    assert isinstance(batch_x, torch.Tensor)
    assert isinstance(batch_y, torch.Tensor)
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.int64

    # 5D Layout: (Batch, Channels=1, Depth=15, Height=5, Width=5)
    assert batch_x.shape == (batch_size, 1, 15, patch_size, patch_size)
    assert batch_y.shape == (batch_size,)
    assert torch.all((batch_y >= 0) & (batch_y < meta.num_classes))


def test_create_dataloaders_incomplete_batch_handling(synthetic_scene):
    cube, gt = synthetic_scene
    total_labeled = int(np.sum(gt != 0))
    batch_size = 13  # Not a divisor of total_labeled

    # drop_last=False (default): final batch has remainder size
    tl_keep, _, _, meta = create_dataloaders(
        cube, gt, patch_size=5, batch_size=batch_size, drop_last=False
    )
    batches = list(tl_keep)
    total_samples_retrieved = sum(b[0].shape[0] for b in batches)
    assert total_samples_retrieved == meta.train_samples
    assert batches[-1][0].shape[0] <= batch_size

    # drop_last=True: drops incomplete final batch
    tl_drop, _, _, _ = create_dataloaders(
        cube, gt, patch_size=5, batch_size=batch_size, drop_last=True
    )
    batches_drop = list(tl_drop)
    assert all(b[0].shape[0] == batch_size for b in batches_drop)


def test_create_dataloaders_integration_with_cnn3d(synthetic_scene):
    """Pass real DataLoader batches directly through the Day 8 Hyperspectral3DCNN model."""
    cube, gt = synthetic_scene
    batch_size = 6
    patch_size = 5

    train_loader, _, _, meta = create_dataloaders(
        cube, gt, patch_size=patch_size, batch_size=batch_size
    )

    # Initialize Day 8 model
    model = Hyperspectral3DCNN(
        Hyperspectral3DCNNConfig(
            in_channels=1,
            spectral_depth=cube.shape[2],
            spatial_size=patch_size,
            num_classes=meta.num_classes,
            conv_channels=(8, 16),
            kernel_spectral=(5, 3),
            dropout_rate=0.2,
        )
    )
    model.eval()

    batch_x, batch_y = next(iter(train_loader))
    with torch.no_grad():
        logits = model(batch_x)

    assert logits.shape == (batch_size, meta.num_classes)
    assert torch.all(torch.isfinite(logits))

    # Loss computation check
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, batch_y)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_create_dataloaders_accepts_salinas_dataset_container(synthetic_scene):
    cube, gt = synthetic_scene
    metadata = DatasetMetadata(
        image_shape=cube.shape,
        spectral_band_count=cube.shape[2],
        label_shape=gt.shape,
        image_dtype=str(cube.dtype),
        label_dtype=str(gt.dtype),
        memory_usage_mb=1.0,
        is_synthetic=True,
        image_path="",
        gt_path="",
        unique_classes=list(np.unique(gt)),
    )
    container = SalinasDataset(cube=cube, gt=gt, metadata=metadata)

    train_loader, val_loader, test_loader, meta = create_dataloaders(
        container, patch_size=5, batch_size=4
    )
    assert meta.train_samples > 0
    bx, by = next(iter(train_loader))
    assert bx.shape == (4, 1, cube.shape[2], 5, 5)
