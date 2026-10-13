"""Tests for Salinas hyperspectral dataset loading, validation, and metadata."""

import numpy as np
import pytest
from pathlib import Path
import scipy.io

from terraspectra.data.loader import (
    SalinasDataset,
    SalinasLoader,
    create_synthetic_salinas_fixture,
    load_salinas_data,
)


@pytest.fixture
def synthetic_dataset_paths(tmp_path: Path):
    """Create a temporary synthetic dataset for deterministic test runs."""
    cube_path = tmp_path / "Salinas_corrected.mat"
    gt_path = tmp_path / "Salinas_gt.mat"
    create_synthetic_salinas_fixture(cube_path, gt_path, shape=(30, 20, 204), n_classes=16)
    return tmp_path, cube_path, gt_path


def test_synthetic_fixture_generation(synthetic_dataset_paths):
    tmp_path, cube_path, gt_path = synthetic_dataset_paths
    assert cube_path.exists()
    assert gt_path.exists()

    cube_mat = scipy.io.loadmat(str(cube_path))
    gt_mat = scipy.io.loadmat(str(gt_path))
    assert "salinas_corrected" in cube_mat
    assert "salinas_gt" in gt_mat
    assert cube_mat["salinas_corrected"].shape == (30, 20, 204)
    assert gt_mat["salinas_gt"].shape == (30, 20)


def test_loader_validates_dimensions_and_compatibility(synthetic_dataset_paths):
    tmp_path, _, _ = synthetic_dataset_paths
    loader = SalinasLoader(raw_data_dir=tmp_path)
    dataset = loader.load()

    assert isinstance(dataset, SalinasDataset)
    assert dataset.rows == 30
    assert dataset.cols == 20
    assert dataset.bands == 204
    assert dataset.cube.shape == (30, 20, 204)
    assert dataset.gt.shape == (30, 20)
    assert dataset.metadata.spectral_band_count == 204
    assert dataset.metadata.memory_usage_mb > 0
    assert len(dataset.metadata.unique_classes) > 0


def test_loader_fails_on_missing_files(tmp_path: Path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    loader = SalinasLoader(raw_data_dir=empty_dir)

    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load(allow_synthetic_fallback=False)
    assert "Required Salinas dataset files are missing" in str(exc_info.value)


def test_loader_rejects_incompatible_dimensions(tmp_path: Path):
    cube = np.zeros((30, 20, 50), dtype=np.float32)
    mismatched_gt = np.zeros((40, 20), dtype=np.int64)

    loader = SalinasLoader(raw_data_dir=tmp_path)
    with pytest.raises(ValueError) as exc_info:
        loader.validate(cube, mismatched_gt, "fake_cube.mat", "fake_gt.mat")
    assert "Spatial dimension mismatch" in str(exc_info.value)


def test_loader_rejects_invalid_dimensions(tmp_path: Path):
    cube_2d = np.zeros((30, 20), dtype=np.float32)
    gt = np.zeros((30, 20), dtype=np.int64)

    loader = SalinasLoader(raw_data_dir=tmp_path)
    with pytest.raises(ValueError) as exc_info:
        loader.validate(cube_2d, gt, "fake_cube.mat", "fake_gt.mat")
    assert "Hyperspectral cube must be 3-dimensional" in str(exc_info.value)
