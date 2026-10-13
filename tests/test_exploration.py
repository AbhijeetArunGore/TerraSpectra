"""Unit tests for Day 3 exploration, spectral signatures, and false-color composites."""

import numpy as np
import pytest
from pathlib import Path

from terraspectra.data.loader import create_synthetic_salinas_fixture, SalinasLoader
from terraspectra.data.exploration import HyperspectralExplorer, QualityAuditReport


@pytest.fixture
def explorer_fixture(tmp_path: Path):
    cube_path = tmp_path / "Salinas_corrected.mat"
    gt_path = tmp_path / "Salinas_gt.mat"
    create_synthetic_salinas_fixture(cube_path, gt_path, shape=(40, 30, 204), n_classes=16)

    loader = SalinasLoader(raw_data_dir=tmp_path)
    dataset = loader.load(auto_download=False)
    explorer = HyperspectralExplorer(dataset=dataset, output_dir=tmp_path / "outputs")
    return explorer, tmp_path


def test_quality_audit(explorer_fixture):
    explorer, _ = explorer_fixture
    audit = explorer.audit_numerical_quality()

    assert isinstance(audit, QualityAuditReport)
    assert audit.nan_count == 0
    assert audit.inf_count == 0
    assert audit.is_finite is True
    assert audit.total_pixels == 40 * 30
    assert audit.min_value <= audit.max_value


def test_quality_audit_detects_nans():
    cube = np.ones((10, 10, 20), dtype=np.float32)
    cube[2, 3, 5] = np.nan
    gt = np.zeros((10, 10), dtype=np.int64)

    from terraspectra.data.loader import SalinasDataset, DatasetMetadata
    meta = DatasetMetadata(
        image_shape=(10, 10, 20),
        spectral_band_count=20,
        label_shape=(10, 10),
        image_dtype="float32",
        label_dtype="int64",
        memory_usage_mb=0.1,
        is_synthetic=True,
        image_path="",
        gt_path="",
        unique_classes=[0],
    )
    dataset = SalinasDataset(cube=cube, gt=gt, metadata=meta)
    explorer = HyperspectralExplorer(dataset=dataset, output_dir=Path("tmp_test_outputs"))
    audit = explorer.audit_numerical_quality()

    assert audit.nan_count == 1
    assert audit.is_finite is False


def test_false_color_composite(explorer_fixture):
    explorer, tmp_path = explorer_fixture
    save_path = tmp_path / "test_fc.png"
    rgb = explorer.generate_false_color_composite(bands=(45, 25, 15), save_path=save_path)

    assert rgb.shape == (40, 30, 3)
    assert rgb.min() >= 0.0
    assert rgb.max() <= 1.0
    assert save_path.exists()


def test_spectral_signatures(explorer_fixture):
    explorer, tmp_path = explorer_fixture
    save_path = tmp_path / "test_sig.png"
    sigs = explorer.generate_spectral_signatures(classes_to_plot=[1, 2], save_path=save_path)

    assert 1 in sigs
    assert 2 in sigs
    assert len(sigs[1]["mean"]) == 204
    assert len(sigs[1]["std"]) == 204
    assert save_path.exists()


def test_full_exploration_workflow(explorer_fixture):
    explorer, tmp_path = explorer_fixture
    summary = explorer.run_full_exploration()

    assert summary.spatial_dimensions == (40, 30)
    assert summary.spectral_bands == 204
    assert (tmp_path / "outputs" / "reports" / "day3_exploration_report.json").exists()
