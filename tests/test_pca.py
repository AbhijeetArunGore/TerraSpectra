"""Unit tests for Day 4 PCA dimensionality reduction."""

import numpy as np
import pytest

from terraspectra.processing.pca import HyperspectralPCA, fit_exploratory_pca, fit_train_pca


def test_pca_cube_reshape_shapes():
    cube = np.random.randn(20, 15, 30).astype(np.float32)
    samples = HyperspectralPCA.reshape_cube_to_samples(cube)
    assert samples.shape == (300, 30)

    reconstructed_cube = HyperspectralPCA.reshape_samples_to_cube(samples, (20, 15))
    assert reconstructed_cube.shape == (20, 15, 30)
    np.testing.assert_allclose(cube, reconstructed_cube)


def test_exploratory_pca_reduces_bands():
    # Synthetic cube with 50 bands, rank correlated
    rng = np.random.default_rng(42)
    base = rng.normal(size=(20, 10, 5))
    # Project to 50 bands with small noise
    proj = rng.normal(size=(5, 50))
    cube = np.dot(base, proj).astype(np.float32)

    model, results = fit_exploratory_pca(cube, n_components=10)

    assert results.reduced_cube.shape == (20, 10, 10)
    assert results.reduced_samples.shape == (200, 10)
    assert len(results.explained_variance_ratio) == 10
    assert len(results.cumulative_variance_ratio) == 10
    # First component should capture more variance than 10th
    assert results.explained_variance_ratio[0] >= results.explained_variance_ratio[-1]
    assert 0.0 < results.total_explained_variance <= 1.0


def test_train_only_pca_prevents_leakage():
    rng = np.random.default_rng(42)
    train_samples = rng.normal(size=(100, 40)).astype(np.float32)
    test_samples = rng.normal(size=(30, 40)).astype(np.float32)

    pca = fit_train_pca(train_samples, n_components=8)
    assert pca.is_fitted is True

    train_reduced = pca.transform_samples(train_samples)
    test_reduced = pca.transform_samples(test_samples)

    assert train_reduced.shape == (100, 8)
    assert test_reduced.shape == (30, 8)


def test_pca_validation_errors():
    with pytest.raises(ValueError):
        HyperspectralPCA(n_components=0)

    pca = HyperspectralPCA(n_components=50)
    cube = np.random.randn(10, 10, 20).astype(np.float32)
    with pytest.raises(ValueError):
        # n_components 50 > 20 bands
        pca.fit_exploratory(cube)
