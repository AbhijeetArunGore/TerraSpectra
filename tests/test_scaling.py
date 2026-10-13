"""Unit tests for Day 5 feature scaling and normalization."""

import numpy as np
import pytest

from terraspectra.processing.scaling import (
    HyperspectralScaler,
    ScalingMethod,
    fit_train_scaler,
)


def test_standard_scaling_statistics():
    rng = np.random.default_rng(42)
    # 2D samples (100, 10) with varying means and variances
    data = rng.normal(loc=50.0, scale=12.0, size=(100, 10)).astype(np.float32)

    scaler = HyperspectralScaler(method=ScalingMethod.STANDARD)
    scaled = scaler.fit_transform(data)

    assert scaled.shape == (100, 10)
    # Mean ~ 0, Std ~ 1
    np.testing.assert_allclose(np.mean(scaled, axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(scaled, axis=0), 1.0, atol=1e-5)


def test_minmax_scaling_bounds():
    rng = np.random.default_rng(42)
    data = rng.uniform(low=-50.0, high=120.0, size=(50, 20, 5)).astype(np.float32)

    scaler = HyperspectralScaler(method=ScalingMethod.MINMAX, feature_range=(0.0, 1.0))
    scaled = scaler.fit_transform(data)

    assert scaled.shape == (50, 20, 5)
    assert scaled.min() >= -1e-6
    assert scaled.max() <= 1.0 + 1e-6


def test_constant_feature_handling():
    # Feature 2 is constant (zero variance)
    data = np.ones((50, 4), dtype=np.float32) * 5.0
    data[:, 0] = np.linspace(1, 10, 50)
    data[:, 1] = np.linspace(10, 100, 50)

    scaler = HyperspectralScaler(method=ScalingMethod.STANDARD)
    scaled = scaler.fit_transform(data)

    assert not np.isnan(scaled).any()
    assert not np.isinf(scaled).any()
    assert 2 in scaler.constant_feature_indices
    # Constant column should be mapped safely to 0
    np.testing.assert_allclose(scaled[:, 2], 0.0)


def test_train_val_scaling_separation():
    rng = np.random.default_rng(42)
    train_data = rng.normal(loc=10.0, scale=2.0, size=(80, 5)).astype(np.float32)
    val_data = rng.normal(loc=10.0, scale=2.0, size=(20, 5)).astype(np.float32)

    scaler = fit_train_scaler(train_data, method=ScalingMethod.STANDARD)
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)

    assert train_scaled.shape == (80, 5)
    assert val_scaled.shape == (20, 5)
    np.testing.assert_allclose(np.mean(train_scaled, axis=0), 0.0, atol=1e-5)
