"""Automated tests for hyperspectral spatial patch extraction and padding."""

import numpy as np
import pytest
import torch

from terraspectra.data.patches import (
    PatchSample,
    extract_spatial_patch,
    pad_cube_spatially,
    validate_patch_parameters,
    iter_pixel_patches,
)


@pytest.fixture
def synthetic_cube_and_gt():
    # Deterministic test cube: shape (40, 30, 204)
    np.random.seed(42)
    h, w, d = 40, 30, 204
    cube = np.arange(h * w * d, dtype=np.float32).reshape(h, w, d)
    gt = np.random.randint(0, 17, size=(h, w), dtype=np.int64)
    return cube, gt


def test_extract_spatial_patch_interior(synthetic_cube_and_gt):
    cube, gt = synthetic_cube_and_gt
    row, col = 15, 20
    patch_size = 7
    radius = patch_size // 2

    sample = extract_spatial_patch(cube, row, col, patch_size=patch_size, gt=gt, as_tensor=True)

    assert isinstance(sample, PatchSample)
    assert isinstance(sample.tensor, torch.Tensor)
    # Shape: (1, 1, Depth, Height, Width)
    assert sample.tensor.shape == (1, 1, 204, 7, 7)
    assert sample.label == int(gt[row, col])
    assert sample.center_row == row
    assert sample.center_col == col
    assert sample.spectral_depth == 204

    # Verify center pixel alignment:
    # In patch tensor of shape (1, 1, D, H, W), the center spatial index is [0, 0, :, radius, radius]
    extracted_center = sample.tensor[0, 0, :, radius, radius].numpy()
    expected_center = cube[row, col, :]
    np.testing.assert_allclose(extracted_center, expected_center)


def test_extract_spatial_patch_boundary_corners(synthetic_cube_and_gt):
    cube, gt = synthetic_cube_and_gt
    h, w, _ = cube.shape
    patch_size = 5
    radius = patch_size // 2

    # Top-Left corner (0, 0)
    tl_sample = extract_spatial_patch(cube, 0, 0, patch_size=patch_size, gt=gt, pad_mode="reflect")
    assert tl_sample.tensor.shape == (1, 1, 204, 5, 5)
    tl_center = tl_sample.tensor[0, 0, :, radius, radius].numpy()
    np.testing.assert_allclose(tl_center, cube[0, 0, :])

    # Bottom-Right corner (H-1, W-1)
    br_sample = extract_spatial_patch(cube, h - 1, w - 1, patch_size=patch_size, gt=gt, pad_mode="reflect")
    assert br_sample.tensor.shape == (1, 1, 204, 5, 5)
    br_center = br_sample.tensor[0, 0, :, radius, radius].numpy()
    np.testing.assert_allclose(br_center, cube[h - 1, w - 1, :])


def test_spatial_padding_modes():
    # Test 3D cube spatial padding
    cube = np.ones((10, 10, 5), dtype=np.float32)
    cube[0, 0, :] = 42.0

    # Margin 2 adds 2 on each spatial side -> 14 x 14 x 5
    padded_reflect = pad_cube_spatially(cube, margin=2, mode="reflect")
    assert padded_reflect.shape == (14, 14, 5)

    padded_edge = pad_cube_spatially(cube, margin=2, mode="edge")
    assert padded_edge.shape == (14, 14, 5)

    padded_const = pad_cube_spatially(cube, margin=2, mode="constant")
    assert padded_const.shape == (14, 14, 5)
    # Boundary padding should be 0 in constant mode
    assert padded_const[0, 0, 0] == 0.0

    with pytest.raises(ValueError):
        pad_cube_spatially(cube, margin=2, mode="unknown_mode")


def test_rejects_even_and_invalid_patch_sizes(synthetic_cube_and_gt):
    cube, _ = synthetic_cube_and_gt

    # Even patch sizes are ambiguous regarding the center pixel
    with pytest.raises(ValueError) as exc:
        extract_spatial_patch(cube, 10, 10, patch_size=6)
    assert "patch_size must be an odd integer" in str(exc.value)

    # Non-positive patch size
    with pytest.raises(ValueError):
        extract_spatial_patch(cube, 10, 10, patch_size=0)


def test_rejects_out_of_bounds_coordinates(synthetic_cube_and_gt):
    cube, _ = synthetic_cube_and_gt
    h, w, _ = cube.shape

    with pytest.raises(IndexError):
        extract_spatial_patch(cube, h + 5, 10, patch_size=7)

    with pytest.raises(IndexError):
        extract_spatial_patch(cube, 10, -1, patch_size=7)


def test_iter_pixel_patches_generator(synthetic_cube_and_gt):
    cube, gt = synthetic_cube_and_gt
    # Generate 5 patches
    samples = list(iter_pixel_patches(cube, gt, patch_size=5, max_samples=5))
    assert len(samples) == 5
    for s in samples:
        assert isinstance(s, PatchSample)
        assert s.tensor.shape == (1, 1, 204, 5, 5)
