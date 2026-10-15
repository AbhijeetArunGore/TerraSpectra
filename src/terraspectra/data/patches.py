"""Spatial-spectral patch extraction utilities for hyperspectral 3D-CNN inputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PatchSample:
    """Container holding an extracted hyperspectral patch and metadata."""
    tensor: torch.Tensor        # Shape: (1, channels, depth, height, width) or (channels, depth, height, width)
    label: int                  # Ground-truth label of the center pixel
    center_row: int             # Row index in original unpadded cube
    center_col: int             # Column index in original unpadded cube
    patch_size: int             # Spatial window width/height (odd integer)
    spectral_depth: int         # Number of spectral bands or PCA components


def validate_patch_parameters(
    cube_shape: Tuple[int, ...],
    row: int,
    col: int,
    patch_size: int,
) -> None:
    """Validate spatial coordinates and patch size parameters."""
    if len(cube_shape) != 3:
        raise ValueError(f"Expected 3D hyperspectral cube (Height, Width, Depth), got shape {cube_shape}")

    h, w, _ = cube_shape

    if not isinstance(patch_size, int) or patch_size < 1:
        raise ValueError(f"patch_size must be a positive integer, got {patch_size}")

    if patch_size % 2 == 0:
        raise ValueError(
            f"patch_size must be an odd integer to have a well-defined center pixel, got {patch_size} (use {patch_size + 1} or {patch_size - 1})"
        )

    if not (0 <= row < h):
        raise IndexError(f"Row coordinate {row} out of bounds for cube with height {h}")

    if not (0 <= col < w):
        raise IndexError(f"Column coordinate {col} out of bounds for cube with width {w}")


def pad_cube_spatially(
    cube: np.ndarray,
    margin: int,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Pad the spatial dimensions (H, W) of the hyperspectral cube while keeping spectral bands untouched.

    Parameters
    ----------
    cube : np.ndarray
        Array of shape (Height, Width, Bands).
    margin : int
        Number of pixels to pad on each side (typically patch_size // 2).
    mode : str
        Numpy padding mode: 'reflect' (default, symmetric reflection without border doubling),
        'edge' (clamps to edge value), or 'constant' (zero padding).
    """
    if margin <= 0:
        return cube

    # Pad ((margin, margin), (margin, margin), (0, 0))
    pad_width = ((margin, margin), (margin, margin), (0, 0))
    if mode == "reflect":
        return np.pad(cube, pad_width, mode="reflect")
    elif mode == "edge":
        return np.pad(cube, pad_width, mode="edge")
    elif mode == "constant":
        return np.pad(cube, pad_width, mode="constant", constant_values=0)
    else:
        raise ValueError(f"Unsupported padding mode '{mode}'. Choose 'reflect', 'edge', or 'constant'.")


def extract_spatial_patch(
    cube: np.ndarray,
    row: int,
    col: int,
    patch_size: int = 7,
    gt: Optional[np.ndarray] = None,
    pad_mode: str = "reflect",
    add_batch_dim: bool = True,
    as_tensor: bool = True,
) -> PatchSample:
    """
    Extract a spatial-spectral patch centered at pixel (row, col).

    Parameters
    ----------
    cube : np.ndarray
        Input hyperspectral cube of shape (H, W, D).
    row : int
        0-indexed spatial row coordinate of center pixel.
    col : int
        0-indexed spatial column coordinate of center pixel.
    patch_size : int
        Spatial diameter of the square patch (must be an odd integer, e.g. 5, 7, 9).
    gt : Optional[np.ndarray]
        Optional ground truth label matrix of shape (H, W).
    pad_mode : str
        Spatial boundary padding strategy ('reflect', 'edge', 'constant').
    add_batch_dim : bool
        If True, returns shape (1, 1, D, P, P); otherwise (1, D, P, P).
    as_tensor : bool
        If True, converts output to torch.FloatTensor.

    Returns
    -------
    PatchSample
        Container holding patch tensor/array and center label.
    """
    validate_patch_parameters(cube.shape, row, col, patch_size)

    radius = patch_size // 2
    h, w, d = cube.shape

    # Extract target label of the center pixel
    label = int(gt[row, col]) if gt is not None else 0

    # Determine if spatial coordinates require border padding
    needs_padding = (row - radius < 0) or (row + radius >= h) or (col - radius < 0) or (col + radius >= w)

    if needs_padding:
        padded_cube = pad_cube_spatially(cube, margin=radius, mode=pad_mode)
        # Shift coordinate by radius in padded space
        padded_r = row + radius
        padded_c = col + radius
        raw_patch = padded_cube[
            padded_r - radius : padded_r + radius + 1,
            padded_c - radius : padded_c + radius + 1,
            :,
        ]
    else:
        raw_patch = cube[
            row - radius : row + radius + 1,
            col - radius : col + radius + 1,
            :,
        ]

    # raw_patch has shape (P, P, D).
    # 3D-CNN expects PyTorch 5D layout: (Batch, Channels, Depth, Height, Width).
    # Transpose (P, P, D) -> (D, P, P), then add channels dimension -> (1, D, P, P).
    patch_transposed = np.transpose(raw_patch, (2, 0, 1)).astype(np.float32)  # Shape: (D, P, P)
    patch_4d = np.expand_dims(patch_transposed, axis=0)                         # Shape: (1, D, P, P)

    if add_batch_dim:
        patch_out = np.expand_dims(patch_4d, axis=0)                            # Shape: (1, 1, D, P, P)
    else:
        patch_out = patch_4d

    tensor_out: Union[torch.Tensor, np.ndarray] = torch.from_numpy(patch_out) if as_tensor else patch_out

    return PatchSample(
        tensor=tensor_out,
        label=label,
        center_row=row,
        center_col=col,
        patch_size=patch_size,
        spectral_depth=d,
    )


def iter_pixel_patches(
    cube: np.ndarray,
    gt: Optional[np.ndarray] = None,
    patch_size: int = 7,
    exclude_background: bool = True,
    pad_mode: str = "reflect",
    max_samples: Optional[int] = None,
) -> Generator[PatchSample, None, None]:
    """
    Generator yielding spatial-spectral patches on demand without eager memory allocation.

    Parameters
    ----------
    cube : np.ndarray
        Hyperspectral cube (H, W, D).
    gt : Optional[np.ndarray]
        Ground-truth labels (H, W).
    patch_size : int
        Odd integer patch size.
    exclude_background : bool
        If True and gt is provided, skips pixels where label == 0 (unclassified).
    max_samples : Optional[int]
        Optional upper limit on yielded samples.
    """
    radius = patch_size // 2
    padded_cube = pad_cube_spatially(cube, margin=radius, mode=pad_mode)
    h, w, d = cube.shape

    count = 0
    for r in range(h):
        for c in range(w):
            lbl = int(gt[r, c]) if gt is not None else 0
            if exclude_background and gt is not None and lbl == 0:
                continue

            padded_r = r + radius
            padded_c = c + radius
            raw_patch = padded_cube[
                padded_r - radius : padded_r + radius + 1,
                padded_c - radius : padded_c + radius + 1,
                :,
            ]
            transposed = np.transpose(raw_patch, (2, 0, 1)).astype(np.float32)
            tensor_5d = torch.from_numpy(transposed).unsqueeze(0).unsqueeze(0)  # (1, 1, D, P, P)

            yield PatchSample(
                tensor=tensor_5d,
                label=lbl,
                center_row=r,
                center_col=c,
                patch_size=patch_size,
                spectral_depth=d,
            )

            count += 1
            if max_samples is not None and count >= max_samples:
                return
