"""PyTorch Dataset and DataLoader pipeline for hyperspectral spatial-spectral tensor batches."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from terraspectra.data.loader import SALINAS_CLASS_NAMES, SalinasDataset
from terraspectra.data.patches import pad_cube_spatially

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderMetadata:
    """Metadata detailing hyperspectral dataset splits, batching, and class mappings."""
    total_samples: int
    train_samples: int
    val_samples: int
    test_samples: int
    num_classes: int
    label_to_idx: Dict[int, int]
    idx_to_label: Dict[int, int]
    train_class_distribution: Dict[int, int]
    val_class_distribution: Dict[int, int]
    test_class_distribution: Dict[int, int]
    batch_size: int
    patch_size: int
    spectral_depth: int
    seed: int
    stratified: bool
    excluded_unlabeled: bool
    spatial_leakage_note: str = (
        "Spatial Autocorrelation Caveat: Random pixel-level stratified splitting distributes "
        "neighboring pixels across train, validation, and test sets. Because spatial patches "
        "(e.g., 7x7) overlap across adjacent pixels, spatial autocorrelation can yield overly "
        "optimistic evaluation compared to spatially disjoint or block cross-validation."
    )


class HyperspectralPatchDataset(Dataset):
    """
    PyTorch Dataset yielding 5D-compatible spatial-spectral patches and center-pixel targets.

    Expected sample output:
        patch_tensor: torch.FloatTensor of shape (1, spectral_depth, patch_size, patch_size)
        target: torch.LongTensor scalar with 0-indexed class index in [0, num_classes - 1]

    When collated by a PyTorch DataLoader, the resulting batch has dimensions:
        features: (batch_size, 1, spectral_depth, patch_size, patch_size)
        targets:  (batch_size,) of dtype torch.int64
    This matches the exact 5D layout expected by the Day 8 Hyperspectral3DCNN model.
    """

    def __init__(
        self,
        cube: np.ndarray,
        gt: np.ndarray,
        patch_size: int = 7,
        coords: Optional[Union[Sequence[Tuple[int, int]], np.ndarray]] = None,
        exclude_unlabeled: bool = True,
        pad_mode: str = "reflect",
        label_to_idx: Optional[Dict[int, int]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """
        Initialize the patch dataset.

        Parameters
        ----------
        cube : np.ndarray
            3D hyperspectral cube of shape (Height, Width, Spectral_Bands).
        gt : np.ndarray
            2D ground-truth matrix of shape (Height, Width).
        patch_size : int
            Odd integer spatial window diameter (e.g. 5, 7, 9).
        coords : Optional[Sequence[Tuple[int, int]] | np.ndarray]
            Explicit list of (row, col) coordinates to include. If None, coordinates
            are extracted automatically from `gt`.
        exclude_unlabeled : bool
            If True and coords is None, excludes pixels with label 0 (background).
        pad_mode : str
            Border padding strategy ('reflect', 'edge', 'constant').
        label_to_idx : Optional[Dict[int, int]]
            Mapping from ground-truth integer labels to contiguous 0-indexed targets.
            If None, inferred automatically from sorted unique labels.
        transform : Optional[Callable]
            Optional transformation applied to the patch tensor.
        """
        self._validate_inputs(cube, gt, patch_size)

        self.cube = cube
        self.gt = gt
        self.patch_size = patch_size
        self.radius = patch_size // 2
        self.pad_mode = pad_mode
        self.transform = transform
        self.spectral_depth = cube.shape[2]

        # Pre-pad cube once for O(1) patch slicing without repeated allocations
        self.padded_cube = pad_cube_spatially(cube, margin=self.radius, mode=pad_mode)

        # Coordinate selection
        if coords is not None:
            coords_arr = np.asarray(coords, dtype=np.int32)
            if len(coords_arr) == 0:
                raise ValueError("coords cannot be an empty sequence.")
            if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
                raise ValueError(f"coords must have shape (N, 2), got {coords_arr.shape}")
            self._validate_coords_bounds(coords_arr, cube.shape[0], cube.shape[1])
            self.coords = coords_arr
        else:
            if exclude_unlabeled:
                valid_r, valid_c = np.nonzero(gt != 0)
            else:
                h, w = gt.shape
                valid_r, valid_c = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                valid_r = valid_r.ravel()
                valid_c = valid_c.ravel()

            if len(valid_r) == 0:
                raise ValueError("No eligible labeled pixels found in ground truth.")

            self.coords = np.column_stack([valid_r, valid_c]).astype(np.int32)

        # Extract center labels for selected coordinates
        self.labels = np.array([int(gt[r, c]) for r, c in self.coords], dtype=np.int64)

        # Setup class mapping
        if label_to_idx is not None:
            # Validate that all labels present in coords are covered by provided mapping
            missing_labels = set(self.labels) - set(label_to_idx.keys())
            if missing_labels:
                raise ValueError(
                    f"Provided label_to_idx is missing mappings for ground-truth labels: {sorted(missing_labels)}"
                )
            self.label_to_idx = dict(label_to_idx)
        else:
            unique_labels = sorted(list(set(self.labels)))
            self.label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}

        self.idx_to_label = {idx: lbl for lbl, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)

        # Pre-map all integer targets for fast __getitem__ lookup
        self.targets = np.array([self.label_to_idx[lbl] for lbl in self.labels], dtype=np.int64)

    @staticmethod
    def _validate_inputs(cube: np.ndarray, gt: np.ndarray, patch_size: int) -> None:
        """Validate input array shapes, dimensions, and patch size."""
        if not isinstance(cube, np.ndarray):
            raise TypeError(f"cube must be a numpy ndarray, got {type(cube)}")
        if not isinstance(gt, np.ndarray):
            raise TypeError(f"gt must be a numpy ndarray, got {type(gt)}")

        if cube.ndim != 3:
            raise ValueError(f"cube must have 3 dimensions (Height, Width, Bands), got shape {cube.shape}")
        if gt.ndim != 2:
            raise ValueError(f"gt must have 2 dimensions (Height, Width), got shape {gt.shape}")

        if cube.shape[0] != gt.shape[0] or cube.shape[1] != gt.shape[1]:
            raise ValueError(
                f"Spatial dimensions mismatch: cube has ({cube.shape[0]}, {cube.shape[1]}) "
                f"while gt has ({gt.shape[0]}, {gt.shape[1]})"
            )

        if not isinstance(patch_size, int) or patch_size < 1:
            raise ValueError(f"patch_size must be a positive integer, got {patch_size}")

        if patch_size % 2 == 0:
            raise ValueError(
                f"patch_size must be an odd integer to maintain an unambiguous center pixel, got {patch_size}"
            )

    @staticmethod
    def _validate_coords_bounds(coords: np.ndarray, height: int, width: int) -> None:
        """Ensure all coordinates fall strictly within scene bounds."""
        min_r, min_c = coords[:, 0].min(), coords[:, 1].min()
        max_r, max_c = coords[:, 0].max(), coords[:, 1].max()

        if min_r < 0 or max_r >= height or min_c < 0 or max_c >= width:
            raise IndexError(
                f"Coordinate out of bounds: rows must be in [0, {height-1}], cols in [0, {width-1}]. "
                f"Found rows in [{min_r}, {max_r}], cols in [{min_c}, {max_c}]."
            )

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single spatial-spectral patch and target label.

        Returns
        -------
        patch : torch.Tensor
            Float32 tensor of shape (1, spectral_depth, patch_size, patch_size).
        target : torch.Tensor
            Long scalar tensor of mapped class index.
        """
        if idx < 0 or idx >= len(self.coords):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.coords)}")

        r, c = self.coords[idx]

        # Extract spatial-spectral patch from pre-padded cube
        # The slice [r : r + patch_size, c : c + patch_size] corresponds exactly
        # to the window centered at unpadded (r, c)
        raw_patch = self.padded_cube[r : r + self.patch_size, c : c + self.patch_size, :]

        # Transpose (patch_h, patch_w, depth) -> (depth, patch_h, patch_w)
        patch_transposed = np.transpose(raw_patch, (2, 0, 1)).astype(np.float32)

        # Add channels dimension: (1, depth, patch_h, patch_w)
        patch_4d = np.expand_dims(patch_transposed, axis=0)
        patch_tensor = torch.from_numpy(patch_4d)

        if self.transform is not None:
            patch_tensor = self.transform(patch_tensor)

        target_tensor = torch.tensor(self.targets[idx], dtype=torch.long)
        return patch_tensor, target_tensor

    def get_sample_metadata(self, idx: int) -> Dict[str, Any]:
        """Return human-readable metadata for a sample by index."""
        if idx < 0 or idx >= len(self.coords):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.coords)}")

        r, c = int(self.coords[idx, 0]), int(self.coords[idx, 1])
        orig_lbl = int(self.labels[idx])
        target_idx = int(self.targets[idx])

        class_name = (
            SALINAS_CLASS_NAMES[orig_lbl]
            if orig_lbl < len(SALINAS_CLASS_NAMES)
            else f"Class_{orig_lbl}"
        )

        return {
            "index": idx,
            "center_row": r,
            "center_col": c,
            "original_label": orig_lbl,
            "mapped_target": target_idx,
            "class_name": class_name,
            "patch_size": self.patch_size,
            "spectral_depth": self.spectral_depth,
            "tensor_shape": (1, self.spectral_depth, self.patch_size, self.patch_size),
        }


def split_hyperspectral_samples(
    coords: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split hyperspectral pixel coordinates into reproducible train, validation, and test subsets.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 2) containing (row, col) pixel coordinates.
    labels : np.ndarray
        Array of shape (N,) containing ground-truth integer labels.
    train_ratio : float
        Fraction of samples for training (must be > 0).
    val_ratio : float
        Fraction of samples for validation.
    test_ratio : float
        Fraction of samples for testing.
    seed : int
        Random seed for deterministic, reproducible splitting.
    stratify : bool
        Whether to preserve class distribution across all splits.

    Returns
    -------
    train_coords : np.ndarray
        Array of shape (N_train, 2).
    val_coords : np.ndarray
        Array of shape (N_val, 2).
    test_coords : np.ndarray
        Array of shape (N_test, 2).
    """
    if len(coords) != len(labels):
        raise ValueError(f"Length mismatch: coords has {len(coords)}, labels has {len(labels)}")
    if len(coords) == 0:
        raise ValueError("Cannot split an empty set of samples.")

    if train_ratio <= 0:
        raise ValueError(f"train_ratio must be positive, got {train_ratio}")
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio cannot be negative.")

    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-4):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.4f}")

    rng = np.random.default_rng(seed)
    n_total = len(coords)

    if not stratify:
        indices = np.arange(n_total)
        rng.shuffle(indices)

        n_train = int(np.round(n_total * train_ratio))
        n_val = int(np.round(n_total * val_ratio))

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]
        return coords[train_idx], coords[val_idx], coords[test_idx]

    # Stratified split: partition per class to preserve label frequencies
    train_indices_list: List[int] = []
    val_indices_list: List[int] = []
    test_indices_list: List[int] = []

    unique_classes = np.unique(labels)

    for c in unique_classes:
        c_indices = np.where(labels == c)[0]
        rng.shuffle(c_indices)
        n_c = len(c_indices)

        if n_c == 1:
            # Single sample: place into training set
            train_indices_list.extend(c_indices)
        elif n_c == 2:
            # Two samples: 1 train, 1 test (or val if test_ratio == 0)
            train_indices_list.append(c_indices[0])
            if test_ratio > 0:
                test_indices_list.append(c_indices[1])
            else:
                val_indices_list.append(c_indices[1])
        else:
            n_c_train = max(1, int(np.round(n_c * train_ratio)))
            n_c_val = int(np.round(n_c * val_ratio)) if val_ratio > 0 else 0

            # Safeguard so train + val does not exceed total samples
            if n_c_train + n_c_val >= n_c:
                if val_ratio > 0 and n_c > 2:
                    n_c_val = 1
                    n_c_train = n_c - n_c_val - (1 if test_ratio > 0 else 0)
                else:
                    n_c_val = 0
                    n_c_train = n_c - (1 if test_ratio > 0 else 0)

            train_indices_list.extend(c_indices[:n_c_train])
            val_indices_list.extend(c_indices[n_c_train : n_c_train + n_c_val])
            test_indices_list.extend(c_indices[n_c_train + n_c_val :])

    train_idx = np.array(train_indices_list, dtype=np.int64)
    val_idx = np.array(val_indices_list, dtype=np.int64)
    test_idx = np.array(test_indices_list, dtype=np.int64)

    # Shuffle final sets
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return coords[train_idx], coords[val_idx], coords[test_idx]


def create_dataloaders(
    cube_or_dataset: Union[np.ndarray, SalinasDataset],
    gt: Optional[np.ndarray] = None,
    patch_size: int = 7,
    batch_size: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify: bool = True,
    exclude_unlabeled: bool = True,
    pad_mode: str = "reflect",
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True,
    drop_last: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoaderMetadata]:
    """
    Construct training, validation, and test PyTorch DataLoaders for hyperspectral patches.

    Parameters
    ----------
    cube_or_dataset : np.ndarray or SalinasDataset
        Hyperspectral cube array (H, W, D) or loaded SalinasDataset instance.
    gt : Optional[np.ndarray]
        Ground-truth label array (H, W). Required if cube_or_dataset is a numpy array.
    patch_size : int
        Odd integer spatial patch diameter (default 7).
    batch_size : int
        Mini-batch size for DataLoader collation.
    train_ratio : float
        Fraction of labeled samples allocated to training (default 0.70).
    val_ratio : float
        Fraction of labeled samples allocated to validation (default 0.15).
    test_ratio : float
        Fraction of labeled samples allocated to testing (default 0.15).
    seed : int
        Random seed for reproducible split generation.
    stratify : bool
        Whether to preserve class balance across splits (default True).
    exclude_unlabeled : bool
        Whether to exclude background / unclassified pixels (label 0).
    pad_mode : str
        Spatial reflection padding strategy ('reflect', 'edge', 'constant').
    num_workers : int
        Number of subprocess workers for data loading (default 0 for Windows safety).
    pin_memory : bool
        Whether DataLoader pins memory in CPU RAM for faster GPU transfer.
    shuffle_train : bool
        Whether training DataLoader shuffles samples each epoch (default True).
    drop_last : bool
        Whether to discard the final incomplete batch.

    Returns
    -------
    train_loader : DataLoader
    val_loader : DataLoader
    test_loader : DataLoader
    metadata : DataLoaderMetadata
    """
    if isinstance(cube_or_dataset, SalinasDataset):
        cube = cube_or_dataset.cube
        gt_arr = cube_or_dataset.gt
    elif isinstance(cube_or_dataset, np.ndarray):
        if gt is None:
            raise ValueError("gt array must be provided when cube_or_dataset is a numpy array.")
        cube = cube_or_dataset
        gt_arr = gt
    else:
        raise TypeError(f"Unsupported input type for cube_or_dataset: {type(cube_or_dataset)}")

    # Extract all eligible coordinates and labels
    if exclude_unlabeled:
        valid_r, valid_c = np.nonzero(gt_arr != 0)
    else:
        h, w = gt_arr.shape
        valid_r, valid_c = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        valid_r = valid_r.ravel()
        valid_c = valid_c.ravel()

    if len(valid_r) == 0:
        raise ValueError("No eligible labeled pixels found in ground-truth matrix.")

    all_coords = np.column_stack([valid_r, valid_c]).astype(np.int32)
    all_labels = np.array([int(gt_arr[r, c]) for r, c in all_coords], dtype=np.int64)

    # Establish universal class mapping across the entire dataset
    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    idx_to_label = {idx: lbl for lbl, idx in label_to_idx.items()}

    # Perform reproducible stratified split on coordinates
    train_coords, val_coords, test_coords = split_hyperspectral_samples(
        coords=all_coords,
        labels=all_labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify=stratify,
    )

    # Instantiate datasets sharing the single pre-padded cube in RAM
    train_dataset = HyperspectralPatchDataset(
        cube=cube,
        gt=gt_arr,
        patch_size=patch_size,
        coords=train_coords,
        exclude_unlabeled=exclude_unlabeled,
        pad_mode=pad_mode,
        label_to_idx=label_to_idx,
    )

    val_dataset = HyperspectralPatchDataset(
        cube=cube,
        gt=gt_arr,
        patch_size=patch_size,
        coords=val_coords,
        exclude_unlabeled=exclude_unlabeled,
        pad_mode=pad_mode,
        label_to_idx=label_to_idx,
    )

    test_dataset = HyperspectralPatchDataset(
        cube=cube,
        gt=gt_arr,
        patch_size=patch_size,
        coords=test_coords,
        exclude_unlabeled=exclude_unlabeled,
        pad_mode=pad_mode,
        label_to_idx=label_to_idx,
    )

    # Create PyTorch DataLoaders
    # Note: Training loader shuffles each epoch; val and test loaders remain deterministic
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # Tally class distributions
    def _count_distribution(dataset: HyperspectralPatchDataset) -> Dict[int, int]:
        u, c = np.unique(dataset.labels, return_counts=True)
        return {int(lbl): int(cnt) for lbl, cnt in zip(u, c)}

    metadata = DataLoaderMetadata(
        total_samples=len(all_coords),
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        test_samples=len(test_dataset),
        num_classes=len(label_to_idx),
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        train_class_distribution=_count_distribution(train_dataset),
        val_class_distribution=_count_distribution(val_dataset),
        test_class_distribution=_count_distribution(test_dataset),
        batch_size=batch_size,
        patch_size=patch_size,
        spectral_depth=cube.shape[2],
        seed=seed,
        stratified=stratify,
        excluded_unlabeled=exclude_unlabeled,
    )

    logger.info(
        f"Created DataLoaders: Train={len(train_dataset)}, Val={len(val_dataset)}, "
        f"Test={len(test_dataset)}, BatchSize={batch_size}, PatchSize={patch_size}, "
        f"Classes={metadata.num_classes}"
    )

    return train_loader, val_loader, test_loader, metadata
