"""Data loading and validation module for the Salinas hyperspectral dataset."""

from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io

logger = logging.getLogger(__name__)

# Primary and fallback URLs for Salinas dataset
ZENODO_URLS = {
    "image": "https://zenodo.org/records/15771735/files/Salinas_corrected.mat?download=1",
    "gt": "https://zenodo.org/records/15771735/files/Salinas_gt.mat?download=1",
}

EHU_MIRROR_URLS = {
    "image": "http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
    "gt": "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
}

# 16 Ground truth class names documented in Salinas scene
SALINAS_CLASS_NAMES = [
    "Unclassified",
    "Broccoli_green_weeds_1",
    "Broccoli_green_weeds_2",
    "Fallow",
    "Fallow_rough_plow",
    "Fallow_smooth",
    "Stubble",
    "Celery",
    "Grapes_untrained",
    "Soil_vinyard_develop",
    "Corn_senesced_green_weeds",
    "Lettuce_romaine_4wk",
    "Lettuce_romaine_5wk",
    "Lettuce_romaine_6wk",
    "Lettuce_romaine_7wk",
    "Vinyard_untrained",
    "Vinyard_vertical_trellis",
]


@dataclass
class DatasetMetadata:
    """Metadata describing the loaded hyperspectral dataset."""
    image_shape: Tuple[int, int, int]
    spectral_band_count: int
    label_shape: Tuple[int, int]
    image_dtype: str
    label_dtype: str
    memory_usage_mb: float
    is_synthetic: bool
    image_path: str
    gt_path: str
    unique_classes: list[int]


@dataclass
class SalinasDataset:
    """Loaded and validated Salinas hyperspectral dataset container."""
    cube: np.ndarray
    gt: np.ndarray
    metadata: DatasetMetadata

    @property
    def rows(self) -> int:
        return self.cube.shape[0]

    @property
    def cols(self) -> int:
        return self.cube.shape[1]

    @property
    def bands(self) -> int:
        return self.cube.shape[2]


class SalinasLoader:
    """Loader and validator for Salinas hyperspectral scene (.mat format)."""

    def __init__(self, raw_data_dir: str | Path = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self.image_filename = "Salinas_corrected.mat"
        self.gt_filename = "Salinas_gt.mat"

    @property
    def image_path(self) -> Path:
        return self.raw_data_dir / self.image_filename

    @property
    def gt_path(self) -> Path:
        return self.raw_data_dir / self.gt_filename

    def download_if_missing(self, timeout_sec: int = 60) -> bool:
        """Download Salinas dataset files if not already present."""
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        files_to_download = [
            (self.image_path, ZENODO_URLS["image"], EHU_MIRROR_URLS["image"]),
            (self.gt_path, ZENODO_URLS["gt"], EHU_MIRROR_URLS["gt"]),
        ]

        for target_path, primary_url, mirror_url in files_to_download:
            if target_path.exists() and target_path.stat().st_size > 1024:
                continue

            downloaded = False
            for url in [primary_url, mirror_url]:
                try:
                    logger.info("Downloading %s from %s...", target_path.name, url)
                    headers = {"User-Agent": "Mozilla/5.0 (TerraSpectra Hyperspectral Pipeline)"}
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=timeout_sec) as response, open(target_path, "wb") as out_file:
                        data = response.read()
                        out_file.write(data)
                    if target_path.stat().st_size > 1024:
                        downloaded = True
                        logger.info("Successfully downloaded %s (%d bytes).", target_path.name, target_path.stat().st_size)
                        break
                except Exception as e:
                    logger.warning("Failed downloading %s from %s: %s", target_path.name, url, e)

            if not downloaded and target_path.exists():
                target_path.unlink()  # remove partial download
                return False

        return self.image_path.exists() and self.gt_path.exists()

    @staticmethod
    def _extract_mat_array(mat_dict: dict, preferred_keys: list[str]) -> np.ndarray:
        """Extract the numeric array from a MATLAB .mat dictionary."""
        # Try preferred keys first
        for key in preferred_keys:
            if key in mat_dict and isinstance(mat_dict[key], np.ndarray):
                return mat_dict[key]

        # Case-insensitive search
        lower_preferred = [k.lower() for k in preferred_keys]
        for key, val in mat_dict.items():
            if key.lower() in lower_preferred and isinstance(val, np.ndarray):
                return val

        # Otherwise find first non-meta array
        for key, val in mat_dict.items():
            if not key.startswith("__") and isinstance(val, np.ndarray) and val.ndim in (2, 3):
                return val

        raise KeyError(
            f"Could not find valid hyperspectral array in .mat file. Keys present: {list(mat_dict.keys())}"
        )

    def load(
        self,
        auto_download: bool = False,
        allow_synthetic_fallback: bool = False,
    ) -> SalinasDataset:
        """
        Load and validate the Salinas hyperspectral dataset.

        Parameters
        ----------
        auto_download : bool
            If True, attempts to download the dataset from Zenodo/EHU mirrors if missing.
        allow_synthetic_fallback : bool
            If True and actual files are missing, generates a small synthetic
            test fixture so automated pipelines and offline checks can run.
        """
        if not self.image_path.exists() or not self.gt_path.exists():
            if auto_download:
                self.download_if_missing()

            if not self.image_path.exists() or not self.gt_path.exists():
                if allow_synthetic_fallback:
                    logger.warning(
                        "Salinas dataset files not found at %s. Creating synthetic fixture for testing.",
                        self.raw_data_dir,
                    )
                    return self._create_and_load_synthetic()

                missing = []
                if not self.image_path.exists():
                    missing.append(str(self.image_path))
                if not self.gt_path.exists():
                    missing.append(str(self.gt_path))
                raise FileNotFoundError(
                    f"Required Salinas dataset files are missing: {missing}.\n"
                    f"Please place 'Salinas_corrected.mat' and 'Salinas_gt.mat' in '{self.raw_data_dir}' "
                    f"or download them from https://zenodo.org/records/15771735 or "
                    f"https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes"
                )

        # Load MATLAB .mat files
        raw_mat = scipy.io.loadmat(str(self.image_path))
        gt_mat = scipy.io.loadmat(str(self.gt_path))

        cube = self._extract_mat_array(raw_mat, ["salinas_corrected", "salinas", "data"])
        gt = self._extract_mat_array(gt_mat, ["salinas_gt", "gt", "labels"])

        return self.validate(cube, gt, str(self.image_path), str(self.gt_path), is_synthetic=False)

    def validate(
        self,
        cube: np.ndarray,
        gt: np.ndarray,
        image_path: str,
        gt_path: str,
        is_synthetic: bool = False,
    ) -> SalinasDataset:
        """Validate shapes, data types, and compatibility between cube and labels."""
        if not isinstance(cube, np.ndarray):
            raise TypeError(f"Hyperspectral cube must be a numpy.ndarray, got {type(cube)}")
        if not isinstance(gt, np.ndarray):
            raise TypeError(f"Ground truth labels must be a numpy.ndarray, got {type(gt)}")

        if cube.ndim != 3:
            raise ValueError(f"Hyperspectral cube must be 3-dimensional (H, W, Bands), got shape {cube.shape}")
        if gt.ndim != 2:
            raise ValueError(f"Ground-truth labels must be 2-dimensional (H, W), got shape {gt.shape}")

        rows, cols, bands = cube.shape
        gt_rows, gt_cols = gt.shape

        if (rows, cols) != (gt_rows, gt_cols):
            raise ValueError(
                f"Spatial dimension mismatch: cube spatial dimensions ({rows}, {cols}) do not match "
                f"ground-truth dimensions ({gt_rows}, {gt_cols})"
            )

        if bands < 1:
            raise ValueError(f"Invalid spectral band count: {bands}")

        # Check numeric data types
        if not np.issubdtype(cube.dtype, np.number):
            raise TypeError(f"Cube data type must be numeric, got {cube.dtype}")
        if not np.issubdtype(gt.dtype, np.integer):
            gt = gt.astype(np.int64)

        memory_usage_mb = (cube.nbytes + gt.nbytes) / (1024 * 1024)
        unique_classes = sorted([int(c) for c in np.unique(gt)])

        metadata = DatasetMetadata(
            image_shape=cube.shape,
            spectral_band_count=bands,
            label_shape=gt.shape,
            image_dtype=str(cube.dtype),
            label_dtype=str(gt.dtype),
            memory_usage_mb=round(memory_usage_mb, 2),
            is_synthetic=is_synthetic,
            image_path=image_path,
            gt_path=gt_path,
            unique_classes=unique_classes,
        )

        logger.info(
            "Dataset validated successfully: shape=%s, bands=%d, gt_shape=%s, memory=%.2f MB",
            cube.shape,
            bands,
            gt.shape,
            memory_usage_mb,
        )

        return SalinasDataset(cube=cube, gt=gt, metadata=metadata)

    def _create_and_load_synthetic(self) -> SalinasDataset:
        """Generate a realistic small synthetic fixture for automated testing."""
        fixture_dir = Path("data/test_fixtures")
        fixture_dir.mkdir(parents=True, exist_ok=True)
        cube_path = fixture_dir / "synthetic_salinas.mat"
        gt_path = fixture_dir / "synthetic_salinas_gt.mat"

        cube, gt = create_synthetic_salinas_fixture(cube_path, gt_path, shape=(50, 40, 204), n_classes=16)
        return self.validate(cube, gt, str(cube_path), str(gt_path), is_synthetic=True)


def create_synthetic_salinas_fixture(
    cube_path: Optional[str | Path] = None,
    gt_path: Optional[str | Path] = None,
    shape: Tuple[int, int, int] = (50, 40, 204),
    n_classes: int = 16,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a deterministic synthetic Salinas fixture mimicking AVIRIS spectral profiles.
    Used for automated unit testing and offline environments.
    """
    rng = np.random.default_rng(random_seed)
    h, w, bands = shape

    # Create ground truth mask with patches
    gt = np.zeros((h, w), dtype=np.int64)
    cell_h, cell_w = max(1, h // 4), max(1, w // 4)
    cls_idx = 1
    for r in range(0, h, cell_h):
        for c in range(0, w, cell_w):
            if cls_idx <= n_classes:
                gt[r : r + cell_h, c : c + cell_w] = cls_idx
                cls_idx += 1

    # Base continuous spectral curves with noise
    wavelengths = np.linspace(400, 2500, bands)
    base_spectrum = np.sin(wavelengths / 200.0) * 500.0 + 1500.0

    cube = np.zeros((h, w, bands), dtype=np.float32)
    for c in range(n_classes + 1):
        mask = gt == c
        class_shift = c * 50.0
        class_sig = base_spectrum + class_shift
        noise = rng.normal(0, 15, size=(int(np.sum(mask)), bands)).astype(np.float32)
        cube[mask] = class_sig + noise

    if cube_path:
        scipy.io.savemat(str(cube_path), {"salinas_corrected": cube})
    if gt_path:
        scipy.io.savemat(str(gt_path), {"salinas_gt": gt})

    return cube, gt


def load_salinas_data(
    raw_data_dir: str | Path = "data/raw",
    allow_synthetic_fallback: bool = False,
) -> SalinasDataset:
    """Convenience functional loader for Salinas hyperspectral dataset."""
    loader = SalinasLoader(raw_data_dir=raw_data_dir)
    return loader.load(allow_synthetic_fallback=allow_synthetic_fallback)
