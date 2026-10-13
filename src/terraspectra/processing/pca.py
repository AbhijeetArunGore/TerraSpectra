"""Principal Component Analysis (PCA) dimensionality reduction for hyperspectral data."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


@dataclass
class PCAResults:
    """Container holding PCA transformation results and variance metrics."""
    reduced_cube: np.ndarray          # Spatial shape: (H, W, n_components)
    reduced_samples: np.ndarray       # Flattened shape: (H * W, n_components)
    explained_variance_ratio: list[float]
    cumulative_variance_ratio: list[float]
    total_explained_variance: float
    n_components: int
    original_bands: int
    is_exploratory: bool


class HyperspectralPCA:
    """
    Dimensionality reduction engine using Principal Component Analysis.
    Supports both exploratory scene-wide transformation and train/test leakage-free fitting.
    """

    def __init__(self, n_components: int = 15, random_state: int = 42):
        if n_components < 1:
            raise ValueError(f"n_components must be a positive integer, got {n_components}")
        self.n_components = n_components
        self.random_state = random_state
        self.pca: Optional[PCA] = None
        self.is_fitted = False

    @staticmethod
    def reshape_cube_to_samples(cube: np.ndarray) -> np.ndarray:
        """
        Reshape a 3D hyperspectral cube (H, W, Bands) into a 2D sample matrix (H * W, Bands).
        Avoids unnecessary memory duplication where possible.
        """
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D cube (H, W, Bands), got shape {cube.shape}")
        h, w, bands = cube.shape
        # Use ascontiguousarray and reshape to avoid memory fragmentation
        return np.ascontiguousarray(cube, dtype=np.float32).reshape(h * w, bands)

    @staticmethod
    def reshape_samples_to_cube(samples: np.ndarray, original_spatial_shape: Tuple[int, int]) -> np.ndarray:
        """Reshape 2D samples (H * W, n_components) back to 3D cube (H, W, n_components)."""
        h, w = original_spatial_shape
        n_components = samples.shape[1]
        return samples.reshape(h, w, n_components)

    def fit_exploratory(self, cube: np.ndarray) -> PCAResults:
        """
        Fit PCA on all spatial pixels of the cube for scene-wide exploratory visualization.
        NOTE: This is strictly for unsupervised scene exploration, NOT for training/evaluating ML classifiers.
        """
        if cube.ndim != 3:
            raise ValueError(f"Expected 3D cube, got shape {cube.shape}")

        h, w, original_bands = cube.shape
        if self.n_components > original_bands:
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed available spectral bands ({original_bands})"
            )

        samples = self.reshape_cube_to_samples(cube)

        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        reduced_samples = self.pca.fit_transform(samples)
        self.is_fitted = True

        reduced_cube = self.reshape_samples_to_cube(reduced_samples, (h, w))

        var_ratio = [float(v) for v in self.pca.explained_variance_ratio_]
        cum_var = [float(v) for v in np.cumsum(var_ratio)]
        total_var = float(cum_var[-1]) if cum_var else 0.0

        return PCAResults(
            reduced_cube=reduced_cube,
            reduced_samples=reduced_samples,
            explained_variance_ratio=var_ratio,
            cumulative_variance_ratio=cum_var,
            total_explained_variance=round(total_var, 6),
            n_components=self.n_components,
            original_bands=original_bands,
            is_exploratory=True,
        )

    def fit_train_data(self, train_samples: np.ndarray) -> HyperspectralPCA:
        """
        Fit PCA strictly on training pixel samples to prevent data leakage during model evaluation.
        """
        if train_samples.ndim != 2:
            raise ValueError(f"Expected 2D training samples (N_samples, Bands), got shape {train_samples.shape}")

        n_samples, n_bands = train_samples.shape
        if self.n_components > n_bands:
            raise ValueError(f"n_components ({self.n_components}) exceeds spectral bands ({n_bands})")
        if self.n_components > n_samples:
            raise ValueError(f"n_components ({self.n_components}) exceeds training sample count ({n_samples})")

        self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self.pca.fit(train_samples.astype(np.float32))
        self.is_fitted = True
        return self

    def transform_samples(self, samples: np.ndarray) -> np.ndarray:
        """Transform arbitrary sample matrix using previously fitted PCA."""
        if not self.is_fitted or self.pca is None:
            raise RuntimeError("HyperspectralPCA must be fitted before calling transform_samples.")
        return self.pca.transform(samples.astype(np.float32))

    def plot_explained_variance(self, results: PCAResults, save_path: str | Path) -> Path:
        """Plot scree plot and cumulative explained variance curve."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        components = np.arange(1, results.n_components + 1)
        var_ratios = np.array(results.explained_variance_ratio) * 100.0
        cum_var = np.array(results.cumulative_variance_ratio) * 100.0

        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)

        # Bar chart for individual variance
        color_bar = "#2b6cb0"
        bars = ax1.bar(components, var_ratios, color=color_bar, alpha=0.7, label="Individual Variance (%)")
        ax1.set_xlabel("Principal Component Index", fontsize=11)
        ax1.set_ylabel("Explained Variance (%)", color=color_bar, fontsize=11)
        ax1.tick_params(axis="y", labelcolor=color_bar)
        ax1.set_xticks(components)

        # Line chart for cumulative variance
        ax2 = ax1.twinx()
        color_line = "#c53030"
        ax2.plot(components, cum_var, color=color_line, marker="o", linewidth=2.2, label="Cumulative Variance (%)")
        ax2.set_ylabel("Cumulative Explained Variance (%)", color=color_line, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=color_line)
        ax2.set_ylim([0, 105])

        # Reference lines for 95% and 99%
        ax2.axhline(95.0, color="#718096", linestyle="--", alpha=0.7, label="95% Threshold")
        ax2.axhline(99.0, color="#a0aec0", linestyle=":", alpha=0.7, label="99% Threshold")

        total_captured = results.total_explained_variance * 100.0
        plt.title(
            f"PCA Explained Variance ({results.n_components} Components from {results.original_bands} Bands)\n"
            f"Total Variance Captured: {total_captured:.2f}%",
            fontsize=12,
        )
        fig.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        logger.info("Saved PCA explained variance plot to %s", save_path)
        return save_path

    def save_results(
        self,
        results: PCAResults,
        output_data_path: str | Path,
        report_path: str | Path,
    ) -> None:
        """Save reduced dataset (numpy compressed format) and metadata report."""
        out_path = Path(output_data_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(out_path),
            reduced_cube=results.reduced_cube.astype(np.float32),
            reduced_samples=results.reduced_samples.astype(np.float32),
            explained_variance_ratio=np.array(results.explained_variance_ratio, dtype=np.float32),
            cumulative_variance_ratio=np.array(results.cumulative_variance_ratio, dtype=np.float32),
        )

        rep_path = Path(report_path)
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "n_components": results.n_components,
            "original_bands": results.original_bands,
            "total_explained_variance_percent": round(results.total_explained_variance * 100.0, 3),
            "explained_variance_ratio": [round(v * 100.0, 3) for v in results.explained_variance_ratio],
            "cumulative_variance_ratio": [round(v * 100.0, 3) for v in results.cumulative_variance_ratio],
            "reduced_cube_shape": list(results.reduced_cube.shape),
            "reduced_samples_shape": list(results.reduced_samples.shape),
            "is_exploratory": results.is_exploratory,
        }
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        logger.info("Saved reduced PCA cube to %s and report to %s", out_path, rep_path)


def fit_exploratory_pca(cube: np.ndarray, n_components: int = 15) -> Tuple[HyperspectralPCA, PCAResults]:
    """Convenience helper to fit exploratory PCA on hyperspectral cube."""
    model = HyperspectralPCA(n_components=n_components)
    results = model.fit_exploratory(cube)
    return model, results


def fit_train_pca(train_samples: np.ndarray, n_components: int = 15) -> HyperspectralPCA:
    """Convenience helper to fit PCA strictly on training pixel samples."""
    model = HyperspectralPCA(n_components=n_components)
    return model.fit_train_data(train_samples)
