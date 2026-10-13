"""Exploratory hyperspectral data analysis, spectral signatures, and false-color compositing."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution and report generation
import matplotlib.pyplot as plt
import numpy as np

from terraspectra.data.loader import SALINAS_CLASS_NAMES, SalinasDataset

logger = logging.getLogger(__name__)


@dataclass
class QualityAuditReport:
    """Numerical quality audit of the hyperspectral cube."""
    total_pixels: int
    nan_count: int
    inf_count: int
    min_value: float
    max_value: float
    mean_value: float
    std_value: float
    is_finite: bool


@dataclass
class ExplorationSummary:
    """Full exploration summary metadata for Day 3."""
    spatial_dimensions: Tuple[int, int]
    spectral_bands: int
    cube_dtype: str
    labels_dtype: str
    georeferencing_status: str
    coordinate_system: str
    quality_audit: QualityAuditReport
    class_distribution: Dict[int, Dict[str, int | str | float]]
    false_color_bands_used: Tuple[int, int, int]


class HyperspectralExplorer:
    """Exploration engine for hyperspectral scene analysis."""

    def __init__(self, dataset: SalinasDataset, output_dir: str | Path = "outputs"):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def audit_numerical_quality(self) -> QualityAuditReport:
        """Check for NaN, Inf, non-finite values, and general numerical range."""
        cube = self.dataset.cube
        nan_count = int(np.isnan(cube).sum())
        inf_count = int(np.isinf(cube).sum())
        is_finite = bool(nan_count == 0 and inf_count == 0)

        min_val = float(np.min(cube))
        max_val = float(np.max(cube))
        mean_val = float(np.mean(cube))
        std_val = float(np.std(cube))

        return QualityAuditReport(
            total_pixels=int(cube.shape[0] * cube.shape[1]),
            nan_count=nan_count,
            inf_count=inf_count,
            min_value=round(min_val, 4),
            max_value=round(max_val, 4),
            mean_value=round(mean_val, 4),
            std_value=round(std_val, 4),
            is_finite=is_finite,
        )

    def compute_class_distribution(self) -> Dict[int, Dict[str, int | str | float]]:
        """Compute pixel frequencies and percentages for each ground-truth class."""
        gt = self.dataset.gt
        total_pixels = gt.size
        unique_classes, counts = np.unique(gt, return_counts=True)

        dist: Dict[int, Dict[str, int | str | float]] = {}
        for cls_id, count in zip(unique_classes, counts):
            cid = int(cls_id)
            cname = SALINAS_CLASS_NAMES[cid] if cid < len(SALINAS_CLASS_NAMES) else f"Class_{cid}"
            dist[cid] = {
                "class_name": cname,
                "pixel_count": int(count),
                "percentage": round(float(count / total_pixels * 100), 2),
            }
        return dist

    def generate_false_color_composite(
        self,
        bands: Tuple[int, int, int] = (45, 25, 15),
        save_path: Optional[str | Path] = None,
    ) -> np.ndarray:
        """
        Create a false-color RGB composite from specified band indices.
        Applies 2nd-98th percentile stretching for enhanced visual contrast.

        Parameters
        ----------
        bands : Tuple[int, int, int]
            0-indexed (Red, Green, Blue) band indices from the cube.
        save_path : Optional[str | Path]
            Where to save the PNG image.
        """
        cube = self.dataset.cube
        n_bands = cube.shape[2]
        r_idx, g_idx, b_idx = bands

        for idx, name in zip(bands, ["Red", "Green", "Blue"]):
            if not (0 <= idx < n_bands):
                raise ValueError(f"{name} band index {idx} out of range for cube with {n_bands} bands.")

        raw_rgb = np.stack([cube[:, :, r_idx], cube[:, :, g_idx], cube[:, :, b_idx]], axis=-1).astype(np.float32)

        # Percentile contrast stretch per channel
        rgb_stretched = np.zeros_like(raw_rgb)
        for ch in range(3):
            ch_data = raw_rgb[:, :, ch]
            p_low, p_high = np.percentile(ch_data, (2.0, 98.0))
            if p_high > p_low:
                stretched = np.clip((ch_data - p_low) / (p_high - p_low), 0.0, 1.0)
            else:
                stretched = np.zeros_like(ch_data)
            rgb_stretched[:, :, ch] = stretched

        if save_path is None:
            save_path = self.figures_dir / "salinas_false_color_composite.png"
        save_path = Path(save_path)

        plt.figure(figsize=(8, 12), dpi=150)
        plt.imshow(rgb_stretched)
        plt.title(f"Salinas False-Color Composite (Bands R:{r_idx}, G:{g_idx}, B:{b_idx})\n[Array Pixel Coordinates]", fontsize=12)
        plt.xlabel("Pixel Column Index (0 to 216)")
        plt.ylabel("Pixel Row Index (0 to 511)")
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        logger.info("Saved false-color composite to %s", save_path)
        return rgb_stretched

    def generate_spectral_signatures(
        self,
        classes_to_plot: Optional[List[int]] = None,
        save_path: Optional[str | Path] = None,
    ) -> Dict[int, Dict[str, List[float]]]:
        """
        Extract mean and standard deviation spectral signatures for ground truth classes.
        """
        cube = self.dataset.cube
        gt = self.dataset.gt
        n_bands = cube.shape[2]
        band_indices = np.arange(n_bands)

        unique_classes = sorted(np.unique(gt))
        if classes_to_plot is None:
            # Default: plot non-background classes (1 to 16)
            classes_to_plot = [c for c in unique_classes if c != 0]

        signatures: Dict[int, Dict[str, List[float]]] = {}

        plt.figure(figsize=(14, 7), dpi=150)
        cmap = plt.get_cmap("tab20")

        for idx, cls_id in enumerate(classes_to_plot):
            mask = gt == cls_id
            if not np.any(mask):
                continue

            pixels = cube[mask]  # shape: (N_pixels, n_bands)
            mean_sig = np.mean(pixels, axis=0)
            std_sig = np.std(pixels, axis=0)

            cname = SALINAS_CLASS_NAMES[cls_id] if cls_id < len(SALINAS_CLASS_NAMES) else f"Class_{cls_id}"
            signatures[int(cls_id)] = {
                "class_name": cname,
                "mean": [round(float(v), 2) for v in mean_sig],
                "std": [round(float(v), 2) for v in std_sig],
            }

            color = cmap(idx % 20)
            plt.plot(band_indices, mean_sig, label=f"{cls_id}: {cname}", color=color, linewidth=1.8)
            plt.fill_between(band_indices, mean_sig - std_sig, mean_sig + std_sig, color=color, alpha=0.15)

        plt.title("Representative Spectral Signatures (Mean ± 1 SD per Class)\n[Salinas Scene]", fontsize=13)
        plt.xlabel("Spectral Band Index (0 to 203)")
        plt.ylabel("Calibrated Radiance / Digital Value")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)
        plt.tight_layout()

        if save_path is None:
            save_path = self.figures_dir / "salinas_spectral_signatures.png"
        save_path = Path(save_path)

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        logger.info("Saved spectral signatures to %s", save_path)
        return signatures

    def generate_class_distribution_plot(self, save_path: Optional[str | Path] = None) -> Path:
        """Plot and save class pixel frequency bar chart."""
        dist = self.compute_class_distribution()
        class_ids = [cid for cid in dist.keys() if cid != 0]  # Focus on agricultural classes
        names = [str(dist[cid]["class_name"]) for cid in class_ids]
        counts = [int(dist[cid]["pixel_count"]) for cid in class_ids]

        plt.figure(figsize=(12, 6), dpi=150)
        bars = plt.barh(names, counts, color="#2b5c8f", edgecolor="#1a365d")
        plt.xlabel("Labeled Pixels Count")
        plt.title("Salinas Ground-Truth Class Pixel Frequencies (Excluding Unclassified)", fontsize=13)
        plt.gca().invert_yaxis()  # Top-down order

        for bar in bars:
            w = bar.get_width()
            plt.text(w + 100, bar.get_y() + bar.get_height() / 2, f"{int(w):,}", va="center", fontsize=8)

        plt.tight_layout()

        if save_path is None:
            save_path = self.figures_dir / "salinas_class_distribution.png"
        save_path = Path(save_path)

        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        return save_path

    def run_full_exploration(self) -> ExplorationSummary:
        """Execute full Day 3 exploration workflow and generate reports."""
        quality = self.audit_numerical_quality()
        dist = self.compute_class_distribution()
        false_color_bands = (45, 25, 15)

        self.generate_false_color_composite(bands=false_color_bands)
        self.generate_spectral_signatures()
        self.generate_class_distribution_plot()

        # Investigate geospatial metadata
        # Notice: In MATLAB Salinas .mat, no GeoTIFF/affine transform, EPSG code, or GPS polygon exists
        georeferencing_status = (
            "No geospatial CRS, affine transform, or GPS coordinates are stored in the Salinas .mat file. "
            "Spatial coordinates are strictly 2D array row/column pixel coordinates: Rows 0..511, Columns 0..216."
        )
        coordinate_system = "Pixel Grid Coordinates (Rows: 512, Columns: 217, Bands: 204)"

        summary = ExplorationSummary(
            spatial_dimensions=(self.dataset.rows, self.dataset.cols),
            spectral_bands=self.dataset.bands,
            cube_dtype=str(self.dataset.cube.dtype),
            labels_dtype=str(self.dataset.gt.dtype),
            georeferencing_status=georeferencing_status,
            coordinate_system=coordinate_system,
            quality_audit=quality,
            class_distribution=dist,
            false_color_bands_used=false_color_bands,
        )

        report_file = self.reports_dir / "day3_exploration_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, indent=2)

        logger.info("Saved exploration report to %s", report_file)
        return summary
