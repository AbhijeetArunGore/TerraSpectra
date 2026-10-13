"""Feature scaling and normalization for hyperspectral and PCA-reduced data."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


class ScalingMethod(str, Enum):
    """Supported scaling and normalization strategies."""
    STANDARD = "standard"  # Z-score standardization (zero mean, unit variance)
    MINMAX = "minmax"      # Min-Max normalization (bounded to [0, 1] or specified range)


@dataclass
class ScalingMetadata:
    """Metadata describing the fitted scaler and feature statistics."""
    method: str
    feature_count: int
    means_or_mins: list[float]
    scales_or_ranges: list[float]
    constant_features_detected: list[int]
    is_fitted: bool
    documentation: str


class HyperspectralScaler:
    """
    Scaler supporting standard and min-max feature transformations.
    Guarantees safety against division by zero on constant-valued/zero-variance features.
    """

    def __init__(
        self,
        method: ScalingMethod | str = ScalingMethod.STANDARD,
        feature_range: Tuple[float, float] = (0.0, 1.0),
        clip_bounds: Optional[Tuple[float, float]] = None,
    ):
        if isinstance(method, str):
            method = ScalingMethod(method.lower())
        self.method = method
        self.feature_range = feature_range
        self.clip_bounds = clip_bounds
        self.scaler: Union[StandardScaler, MinMaxScaler, None] = None
        self.constant_feature_indices: list[int] = []
        self.is_fitted = False

    def _get_documentation(self) -> str:
        if self.method == ScalingMethod.STANDARD:
            return (
                "Standardization (Z-score): Centers features to mean=0 and variance=1. "
                "Maintains relative distance and outlier information; ideal for PCA and linear/gradient ML."
            )
        else:
            return (
                f"Min-Max Normalization: Scales features linearly into range {self.feature_range}. "
                "Guarantees bounded inputs (e.g. for RGB visualization or bounded neural activations), "
                "but sensitive to extreme outlier values."
            )

    def fit(self, data: np.ndarray) -> HyperspectralScaler:
        """
        Fit scaler parameters on training data.
        Accepts 2D samples (N, Features) or 3D cube (H, W, Features).
        """
        samples = self._to_2d(data)
        n_features = samples.shape[1]

        # Check for zero-variance / constant features
        stds = np.std(samples, axis=0)
        self.constant_feature_indices = [int(i) for i in np.where(stds < 1e-10)[0]]
        if self.constant_feature_indices:
            logger.warning(
                "Constant features detected at indices: %s. Handled with zero-variance safeguards.",
                self.constant_feature_indices,
            )

        if self.method == ScalingMethod.STANDARD:
            self.scaler = StandardScaler(copy=True)
            self.scaler.fit(samples)
            # Safeguard scikit-learn's internal scale_ against zero-division
            if self.scaler.scale_ is not None:
                self.scaler.scale_[self.scaler.scale_ < 1e-10] = 1.0
        elif self.method == ScalingMethod.MINMAX:
            self.scaler = MinMaxScaler(feature_range=self.feature_range, copy=True)
            self.scaler.fit(samples)
            if self.scaler.scale_ is not None:
                self.scaler.scale_[self.scaler.scale_ < 1e-10] = 1.0

        self.is_fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted parameters. Preserves original 2D or 3D shape."""
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("Scaler must be fitted before calling transform().")

        original_shape = data.shape
        samples = self._to_2d(data)

        transformed = self.scaler.transform(samples)

        # Ensure constant features map cleanly to 0
        if self.constant_feature_indices:
            for idx in self.constant_feature_indices:
                transformed[:, idx] = 0.0

        # Replace any non-finite values if present
        if not np.all(np.isfinite(transformed)):
            transformed = np.nan_to_num(transformed, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.clip_bounds is not None:
            c_min, c_max = self.clip_bounds
            transformed = np.clip(transformed, c_min, c_max)

        if len(original_shape) == 3:
            h, w, c = original_shape
            return transformed.reshape(h, w, c).astype(np.float32)
        return transformed.astype(np.float32)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit scaler and transform data in one step."""
        return self.fit(data).transform(data)

    def get_metadata(self) -> ScalingMetadata:
        """Extract metadata about fitted transformation parameters."""
        if not self.is_fitted or self.scaler is None:
            raise RuntimeError("Scaler is not fitted yet.")

        if self.method == ScalingMethod.STANDARD:
            means = [float(m) for m in self.scaler.mean_] if self.scaler.mean_ is not None else []
            scales = [float(s) for s in self.scaler.scale_] if self.scaler.scale_ is not None else []
        else:
            means = [float(m) for m in self.scaler.data_min_] if self.scaler.data_min_ is not None else []
            scales = [float(s) for s in self.scaler.data_range_] if self.scaler.data_range_ is not None else []

        return ScalingMetadata(
            method=self.method.value,
            feature_count=len(means),
            means_or_mins=[round(v, 4) for v in means],
            scales_or_ranges=[round(v, 4) for v in scales],
            constant_features_detected=self.constant_feature_indices,
            is_fitted=self.is_fitted,
            documentation=self._get_documentation(),
        )

    def save(self, model_path: str | Path, report_path: Optional[str | Path] = None) -> None:
        """Persist fitted scaler using joblib and optionally save JSON metadata report."""
        m_path = Path(model_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(m_path))

        if report_path is not None:
            r_path = Path(report_path)
            r_path.parent.mkdir(parents=True, exist_ok=True)
            with open(r_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.get_metadata()), f, indent=2)

        logger.info("Saved fitted scaler to %s", m_path)

    @classmethod
    def load(cls, model_path: str | Path) -> HyperspectralScaler:
        """Load previously saved fitted scaler from disk."""
        return joblib.load(str(model_path))

    @staticmethod
    def _to_2d(data: np.ndarray) -> np.ndarray:
        """Convert input array to 2D matrix (N, Features)."""
        if data.ndim == 2:
            return data.astype(np.float32)
        elif data.ndim == 3:
            h, w, c = data.shape
            return data.reshape(h * w, c).astype(np.float32)
        raise ValueError(f"Expected 2D or 3D array, got ndim={data.ndim} shape={data.shape}")


def fit_train_scaler(
    train_data: np.ndarray,
    method: ScalingMethod | str = ScalingMethod.STANDARD,
) -> HyperspectralScaler:
    """Helper to fit scaler strictly on training split."""
    scaler = HyperspectralScaler(method=method)
    scaler.fit(train_data)
    return scaler
