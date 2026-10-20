"""Model evaluation, classification metrics, error analysis, and confusion matrix rendering."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from terraspectra.data.loader import SALINAS_CLASS_NAMES

logger = logging.getLogger(__name__)


@dataclass
class ClassEvaluationMetrics:
    """Detailed performance metrics for an individual land-cover class."""
    class_id: int               # Original ground truth integer ID (e.g., 1..16)
    mapped_index: int           # Contiguous 0-indexed integer model target (0..15)
    class_name: str             # Human-readable semantic class name
    precision: float            # True Positives / (True Positives + False Positives)
    recall: float               # True Positives / (True Positives + False Negatives)
    f1_score: float             # Harmonic mean of precision and recall
    support: int                # Number of ground-truth test samples in this class


@dataclass
class EvaluationReport:
    """Comprehensive test evaluation report containing overall, per-class, and confusion metrics."""
    overall_accuracy: float                 # Overall Accuracy (OA)
    average_accuracy: float                 # Average Accuracy (AA: unweighted mean recall)
    macro_precision: float                  # Arithmetic mean of precision across classes
    macro_recall: float                     # Arithmetic mean of recall across classes
    macro_f1: float                         # Arithmetic mean of F1-scores across classes
    weighted_f1: float                      # Support-weighted average F1-score
    cohen_kappa: float                      # Cohen's Kappa coefficient (inter-rater agreement)
    total_test_samples: int                 # Total evaluated labeled test samples
    num_classes: int                        # Number of evaluated classes
    per_class_metrics: List[ClassEvaluationMetrics] = field(default_factory=list)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    checkpoint_epoch: int = 0
    spatial_leakage_note: str = (
        "Spatial Autocorrelation Caveat: Random pixel-level stratified splitting distributes "
        "neighboring pixels across train, validation, and test sets. Because spatial patches "
        "(e.g., 7x7) overlap across adjacent pixels, spatial autocorrelation can yield overly "
        "optimistic evaluation compared to spatially disjoint flightline cross-validation."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation report to a JSON-serializable dictionary."""
        return {
            "overall_accuracy": self.overall_accuracy,
            "average_accuracy": self.average_accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "cohen_kappa": self.cohen_kappa,
            "total_test_samples": self.total_test_samples,
            "num_classes": self.num_classes,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_epoch": self.checkpoint_epoch,
            "per_class_metrics": [asdict(cm) for cm in self.per_class_metrics],
            "confusion_matrix": self.confusion_matrix,
            "class_names": self.class_names,
            "spatial_leakage_note": self.spatial_leakage_note,
        }

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """Save report to a JSON file on disk."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved evaluation report to '{target}'")
        return target


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[Union[str, torch.device]] = None,
    label_to_idx: Optional[Dict[int, int]] = None,
    idx_to_label: Optional[Dict[int, int]] = None,
    class_names: Optional[List[str]] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_epoch: int = 0,
) -> EvaluationReport:
    """
    Rigorously evaluate a trained 3D-CNN model on the held-out test DataLoader.

    Parameters
    ----------
    model : nn.Module
        Trained Hyperspectral3DCNN model.
    test_loader : DataLoader
        DataLoader providing held-out test batches.
    device : Optional[str | torch.device]
        Execution device. Defaults to model's existing device or CPU.
    label_to_idx : Optional[Dict[int, int]]
        Mapping from original class ID (e.g., 1..16) to contiguous model target (0..15).
    idx_to_label : Optional[Dict[int, int]]
        Reverse mapping from model target (0..15) to original class ID (1..16).
    class_names : Optional[List[str]]
        List of human-readable class names aligned with model targets.
    checkpoint_path : Optional[str]
        Path to the evaluated model checkpoint for documentation.
    checkpoint_epoch : int
        Epoch at which the evaluated checkpoint was saved.

    Returns
    -------
    EvaluationReport
        Structured report containing overall metrics, per-class breakdown, and confusion matrix.
    """
    if device is None:
        device = next(model.parameters()).device
    elif isinstance(device, str):
        device = torch.device(device)

    model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []

    # Inference without gradient computation or weight modification
    with torch.inference_mode():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=-1).cpu().numpy()
            targets = batch_y.cpu().numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    y_pred = np.array(all_preds, dtype=np.int64)
    y_true = np.array(all_targets, dtype=np.int64)

    total_samples = len(y_true)
    if total_samples == 0:
        logger.warning("Test DataLoader provided 0 samples. Returning empty EvaluationReport.")
        return EvaluationReport(
            overall_accuracy=0.0,
            average_accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            weighted_f1=0.0,
            cohen_kappa=0.0,
            total_test_samples=0,
            num_classes=0,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=checkpoint_epoch,
        )

    # Determine unique classes present in either true labels or model targets
    if label_to_idx is not None:
        num_classes = len(label_to_idx)
        target_indices = sorted(list(label_to_idx.values()))
    else:
        num_classes = max(int(np.max(y_true)), int(np.max(y_pred))) + 1
        target_indices = list(range(num_classes))

    # Overall Accuracy
    oa = float(accuracy_score(y_true, y_pred))

    # Per-class Precision, Recall, F1, Support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=target_indices,
        zero_division=0.0,
    )

    # Average Accuracy (AA: unweighted mean of per-class recalls)
    aa = float(np.mean(recall)) if len(recall) > 0 else 0.0

    # Macro & Weighted Averages
    macro_prec = float(np.mean(precision)) if len(precision) > 0 else 0.0
    macro_rec = float(np.mean(recall)) if len(recall) > 0 else 0.0
    macro_f1 = float(np.mean(f1)) if len(f1) > 0 else 0.0

    # Weighted F1
    total_support = int(np.sum(support))
    if total_support > 0:
        weighted_f1 = float(np.sum(f1 * support) / total_support)
    else:
        weighted_f1 = 0.0

    # Cohen's Kappa
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kappa = float(cohen_kappa_score(y_true, y_pred, labels=target_indices))
        if np.isnan(kappa):
            kappa = 0.0
    except Exception:
        kappa = 0.0

    # Confusion Matrix
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = confusion_matrix(y_true, y_pred, labels=target_indices)

    # Build per-class structured reports
    per_class_list: List[ClassEvaluationMetrics] = []
    names_list: List[str] = []

    for idx in target_indices:
        # Determine original class ID
        orig_class_id = idx_to_label.get(idx, idx + 1) if idx_to_label else idx + 1

        # Determine class name
        if class_names and idx < len(class_names):
            c_name = class_names[idx]
        elif orig_class_id < len(SALINAS_CLASS_NAMES):
            c_name = SALINAS_CLASS_NAMES[orig_class_id]
        else:
            c_name = f"Class_{orig_class_id}"

        names_list.append(c_name)

        per_class_list.append(
            ClassEvaluationMetrics(
                class_id=int(orig_class_id),
                mapped_index=int(idx),
                class_name=c_name,
                precision=float(precision[idx]),
                recall=float(recall[idx]),
                f1_score=float(f1[idx]),
                support=int(support[idx]),
            )
        )

    return EvaluationReport(
        overall_accuracy=oa,
        average_accuracy=aa,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        cohen_kappa=kappa,
        total_test_samples=total_samples,
        num_classes=num_classes,
        per_class_metrics=per_class_list,
        confusion_matrix=cm.tolist(),
        class_names=names_list,
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=checkpoint_epoch,
    )


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: List[str],
    normalize: bool = False,
    title: str = "Salinas 3D-CNN Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Render and optionally save a publication-quality confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray or List[List[int]]
        Square confusion matrix array (rows: true classes, cols: predicted classes).
    class_names : List[str]
        List of class label names corresponding to matrix rows/cols.
    normalize : bool
        If True, normalizes counts by ground-truth row totals.
    title : str
        Figure title.
    save_path : Optional[str | Path]
        Destination image file path (e.g., outputs/figures/salinas_confusion_matrix.png).
    figsize : Tuple[int, int]
        Dimensions of figure in inches.

    Returns
    -------
    plt.Figure
        Rendered Matplotlib figure object.
    """
    matrix = np.array(cm, dtype=np.float64)

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero for classes with zero support
        row_sums[row_sums == 0] = 1.0
        matrix_display = matrix / row_sums
        fmt = ".2f"
        cbar_label = "Normalized Proportion"
    else:
        matrix_display = matrix
        fmt = ".0f"
        cbar_label = "Sample Count"

    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix_display, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=11)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Ground-Truth Class",
        xlabel="Model Predicted Class",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    # Threshold for contrasting text color
    thresh = matrix_display.max() / 2.0 if matrix_display.max() > 0 else 0.5
    for i in range(n_classes):
        for j in range(n_classes):
            val = matrix_display[i, j]
            count = int(matrix[i, j])
            text = f"{val:{fmt}}" if normalize else f"{count}"
            if count > 0:
                ax.text(
                    j, i, text,
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=8 if n_classes > 10 else 10,
                )

    ax.grid(False)
    fig.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        logger.info(f"Saved confusion matrix figure to '{out_file}'")

    return fig
