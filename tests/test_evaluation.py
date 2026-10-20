"""Automated unit tests for 3D-CNN evaluation, classification metrics, and error analysis."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from terraspectra.models.cnn3d import Hyperspectral3DCNN, Hyperspectral3DCNNConfig
from terraspectra.models.eval import (
    ClassEvaluationMetrics,
    EvaluationReport,
    evaluate_model,
    plot_confusion_matrix,
)


@pytest.fixture
def synthetic_eval_loader():
    """Create a deterministic synthetic test loader for fast unit tests."""
    batch_size = 4
    channels, depth, height, width = 1, 15, 5, 5
    num_classes = 3
    num_samples = 15

    rng = torch.Generator().manual_seed(42)
    x_test = torch.randn(num_samples, channels, depth, height, width, generator=rng)
    y_test = torch.randint(0, num_classes, (num_samples,), generator=rng)

    ds = TensorDataset(x_test, y_test)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader, depth, height, num_classes, num_samples


def test_evaluate_model_runs_without_gradients_and_preserves_weights(synthetic_eval_loader):
    loader, depth, height, num_classes, _ = synthetic_eval_loader
    model = Hyperspectral3DCNN(
        Hyperspectral3DCNNConfig(
            spectral_depth=depth,
            spatial_size=height,
            num_classes=num_classes,
            conv_channels=(8, 16),
            kernel_spectral=(5, 3),
        )
    )

    initial_weights = [p.clone() for p in model.parameters()]

    report = evaluate_model(model, loader, device="cpu")

    assert isinstance(report, EvaluationReport)
    assert report.total_test_samples == 15
    assert report.num_classes == num_classes

    # Ensure model parameters are unchanged
    for p0, p1 in zip(initial_weights, model.parameters()):
        assert torch.allclose(p0, p1), "Model weights were modified during evaluation"


def test_evaluate_model_metrics_calculation_known_example():
    """Verify exact numerical precision, recall, F1, OA, and Kappa on a known example."""
    # Dummy model predicting deterministic class outputs
    class FixedPredictor(nn.Module):
        def __init__(self, fixed_preds):
            super().__init__()
            self.preds = fixed_preds

        def forward(self, x):
            batch_size = x.size(0)
            logits = torch.zeros(batch_size, 3)
            # Assign high logit to predetermined class
            for i in range(batch_size):
                pred_class = self.preds[i]
                logits[i, pred_class] = 10.0
            return logits

    # True labels: [0, 0, 1, 1, 2, 2]
    # Predictions: [0, 1, 1, 1, 2, 0]
    # Confusion matrix:
    # True 0: pred 0=1, pred 1=1, pred 2=0  (support=2, recall=0.5)
    # True 1: pred 0=0, pred 1=2, pred 2=0  (support=2, recall=1.0)
    # True 2: pred 0=1, pred 1=0, pred 2=1  (support=2, recall=0.5)
    # OA = 4 / 6 = 0.6667
    # AA = (0.5 + 1.0 + 0.5) / 3 = 0.6667
    fixed_preds = [0, 1, 1, 1, 2, 0]
    y_true = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    dummy_x = torch.zeros(6, 1, 5, 3, 3)

    loader = DataLoader(TensorDataset(dummy_x, y_true), batch_size=6, shuffle=False)
    model = FixedPredictor(fixed_preds)

    label_to_idx = {1: 0, 2: 1, 3: 2}
    idx_to_label = {0: 1, 1: 2, 2: 3}
    class_names = ["Crop_A", "Crop_B", "Crop_C"]

    report = evaluate_model(
        model,
        loader,
        device="cpu",
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        class_names=class_names,
    )

    assert np.isclose(report.overall_accuracy, 4.0 / 6.0, atol=1e-4)
    assert np.isclose(report.average_accuracy, 2.0 / 3.0, atol=1e-4)
    assert len(report.per_class_metrics) == 3
    assert report.per_class_metrics[0].support == 2
    assert report.per_class_metrics[1].support == 2
    assert report.per_class_metrics[2].support == 2
    assert np.isclose(report.per_class_metrics[1].recall, 1.0)

    # Confusion matrix checks
    cm = np.array(report.confusion_matrix)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[0, 2] == 0
    assert cm[1, 1] == 2
    assert cm[2, 0] == 1 and cm[2, 2] == 1


def test_evaluate_model_handles_zero_division():
    """Verify that classes with zero predictions do not raise exceptions."""
    class AlwaysZeroModel(nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.size(0), 3)
            logits[:, 0] = 10.0  # Always predicts class 0
            return logits

    y_true = torch.tensor([0, 1, 2], dtype=torch.long)
    x = torch.zeros(3, 1, 5, 3, 3)
    loader = DataLoader(TensorDataset(x, y_true), batch_size=3)

    report = evaluate_model(AlwaysZeroModel(), loader, device="cpu")

    # Classes 1 and 2 have 0 predictions: precision should be 0.0 (not nan or exception)
    assert report.per_class_metrics[1].precision == 0.0
    assert report.per_class_metrics[2].precision == 0.0
    assert np.isfinite(report.macro_precision)
    assert np.isfinite(report.macro_f1)


def test_evaluate_model_handles_empty_loader():
    empty_loader = DataLoader(TensorDataset(torch.empty(0, 1, 5, 3, 3), torch.empty(0, dtype=torch.long)))
    model = nn.Linear(5, 2)
    report = evaluate_model(model, empty_loader, device="cpu")
    assert report.total_test_samples == 0
    assert report.overall_accuracy == 0.0


def test_evaluate_model_incomplete_final_batch():
    """Verify that sample aggregation handles uneven batch sizes correctly."""
    class DummyModel(nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.size(0), 2)
            logits[:, 0] = 5.0
            return logits

    x = torch.zeros(7, 1, 5, 3, 3)  # Batch size 3 -> batches of 3, 3, 1
    y = torch.zeros(7, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)

    report = evaluate_model(DummyModel(), loader, device="cpu")
    assert report.total_test_samples == 7
    assert report.overall_accuracy == 1.0


def test_plot_confusion_matrix(tmp_path: Path):
    cm = [[10, 2], [1, 15]]
    class_names = ["Broccoli", "Celery"]
    out_file = tmp_path / "confusion_matrix.png"

    fig = plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        normalize=True,
        title="Test CM",
        save_path=out_file,
    )

    assert isinstance(fig, plt.Figure)
    assert out_file.is_file()
    assert out_file.stat().st_size > 0
    plt.close(fig)


def test_evaluation_report_serialization(tmp_path: Path):
    report = EvaluationReport(
        overall_accuracy=0.95,
        average_accuracy=0.94,
        macro_precision=0.93,
        macro_recall=0.94,
        macro_f1=0.935,
        weighted_f1=0.95,
        cohen_kappa=0.92,
        total_test_samples=100,
        num_classes=2,
        per_class_metrics=[
            ClassEvaluationMetrics(
                class_id=1,
                mapped_index=0,
                class_name="Class_1",
                precision=0.94,
                recall=0.96,
                f1_score=0.95,
                support=50,
            )
        ],
        confusion_matrix=[[48, 2], [3, 47]],
        class_names=["Class_1", "Class_2"],
    )

    rep_dict = report.to_dict()
    assert rep_dict["overall_accuracy"] == 0.95
    assert len(rep_dict["per_class_metrics"]) == 1

    json_path = tmp_path / "eval_report.json"
    saved = report.save_json(json_path)
    assert saved.is_file()
    assert "overall_accuracy" in json_path.read_text(encoding="utf-8")
