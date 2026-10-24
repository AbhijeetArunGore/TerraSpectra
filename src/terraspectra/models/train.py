"""Model training, validation, metric aggregation, and checkpointing pipeline."""

from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from terraspectra.data.dataset import DataLoaderMetadata
from terraspectra.models.cnn3d import Hyperspectral3DCNN, Hyperspectral3DCNNConfig
from terraspectra.models.hybrid import HyperspectralHybridCNNViT, HyperspectralHybridConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Hyperparameters and execution configuration for 3D-CNN or Hybrid model training."""
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_filename: str = "best_3d_cnn.pt"
    save_best_only: bool = True
    monitor_metric: str = "val_loss"  # 'val_loss' (min) or 'val_acc' (max)
    model_type: str = "3d_cnn"  # '3d_cnn' or 'hybrid'

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay cannot be negative, got {self.weight_decay}")
        if self.monitor_metric not in ("val_loss", "val_acc"):
            raise ValueError(f"monitor_metric must be 'val_loss' or 'val_acc', got {self.monitor_metric}")
        valid_types = ("3d_cnn", "cnn3d", "cnn", "hybrid", "hybrid_cnn_vit", "cnn_vit")
        if self.model_type.lower() not in valid_types:
            raise ValueError(f"model_type must be one of {valid_types}, got '{self.model_type}'")


@dataclass
class EpochMetrics:
    """Metrics recorded for a single training/validation epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    duration_seconds: float
    lr: float


@dataclass
class TrainingHistory:
    """Container tracking epoch-by-epoch training and validation progress."""
    epochs: List[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    checkpoint_path: Optional[str] = None

    def add_epoch(self, metrics: EpochMetrics) -> None:
        self.epochs.append(metrics)

    def get_metric_series(self, metric_name: str) -> List[float]:
        if not hasattr(EpochMetrics, metric_name):
            raise AttributeError(f"Unknown metric '{metric_name}' in EpochMetrics")
        return [getattr(ep, metric_name) for ep in self.epochs]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epochs": [asdict(ep) for ep in self.epochs],
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": self.best_val_acc,
            "checkpoint_path": self.checkpoint_path,
        }


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch for deterministic execution.

    Parameters
    ----------
    seed : int
        Seed value for pseudo-random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Configure deterministic algorithms where practical
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_str: Optional[str] = None) -> torch.device:
    """
    Select execution device supporting CPU default and optional CUDA acceleration.

    Parameters
    ----------
    device_str : Optional[str]
        'auto', 'cuda', 'cpu', or specific device like 'cuda:0'. If None or 'auto',
        chooses CUDA when available, otherwise falls back to CPU.

    Returns
    -------
    torch.device
        Selected PyTorch hardware device.
    """
    if device_str is None or device_str.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = device_str.lower()
    if target.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(target)


def build_model_for_training(
    model_type: str = "3d_cnn",
    spectral_depth: int = 15,
    spatial_size: int = 7,
    num_classes: int = 16,
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function to construct either the baseline 3D-CNN or the CNN-ViT Hybrid model.

    Parameters
    ----------
    model_type : str
        '3d_cnn' or 'hybrid' (case-insensitive).
    spectral_depth : int
        Number of spectral bands / components (default: 15).
    spatial_size : int
        Spatial window height and width (default: 7).
    num_classes : int
        Number of output classification classes (default: 16).
    **kwargs : Any
        Additional keyword arguments forwarded to the respective configuration.

    Returns
    -------
    nn.Module
        Instantiated model (Hyperspectral3DCNN or HyperspectralHybridCNNViT).
    """
    m_type = model_type.lower().strip()
    if m_type in ("3d_cnn", "cnn3d", "cnn"):
        cfg_kwargs = {
            "spectral_depth": spectral_depth,
            "spatial_size": spatial_size,
            "num_classes": num_classes,
        }
        cfg_kwargs.update(kwargs)
        cfg = Hyperspectral3DCNNConfig(**cfg_kwargs)
        return Hyperspectral3DCNN(cfg)
    elif m_type in ("hybrid", "hybrid_cnn_vit", "cnn_vit"):
        cfg_kwargs = {
            "spectral_depth": spectral_depth,
            "spatial_size": spatial_size,
            "num_classes": num_classes,
        }
        cfg_kwargs.update(kwargs)
        cfg = HyperspectralHybridConfig(**cfg_kwargs)
        return HyperspectralHybridCNNViT(cfg)
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Supported types: '3d_cnn', 'hybrid'."
        )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Execute one training epoch with gradient updates.

    Parameters
    ----------
    model : nn.Module
        Hyperspectral3DCNN model.
    dataloader : DataLoader
        Training DataLoader yielding (batch_x, batch_y).
    criterion : nn.Module
        Loss function (e.g., nn.CrossEntropyLoss).
    optimizer : torch.optim.Optimizer
        Parameter optimizer (e.g., AdamW).
    device : torch.device
        Device on which tensors and model reside.

    Returns
    -------
    avg_loss : float
        Mean cross-entropy loss across all training samples.
    accuracy : float
        Fraction of correctly classified samples [0.0, 1.0].
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        current_batch_size = batch_x.size(0)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        # Weighted loss accumulation accounting for incomplete final batch
        running_loss += loss.item() * current_batch_size
        preds = logits.argmax(dim=-1)
        running_correct += (preds == batch_y).sum().item()
        total_samples += current_batch_size

    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = running_loss / total_samples
    accuracy = running_correct / total_samples
    return avg_loss, accuracy


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate model performance on validation data without gradient tracking.

    Parameters
    ----------
    model : nn.Module
        Hyperspectral3DCNN model.
    dataloader : DataLoader
        Validation DataLoader.
    criterion : nn.Module
        Loss function.
    device : torch.device
        Execution device.

    Returns
    -------
    avg_loss : float
        Validation mean loss.
    accuracy : float
        Validation classification accuracy.
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            current_batch_size = batch_x.size(0)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            running_loss += loss.item() * current_batch_size
            preds = logits.argmax(dim=-1)
            running_correct += (preds == batch_y).sum().item()
            total_samples += current_batch_size

    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = running_loss / total_samples
    accuracy = running_correct / total_samples
    return avg_loss, accuracy


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    val_loss: float = 0.0,
    val_acc: float = 0.0,
    metadata: Optional[DataLoaderMetadata] = None,
    config: Optional[Union[Hyperspectral3DCNNConfig, HyperspectralHybridConfig, Dict[str, Any], Any]] = None,
) -> Path:
    """
    Atomically save a complete, reproducible model checkpoint to disk.

    Parameters
    ----------
    checkpoint_path : Union[str, Path]
        Target destination path for .pt file.
    model : nn.Module
        Model whose state_dict is saved.
    optimizer : Optional[torch.optim.Optimizer]
        Optimizer whose state_dict is saved.
    epoch : int
        Current epoch number.
    val_loss : float
        Validation loss associated with this checkpoint.
    val_acc : float
        Validation accuracy associated with this checkpoint.
    metadata : Optional[DataLoaderMetadata]
        Metadata detailing class mappings, splits, and spectral dimensions.
    config : Optional[Hyperspectral3DCNNConfig | HyperspectralHybridConfig | Dict]
        Model architecture configuration needed to reconstruct model.

    Returns
    -------
    Path
        Canonical absolute path to the saved checkpoint.
    """
    target = Path(checkpoint_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if is_dataclass(config) and not isinstance(config, type):
        config_dict = asdict(config)
    elif isinstance(config, dict):
        config_dict = config
    else:
        config_dict = {}

    checkpoint_payload: Dict[str, Any] = {
        "model_architecture": model.__class__.__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "model_config": config_dict,
        "timestamp": time.time(),
    }

    if metadata is not None:
        checkpoint_payload["label_to_idx"] = metadata.label_to_idx
        checkpoint_payload["idx_to_label"] = metadata.idx_to_label
        checkpoint_payload["num_classes"] = metadata.num_classes
        checkpoint_payload["spectral_depth"] = metadata.spectral_depth
        checkpoint_payload["patch_size"] = metadata.patch_size

    # Atomic write pattern: write to temporary file first, then rename
    temp_target = target.with_suffix(".tmp")
    torch.save(checkpoint_payload, temp_target)
    temp_target.replace(target)

    logger.info(f"Saved checkpoint to {target} (epoch={epoch}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")
    return target


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """
    Load weights and metadata from a saved checkpoint into model and optimizer.

    Parameters
    ----------
    checkpoint_path : Union[str, Path]
        Path to existing checkpoint file.
    model : nn.Module
        Target model instance into which weights are loaded.
    optimizer : Optional[torch.optim.Optimizer]
        Optional optimizer into which state is loaded.
    device : Union[str, torch.device]
        Target device for tensors.

    Returns
    -------
    Dict[str, Any]
        Checkpoint metadata dictionary (epoch, metrics, config, label mappings).
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at '{path}'")

    dev = select_device(str(device)) if isinstance(device, str) else device
    payload = torch.load(path, map_location=dev, weights_only=False)

    model.load_state_dict(payload["model_state_dict"])
    model.to(dev)

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    logger.info(f"Loaded checkpoint from '{path}' (epoch={payload.get('epoch', 0)})")
    return payload


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None,
    metadata: Optional[DataLoaderMetadata] = None,
    model_config: Optional[Union[Hyperspectral3DCNNConfig, HyperspectralHybridConfig, Any]] = None,
) -> Tuple[nn.Module, TrainingHistory]:
    """
    Orchestrate full model training across epochs with validation and checkpointing.

    Supports both Hyperspectral3DCNN and HyperspectralHybridCNNViT models.

    Parameters
    ----------
    model : nn.Module
        Hyperspectral3DCNN or HyperspectralHybridCNNViT model instance.
    train_loader : DataLoader
        DataLoader providing training batches (shuffled).
    val_loader : DataLoader
        DataLoader providing validation batches (deterministic).
    config : Optional[TrainingConfig]
        Training hyperparameters and configuration.
    metadata : Optional[DataLoaderMetadata]
        Metadata detailing dataset splits and class mappings.
    model_config : Optional[Hyperspectral3DCNNConfig | HyperspectralHybridConfig | Any]
        Model architecture configuration stored in checkpoint.

    Returns
    -------
    model : nn.Module
        Trained model (loaded with best checkpoint weights).
    history : TrainingHistory
        Epoch-by-epoch training and validation loss and accuracy metrics.
    """
    if config is None:
        config = TrainingConfig()

    set_seed(config.seed)
    device = select_device(config.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history = TrainingHistory()
    best_criterion_val = float("inf") if config.monitor_metric == "val_loss" else -float("inf")
    checkpoint_full_path = Path(config.checkpoint_dir) / config.checkpoint_filename

    logger.info(
        f"Starting training: {config.epochs} epochs, lr={config.lr}, "
        f"device={device}, monitor={config.monitor_metric}"
    )

    for epoch in range(1, config.epochs + 1):
        t_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        duration = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            duration_seconds=duration,
            lr=current_lr,
        )
        history.add_epoch(metrics)

        # Check if current epoch improves the monitored validation metric
        is_best = False
        if config.monitor_metric == "val_loss":
            if val_loss < best_criterion_val:
                best_criterion_val = val_loss
                is_best = True
        else:  # val_acc
            if val_acc > best_criterion_val:
                best_criterion_val = val_acc
                is_best = True

        best_flag = " *" if is_best else ""
        logger.info(
            f"Epoch {epoch:02d}/{config.epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}% | "
            f"Time: {duration:.2f}s{best_flag}"
        )

        if is_best:
            history.best_epoch = epoch
            history.best_val_loss = val_loss
            history.best_val_acc = val_acc
            saved_path = save_checkpoint(
                checkpoint_path=checkpoint_full_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                metadata=metadata,
                config=model_config or getattr(model, "config", None),
            )
            history.checkpoint_path = str(saved_path)

    # Reload best weights into model before returning
    if history.checkpoint_path and Path(history.checkpoint_path).is_file():
        load_checkpoint(history.checkpoint_path, model=model, device=device)

    return model, history
