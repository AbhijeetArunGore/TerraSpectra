"""Day 14: Train the CNN–ViT Hybrid Fusion Network on the Salinas dataset."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Insert project source directory into path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from terraspectra.data import SalinasLoader, create_dataloaders
from terraspectra.models import (
    HyperspectralHybridCNNViT,
    HyperspectralHybridConfig,
    TrainingConfig,
    load_checkpoint,
    select_device,
    set_seed,
    train_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("terraspectra.day14")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Day 14: Train the CNN-ViT Hybrid Fusion Network on Salinas Hyperspectral Data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw Salinas files (Salinas_corrected.mat, Salinas_gt.mat)",
    )
    parser.add_argument(
        "--pca-file",
        type=str,
        default="data/processed/salinas_pca_reduced.npz",
        help="Path to PCA-reduced Salinas cube file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2 for smoke test)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for DataLoaders (default: 128)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for AdamW optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Decoupled weight decay for AdamW regularization (default: 1e-4)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability in fusion projection and classification head (default: 0.2)",
    )
    parser.add_argument(
        "--fusion-dim",
        type=int,
        default=64,
        help="Fused latent embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--vit-embed-dim",
        type=int,
        default=64,
        help="ViT latent token embedding dimension (default: 64)",
    )
    parser.add_argument(
        "--vit-heads",
        type=int,
        default=4,
        help="Number of multi-head self-attention heads (default: 4)",
    )
    parser.add_argument(
        "--vit-layers",
        type=int,
        default=2,
        help="Number of Transformer encoder blocks in ViT branch (default: 2)",
    )
    parser.add_argument(
        "--vit-mlp-dim",
        type=int,
        default=128,
        help="Hidden dimension of feed-forward MLP inside ViT blocks (default: 128)",
    )
    parser.add_argument(
        "--vit-pooling",
        type=str,
        default="mean",
        choices=["mean", "cls"],
        help="ViT token pooling method (default: 'mean')",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=7,
        help="Spatial window patch size (default: 7)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cpu', 'cuda') (default: 'auto')",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_hybrid_cnn_vit.pt",
        help="Target filename for best hybrid model checkpoint",
    )
    parser.add_argument(
        "--history-out",
        type=str,
        default="outputs/reports/day14_training_history.json",
        help="Path to save training history JSON report",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("TerraSpectra — Day 14: Train CNN-ViT Hybrid Fusion Network")
    print("=" * 70)

    # 1. Reproducibility & Device Setup
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"Random Seed:      {args.seed}")
    print(f"Selected Device:  {device}")

    # 2. Dataset Loading (PCA-Reduced Salinas & Ground Truth)
    pca_path = Path(args.pca_file)
    loader = SalinasLoader(raw_data_dir=args.data_dir)
    dataset = loader.load(allow_synthetic_fallback=True)
    gt = dataset.gt

    if pca_path.is_file():
        pca_data = np.load(pca_path)
        cube = pca_data["reduced_cube"]
        print(f"Loaded PCA Cube:  {pca_path} (Shape: {cube.shape})")
    else:
        print(f"[WARN] PCA file not found at {pca_path}. Using first 15 bands of raw cube.")
        cube = dataset.cube[:, :, :15]

    print(f"Ground Truth:     Shape {gt.shape}, Total labeled: {int(np.sum(gt > 0)):,}")

    # 3. Initialize Day 9 DataLoaders
    print(f"\nConstructing DataLoaders (patch_size={args.patch_size}, batch_size={args.batch_size})...")
    train_loader, val_loader, test_loader, meta = create_dataloaders(
        cube_or_dataset=cube,
        gt=gt,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
        stratify=True,
        exclude_unlabeled=True,
    )
    print(f"  Training Samples:   {meta.train_samples:,} ({len(train_loader)} batches)")
    print(f"  Validation Samples: {meta.val_samples:,} ({len(val_loader)} batches)")
    print(f"  Test Samples:       {meta.test_samples:,} ({len(test_loader)} batches)")
    print(f"  Target Classes:     {meta.num_classes} ({', '.join(str(k) for k in list(meta.label_to_idx.keys())[:5])}...)")

    # 4. Instantiate Day 13 CNN-ViT Hybrid Fusion Network
    hybrid_config = HyperspectralHybridConfig(
        in_channels=1,
        spectral_depth=cube.shape[2],
        spatial_size=args.patch_size,
        num_classes=meta.num_classes,
        cnn_conv_channels=(8, 16, 32),
        cnn_kernel_spectral=(7, 5, 3),
        vit_embed_dim=args.vit_embed_dim,
        vit_num_heads=args.vit_heads,
        vit_num_layers=args.vit_layers,
        vit_mlp_dim=args.vit_mlp_dim,
        vit_pooling=args.vit_pooling,
        fusion_dim=args.fusion_dim,
        dropout_rate=args.dropout,
    )
    model = HyperspectralHybridCNNViT(hybrid_config).to(device)

    param_info = model.count_parameters()
    print(f"\nModel Architecture: {model.__class__.__name__}")
    print(f"  Total Parameters:     {param_info['total_parameters']:,}")
    print(f"  Trainable Parameters: {param_info['trainable_parameters']:,}")
    print(f"  CNN Branch:           {param_info['component_breakdown']['cnn_branch_parameters']:,}")
    print(f"  ViT Branch:           {param_info['component_breakdown']['vit_branch_parameters']:,}")
    print(f"  Fusion Projection:    {param_info['component_breakdown']['fusion_projection_parameters']:,}")
    print(f"  Classifier Head:      {param_info['component_breakdown']['classifier_parameters']:,}")

    # 5. Training Configuration
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_config = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=str(device),
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_filename=args.checkpoint_name,
        save_best_only=True,
        monitor_metric="val_loss",
        model_type="hybrid",
        seed=args.seed,
    )

    print(f"\nHyperparameters & Regularization:")
    print(f"  Optimizer:    AdamW (lr={train_config.lr}, weight_decay={train_config.weight_decay})")
    print(f"  Dropout Rate: {hybrid_config.dropout_rate} (active in fusion projection)")
    print(f"  Loss:         CrossEntropyLoss (raw unnormalized logits)")
    print(f"  Epochs:       {train_config.epochs}")
    print(f"  Checkpoint:   {checkpoint_dir / args.checkpoint_name}")

    # 6. Execute Training Loop
    t_start = time.time()
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        metadata=meta,
        model_config=hybrid_config,
    )
    total_duration = time.time() - t_start

    print("\n" + "=" * 70)
    print("Training Complete — Summary:")
    print(f"  Total Time:          {total_duration:.2f} seconds")
    print(f"  Best Validation Epoch: {history.best_epoch}")
    print(f"  Best Val Loss:       {history.best_val_loss:.4f}")
    print(f"  Best Val Accuracy:   {history.best_val_acc * 100:.2f}%")
    print(f"  Saved Checkpoint:    {history.checkpoint_path}")

    # 7. Save Training History JSON Report
    history_out_path = Path(args.history_out)
    history_out_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "model_architecture": "HyperspectralHybridCNNViT",
        "parameters": param_info,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout_rate": args.dropout,
            "seed": args.seed,
            "device": str(device),
        },
        "history": history.to_dict(),
        "total_duration_seconds": total_duration,
    }
    with open(history_out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Training History:    {history_out_path}")

    # 8. Checkpoint Reload Verification
    print(f"\nVerifying Checkpoint Reload...")
    verify_model = HyperspectralHybridCNNViT(hybrid_config).to(device)
    payload = load_checkpoint(history.checkpoint_path, model=verify_model, device=device)
    verify_model.eval()

    sample_x, sample_y = next(iter(val_loader))
    with torch.no_grad():
        out_logits = verify_model(sample_x.to(device))
        preds = out_logits.argmax(dim=-1).cpu()

    acc = (preds == sample_y).float().mean().item() * 100.0
    print(f"  Reloaded Epoch:      {payload.get('epoch')}")
    print(f"  Batch Logits Shape:  {tuple(out_logits.shape)}")
    print(f"  Sample Batch Acc:    {acc:.2f}%")
    print("\n[SUCCESS] Day 14 CNN-ViT Hybrid training pipeline executed successfully!\n")


if __name__ == "__main__":
    main()
