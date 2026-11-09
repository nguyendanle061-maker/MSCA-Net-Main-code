"""Evaluate classification metrics for a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data import create_dataloaders
from src.models import build_model
from src.utils import (
    accuracy_score_np,
    compute_multiclass_auc,
    confusion_matrix_np,
    count_parameters,
    f1_scores_from_confusion,
    get_device,
    load_config,
    make_dummy_dataloaders,
    per_class_f1_from_confusion,
    save_confusion_matrix_png,
    save_json,
    save_roc_curve_png,
    save_table_csv,
    save_tsne_png,
)


def load_model(config: dict, checkpoint_path: str | Path | None, device: torch.device) -> torch.nn.Module:
    """Build a model and optionally restore checkpoint weights."""
    model = build_model(config["model"]).to(device)
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    return model


def build_test_loader(config: dict) -> torch.utils.data.DataLoader:
    """Create the test DataLoader from real or dummy data."""
    data_cfg = config["data"]
    data_dir = Path(data_cfg["data_dir"])
    batch_size = int(config["training"]["batch_size"])
    input_file = str(data_cfg.get("input_file", "X.npy"))
    label_file = str(data_cfg.get("label_file", "y.npy"))
    if (data_dir / input_file).exists() and (data_dir / label_file).exists():
        _, _, test_loader = create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=int(data_cfg.get("num_workers", 0)),
            train_ratio=float(data_cfg.get("train_ratio", 0.7)),
            val_ratio=float(data_cfg.get("val_ratio", 0.15)),
            seed=int(config.get("seed", 42)),
            augment=False,
            jitter_std=float(data_cfg.get("jitter_std", 0.01)),
            scaling_std=float(data_cfg.get("scaling_std", 0.1)),
            time_mask_ratio=float(data_cfg.get("time_mask_ratio", 0.05)),
            channel_mask_ratio=float(data_cfg.get("channel_mask_ratio", 0.0)),
            input_file=input_file,
            label_file=label_file,
            input_layout=str(data_cfg.get("input_layout", "ntc")),
            normalization=str(data_cfg.get("normalization", "train")),
            window_length=data_cfg.get("window_length"),
            crop_mode=str(data_cfg.get("crop_mode", "none")),
            temporal_stride=int(data_cfg.get("temporal_stride", 1)),
            transform=str(data_cfg.get("transform", "raw")),
            eval_num_crops=int(data_cfg.get("eval_num_crops", 1)),
            window_mode=str(data_cfg.get("window_mode", "crop")),
            hop_length=data_cfg.get("hop_length"),
            taper=str(data_cfg.get("taper", "none")),
            bandpass_filter=bool(data_cfg.get("bandpass_filter", False)),
            sampling_rate=float(data_cfg.get("sampling_rate", 100.0)),
            lowcut=float(data_cfg.get("lowcut", 0.5)),
            highcut=float(data_cfg.get("highcut", 40.0)),
            filter_order=int(data_cfg.get("filter_order", 4)),
            split_strategy=str(data_cfg.get("split_strategy", "stratified")),
            group_file=str(data_cfg.get("group_file", "groups.npy")),
            holdout_group=data_cfg.get("holdout_group"),
        )
        return test_loader

    print(f"Data not found in {data_dir}. Using dummy data.")
    _, _, test_loader = make_dummy_dataloaders(
        num_samples=int(data_cfg.get("dummy_samples", 128)),
        seq_len=int(data_cfg["seq_len"]),
        num_channels=int(config["model"]["num_channels"]),
        num_classes=int(config["model"]["num_classes"]),
        batch_size=batch_size,
    )
    return test_loader


def evaluate(config_path: str, checkpoint_path: str | None = None) -> dict[str, object]:
    """Evaluate accuracy, macro-F1, weighted-F1, and confusion matrix."""
    config = load_config(config_path)
    device = get_device(str(config["training"].get("device", "auto")))
    training_cfg = config["training"]
    checkpoint_path = checkpoint_path or training_cfg.get(
        "best_checkpoint_path",
        training_cfg.get("checkpoint_path"),
    )
    model = load_model(config, checkpoint_path, device)
    parameter_count = count_parameters(model)
    test_loader = build_test_loader(config)

    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    y_logits: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            if x.ndim == 4:
                batch_size, num_crops, seq_len, channels = x.shape
                logits = model(x.reshape(batch_size * num_crops, seq_len, channels))
                logits = logits.reshape(batch_size, num_crops, -1).mean(dim=1)
            else:
                logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred.append(preds)
            y_prob.append(probabilities)
            y_logits.append(logits.cpu().numpy())
            y_true.append(y.numpy())

    true = np.concatenate(y_true)
    pred = np.concatenate(y_pred)
    probabilities = np.concatenate(y_prob)
    logits_np = np.concatenate(y_logits)
    num_classes = int(config["model"]["num_classes"])
    confusion = confusion_matrix_np(true, pred, num_classes)
    macro_f1, weighted_f1 = f1_scores_from_confusion(confusion)
    auc_metrics = compute_multiclass_auc(true, probabilities, num_classes)
    metrics = {
        "accuracy": accuracy_score_np(true, pred),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        **auc_metrics,
        "parameter_count": parameter_count,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "confusion_matrix": confusion,
    }
    metrics_path = training_cfg.get("metrics_path", "outputs/metrics.json")
    confusion_npy_path = training_cfg.get("confusion_matrix_npy_path", "outputs/confusion_matrix.npy")
    confusion_png_path = training_cfg.get("confusion_matrix_png_path", "outputs/confusion_matrix.png")
    f1_scores_path = training_cfg.get("f1_scores_path", "outputs/f1_scores.csv")
    roc_curve_path = training_cfg.get("roc_curve_path", "outputs/roc_curve.png")
    tsne_path = training_cfg.get("tsne_path", "outputs/tsne.png")
    Path(confusion_npy_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(confusion_npy_path, confusion)
    save_confusion_matrix_png(confusion_png_path, confusion)
    save_table_csv(f1_scores_path, per_class_f1_from_confusion(confusion))
    save_roc_curve_png(roc_curve_path, true, probabilities, num_classes)
    save_tsne_png(tsne_path, logits_np, true, seed=int(config.get("seed", 42)))
    save_json(
        metrics_path,
        {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "roc_auc_macro": metrics["roc_auc_macro"],
            "roc_auc_micro": metrics["roc_auc_micro"],
            "roc_auc_weighted": metrics["roc_auc_weighted"],
            "parameter_count": metrics["parameter_count"],
            "checkpoint_path": metrics["checkpoint_path"],
            "confusion_matrix_npy_path": confusion_npy_path,
            "confusion_matrix_png_path": confusion_png_path,
            "f1_scores_path": f1_scores_path,
            "roc_curve_path": roc_curve_path,
            "tsne_path": tsne_path,
        },
    )
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"weighted_f1={metrics['weighted_f1']:.4f}")
    print(f"roc_auc_macro={metrics['roc_auc_macro']}")
    print(f"roc_auc_micro={metrics['roc_auc_micro']}")
    print(f"parameter_count={parameter_count}")
    print("confusion_matrix=")
    print(confusion)
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/z24.yaml")
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config, args.checkpoint)
