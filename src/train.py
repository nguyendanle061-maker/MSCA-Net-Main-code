"""Train a model with CrossEntropyLoss."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data import create_dataloaders
from src.models import build_model
from src.utils import (
    accuracy_score_np,
    confusion_matrix_np,
    count_parameters,
    f1_scores_from_confusion,
    get_device,
    load_config,
    matthews_corrcoef_from_confusion,
    make_dummy_dataloaders,
    per_class_f1_from_confusion,
    save_checkpoint,
    save_confusion_matrix_png,
    save_history_csv,
    save_history_plots,
    save_table_csv,
    save_tsne_png,
    set_seed,
)


def build_dataloaders(config: dict) -> tuple:
    """Create real-data loaders, or dummy loaders when data files are absent."""
    data_cfg = config["data"]
    data_dir = Path(data_cfg["data_dir"])
    batch_size = int(config["training"]["batch_size"])

    input_file = str(data_cfg.get("input_file", "X.npy"))
    label_file = str(data_cfg.get("label_file", "y.npy"))
    if (data_dir / input_file).exists() and (data_dir / label_file).exists():
        return create_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=int(data_cfg.get("num_workers", 0)),
            train_ratio=float(data_cfg.get("train_ratio", 0.7)),
            val_ratio=float(data_cfg.get("val_ratio", 0.15)),
            seed=int(config.get("seed", 42)),
            augment=bool(data_cfg.get("augment", True)),
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

    raise FileNotFoundError(
        f"Z24 arrays not found in {data_dir}. Obtain authorized data and prepare "
        "inputs.npy and labels.npy as documented in README.md."
    )


def evaluate_loader(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, object]:
    """Evaluate loss, accuracy, macro-F1, and weighted-F1 on a loader."""
    model.eval()
    total_loss = 0.0
    total_items = 0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            if x.ndim == 4:
                batch_size, num_crops, seq_len, channels = x.shape
                logits = model(x.reshape(batch_size * num_crops, seq_len, channels))
                logits = logits.reshape(batch_size, num_crops, -1).mean(dim=1)
            else:
                logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            total_items += x.size(0)
            y_true.append(y.cpu().numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())

    true = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    confusion = confusion_matrix_np(true, pred, num_classes)
    macro_f1, weighted_f1 = f1_scores_from_confusion(confusion)
    return {
        "loss": total_loss / max(1, total_items),
        "accuracy": accuracy_score_np(true, pred),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": matthews_corrcoef_from_confusion(confusion),
        "confusion_matrix": confusion,
    }


def _load_history_csv(path: str | Path) -> list[dict[str, float | int]]:
    """Load saved history rows, preserving numeric values."""
    history_path = Path(path)
    if not history_path.exists():
        return []
    rows: list[dict[str, float | int]] = []
    with history_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float | int] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                parsed[key] = int(value) if key in {"epoch", "parameter_count", "is_best"} else float(value)
            rows.append(parsed)
    return rows


def _restore_scheduler_from_history(
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    scheduler_name: str,
    history: list[dict[str, float | int]],
) -> None:
    """Replay scheduler steps when an old checkpoint lacks scheduler state."""
    if scheduler is None:
        return
    for row in history:
        if scheduler_name == "reduce_on_plateau":
            scheduler.step(float(row.get("val_macro_f1", 0.0)))
        else:
            scheduler.step()


def extract_train_features(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_points: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate-layer features from deterministic train samples."""
    final_linear = next(
        (module for module in reversed(list(model.modules())) if isinstance(module, nn.Linear)),
        None,
    )
    if final_linear is None:
        raise ValueError("Cannot extract train features because the model has no Linear classifier")

    captured: list[torch.Tensor] = []

    def capture_input(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach())

    dataset = loader.dataset
    original_augment = getattr(dataset, "augment", None)
    original_crop_mode = getattr(dataset, "crop_mode", None)
    if original_augment is not None:
        dataset.augment = False
    if original_crop_mode == "random_train_center_eval":
        dataset.crop_mode = "center"

    was_training = model.training
    model.eval()
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    collected = 0
    handle = final_linear.register_forward_pre_hook(capture_input)
    try:
        with torch.no_grad():
            for x, y in loader:
                captured.clear()
                x = x.to(device)
                if x.ndim == 4:
                    batch_size, num_crops, seq_len, channels = x.shape
                    model(x.reshape(batch_size * num_crops, seq_len, channels))
                    batch_features = captured[-1].reshape(batch_size, num_crops, -1).mean(dim=1)
                else:
                    model(x)
                    batch_features = captured[-1].reshape(x.shape[0], -1)
                remaining = max_points - collected
                if remaining <= 0:
                    break
                batch_features = batch_features[:remaining]
                batch_labels = y[:remaining]
                features.append(batch_features.cpu().numpy())
                labels.append(batch_labels.cpu().numpy())
                collected += batch_features.shape[0]
                if collected >= max_points:
                    break
    finally:
        handle.remove()
        model.train(was_training)
        if original_augment is not None:
            dataset.augment = original_augment
        if original_crop_mode is not None:
            dataset.crop_mode = original_crop_mode

    if not features:
        raise ValueError("No train features were extracted")
    return np.concatenate(features), np.concatenate(labels)


def save_train_tsne_snapshot(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str | Path,
    epoch: int,
    max_points: int = 2000,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Save one train-set t-SNE snapshot and append its silhouette metrics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    features, labels = extract_train_features(model, train_loader, device, max_points=max_points)
    metrics = save_tsne_png(
        output_path / f"tsne_epoch_{epoch:03d}.png",
        features,
        labels,
        max_points=max_points,
        seed=seed,
        title=f"Train-set t-SNE visualization (epoch {epoch})",
    )
    metrics_row: dict[str, float | int | None] = {"epoch": epoch, **metrics}
    metrics_path = output_path / "tsne_silhouette_log.csv"
    existing_rows: list[dict[str, object]] = []
    if metrics_path.exists():
        with metrics_path.open("r", newline="", encoding="utf-8") as f:
            existing_rows = [dict(row) for row in csv.DictReader(f) if int(row["epoch"]) != epoch]
    save_table_csv(metrics_path, [*existing_rows, metrics_row])
    return metrics


def should_save_train_tsne(epoch: int, interval: int) -> bool:
    """Return whether a train-set t-SNE snapshot is due for this epoch."""
    return interval > 0 and (epoch == 1 or epoch % interval == 0)


def train(config_path: str, resume_checkpoint_path: str | None = None) -> list[dict[str, float | int]]:
    """Run model training."""
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))
    device = get_device(str(config["training"].get("device", "auto")))
    train_loader, val_loader, _ = build_dataloaders(config)

    model = build_model(config["model"]).to(device)
    resume_checkpoint = None
    if resume_checkpoint_path:
        resume_path = Path(resume_checkpoint_path)
        if resume_path.exists():
            resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(resume_checkpoint["model_state_dict"])
            print(f"resuming_from={resume_path}")
        else:
            print(f"resume checkpoint not found: {resume_path}. Starting from scratch.")
    parameter_count = count_parameters(model)
    print(f"parameter_count={parameter_count}")

    training_cfg = config["training"]
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training_cfg.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
    scheduler = None
    if scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(training_cfg.get("scheduler_factor", 0.5)),
            patience=int(training_cfg.get("scheduler_patience", 4)),
            min_lr=float(training_cfg.get("min_learning_rate", 1e-5)),
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(training_cfg.get("epochs", 10)),
            eta_min=float(training_cfg.get("min_learning_rate", 1e-5)),
        )
    best_checkpoint_path = training_cfg.get(
        "best_checkpoint_path",
        training_cfg.get("checkpoint_path", "outputs/best_checkpoint.pt"),
    )
    last_checkpoint_path = training_cfg.get("last_checkpoint_path", "outputs/last_checkpoint.pt")
    history_path = training_cfg.get("history_path", "outputs/history.csv")
    history_plot_path = training_cfg.get("history_plot_path", "outputs/history.png")
    val_confusion_npy_path = training_cfg.get("val_confusion_matrix_npy_path", "outputs/val_confusion_matrix.npy")
    val_confusion_png_path = training_cfg.get("val_confusion_matrix_png_path", "outputs/val_confusion_matrix.png")
    val_f1_scores_path = training_cfg.get("val_f1_scores_path", "outputs/val_f1_scores.csv")
    train_tsne_interval = int(training_cfg.get("train_tsne_interval", 20))
    train_tsne_max_points = int(training_cfg.get("train_tsne_max_points", 2000))
    train_tsne_dir = Path(training_cfg.get("train_tsne_dir", Path(history_path).parent / "train_tsne"))
    training_log_path = Path(training_cfg.get("training_log_path", "outputs/training_log.txt"))
    training_log_path.parent.mkdir(parents=True, exist_ok=True)
    if resume_checkpoint is None:
        training_log_path.write_text("", encoding="utf-8")
    patience = int(training_cfg.get("early_stopping_patience", 0))
    min_delta = float(training_cfg.get("early_stopping_min_delta", 0.0))
    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    total_epochs = int(config["training"].get("epochs", 10))
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    if resume_checkpoint is not None:
        history = _load_history_csv(history_path)
        checkpoint_epoch = int(resume_checkpoint.get("epoch", 0))
        start_epoch = checkpoint_epoch + 1
        completed_history = [row for row in history if int(row.get("epoch", 0)) <= checkpoint_epoch]
        history = completed_history
        if history:
            best_val_macro_f1 = max(float(row.get("val_macro_f1", -1.0)) for row in history)
            last_best_index = max(
                (index for index, row in enumerate(history) if int(row.get("is_best", 0)) == 1),
                default=-1,
            )
            epochs_without_improvement = len(history) - last_best_index - 1 if last_best_index >= 0 else 0
        else:
            val_metrics = resume_checkpoint.get("val_metrics", {})
            best_val_macro_f1 = float(resume_checkpoint.get("best_val_macro_f1", val_metrics.get("macro_f1", -1.0)))
            epochs_without_improvement = int(resume_checkpoint.get("epochs_without_improvement", 0))
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if scheduler is not None and resume_checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        elif scheduler is not None:
            _restore_scheduler_from_history(scheduler, scheduler_name, history)
        print(
            f"resume_start_epoch={start_epoch} best_val_macro_f1={best_val_macro_f1:.4f} "
            f"epochs_without_improvement={epochs_without_improvement}"
        )

    if start_epoch > total_epochs:
        print(f"checkpoint already reached total_epochs={total_epochs}; nothing to train.")
        return history

    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_items = 0
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            gradient_clip_norm = float(training_cfg.get("gradient_clip_norm", 0.0))
            if gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_items += x.size(0)
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

        train_loss = total_loss / max(1, total_items)
        train_true_np = np.concatenate(train_true) if train_true else np.array([], dtype=np.int64)
        train_pred_np = np.concatenate(train_pred) if train_pred else np.array([], dtype=np.int64)
        train_confusion = confusion_matrix_np(train_true_np, train_pred_np, int(config["model"]["num_classes"]))
        train_macro_f1, train_weighted_f1 = f1_scores_from_confusion(train_confusion)
        train_acc = accuracy_score_np(train_true_np, train_pred_np)
        val_metrics = evaluate_loader(
            model,
            val_loader,
            criterion,
            device,
            int(config["model"]["num_classes"]),
        )
        val_loss = float(val_metrics["loss"])
        val_accuracy = float(val_metrics["accuracy"])
        val_macro_f1 = float(val_metrics["macro_f1"])
        val_weighted_f1 = float(val_metrics["weighted_f1"])
        val_mcc = float(val_metrics["mcc"])
        val_confusion = val_metrics["confusion_matrix"]
        if not isinstance(val_confusion, np.ndarray):
            raise TypeError("Validation confusion matrix must be a NumPy array")
        improved = val_macro_f1 > best_val_macro_f1 + min_delta
        if improved:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_macro_f1,
            "train_weighted_f1": train_weighted_f1,
            "train_mcc": matthews_corrcoef_from_confusion(train_confusion),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "val_mcc": val_mcc,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
            "parameter_count": parameter_count,
            "is_best": int(improved),
        }
        history.append(row)
        save_history_csv(history_path, history)
        save_history_plots(history_plot_path, history)
        checkpoint_val_metrics = {
            "loss": val_loss,
            "accuracy": val_accuracy,
            "macro_f1": val_macro_f1,
            "weighted_f1": val_weighted_f1,
            "mcc": val_mcc,
        }
        save_checkpoint(
            last_checkpoint_path,
            model,
            config,
            extra={
                "epoch": epoch,
                "parameter_count": parameter_count,
                "val_metrics": checkpoint_val_metrics,
                "is_best": improved,
                "best_val_macro_f1": best_val_macro_f1,
                "epochs_without_improvement": epochs_without_improvement,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            },
        )
        if improved:
            Path(val_confusion_npy_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(val_confusion_npy_path, val_confusion)
            save_confusion_matrix_png(val_confusion_png_path, val_confusion)
            save_table_csv(val_f1_scores_path, per_class_f1_from_confusion(val_confusion))
            save_checkpoint(
                best_checkpoint_path,
                model,
                config,
                extra={
                    "epoch": epoch,
                    "parameter_count": parameter_count,
                    "val_metrics": checkpoint_val_metrics,
                    "is_best": True,
                    "best_val_macro_f1": best_val_macro_f1,
                    "epochs_without_improvement": epochs_without_improvement,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                },
            )

        if scheduler is not None:
            if scheduler_name == "reduce_on_plateau":
                scheduler.step(val_macro_f1)
            else:
                scheduler.step()

        best_marker = " *" if improved else ""
        log_line = (
            f"Epoch {epoch:3d}/{total_epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_accuracy:.4f}{best_marker} | "
            f"val_f1: {val_macro_f1:.4f} | "
            f"val_mcc: {val_mcc:.4f} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
            f"{row['epoch_seconds']:.1f}s"
        )
        print(log_line)
        with training_log_path.open("a", encoding="utf-8") as f:
            f.write(log_line + "\n")
        if should_save_train_tsne(epoch, train_tsne_interval):
            try:
                tsne_metrics = save_train_tsne_snapshot(
                    model,
                    train_loader,
                    device,
                    train_tsne_dir,
                    epoch,
                    max_points=train_tsne_max_points,
                    seed=int(config.get("seed", 42)),
                )
                tsne_line = (
                    f"train_tsne epoch={epoch} "
                    f"silhouette_feature_space={tsne_metrics['silhouette_feature_space']} "
                    f"silhouette_tsne_3d={tsne_metrics['silhouette_tsne_3d']}"
                )
                print(tsne_line)
                with training_log_path.open("a", encoding="utf-8") as f:
                    f.write(tsne_line + "\n")
            except Exception as exc:
                error_line = f"train_tsne_failed epoch={epoch}: {type(exc).__name__}: {exc}"
                print(error_line)
                with training_log_path.open("a", encoding="utf-8") as f:
                    f.write(error_line + "\n")
        if patience > 0 and epochs_without_improvement >= patience:
            stop_line = f"early_stopping epoch={epoch} best_val_macro_f1={best_val_macro_f1:.4f}"
            print(stop_line)
            with training_log_path.open("a", encoding="utf-8") as f:
                f.write(stop_line + "\n")
            break
    return history


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/z24.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from the config's last_checkpoint_path.")
    parser.add_argument("--resume-checkpoint", default=None, help="Checkpoint path to resume from.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resume_path = args.resume_checkpoint
    if args.resume and resume_path is None:
        resume_path = load_config(args.config)["training"].get("last_checkpoint_path")
    train(args.config, resume_path)
