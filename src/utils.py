"""Shared training and evaluation utilities."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Resolve the torch device from a config string."""
    if preferred == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preferred)


def make_dummy_dataloaders(
    num_samples: int = 128,
    seq_len: int = 256,
    num_channels: int = 3,
    num_classes: int = 4,
    batch_size: int = 16,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Create dummy [B, T, C] accelerometer DataLoaders for smoke tests."""
    x = torch.randn(num_samples, seq_len, num_channels)
    y = torch.randint(0, num_classes, (num_samples,))
    train_end = int(0.7 * num_samples)
    val_end = int(0.85 * num_samples)

    datasets = (
        TensorDataset(x[:train_end], y[:train_end]),
        TensorDataset(x[train_end:val_end], y[train_end:val_end]),
        TensorDataset(x[val_end:], y[val_end:]),
    )
    return (
        DataLoader(datasets[0], batch_size=batch_size, shuffle=True),
        DataLoader(datasets[1], batch_size=batch_size, shuffle=False),
        DataLoader(datasets[2], batch_size=batch_size, shuffle=False),
    )


def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy."""
    return float((y_true == y_pred).mean()) if y_true.size else 0.0


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute a confusion matrix with rows=true labels and columns=predictions."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def f1_scores_from_confusion(confusion: np.ndarray) -> tuple[float, float]:
    """Compute macro-F1 and weighted-F1 from a confusion matrix."""
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    support = confusion.sum(axis=1).astype(np.float64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    macro_f1 = float(f1.mean()) if f1.size else 0.0
    weighted_f1 = float((f1 * support).sum() / support.sum()) if support.sum() > 0 else 0.0
    return macro_f1, weighted_f1


def per_class_f1_from_confusion(confusion: np.ndarray) -> list[dict[str, float | int]]:
    """Compute precision, recall, F1, and support for every class."""
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    support = confusion.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    return [
        {
            "class_id": int(class_id),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
            "f1_score": float(f1[class_id]),
            "support": int(support[class_id]),
        }
        for class_id in range(confusion.shape[0])
    ]


def matthews_corrcoef_from_confusion(confusion: np.ndarray) -> float:
    """Compute multiclass Matthews correlation coefficient from a confusion matrix."""
    total = confusion.sum()
    if total == 0:
        return 0.0
    true_counts = confusion.sum(axis=1).astype(np.float64)
    pred_counts = confusion.sum(axis=0).astype(np.float64)
    correct = np.trace(confusion).astype(np.float64)
    numerator = correct * total - np.dot(true_counts, pred_counts)
    denominator_left = total**2 - np.dot(pred_counts, pred_counts)
    denominator_right = total**2 - np.dot(true_counts, true_counts)
    denominator = np.sqrt(denominator_left * denominator_right)
    return float(numerator / denominator) if denominator > 0 else 0.0


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_history_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save per-epoch training history to CSV."""
    if not rows:
        return
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_table_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save a list of dictionaries to CSV."""
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Save a JSON artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# Shared matplotlib style for SCIE-grade publication figures (modern aesthetic:
# light panel background, bold dark spines, dashed grid, minor ticks).
_PAPER_STYLE: dict[str, Any] = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.5,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "axes.axisbelow": True,
    "axes.grid": True,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "grid.color": "#d0d0d0",
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.7,
    "legend.fontsize": 9.5,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "legend.fancybox": True,
    "legend.shadow": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "mathtext.default": "regular",
    "axes.unicode_minus": False,
}

# Distinct, colour-blind-friendly accents for train/validation curves.
_TRAIN_COLOR = "#0072B2"
_VAL_COLOR = "#D55E00"
_BEST_COLOR = "#009E73"


def _resolve_class_names(class_names: list[str] | None, count: int) -> list[str]:
    """Return string labels for each class, defaulting to integer indices."""
    if class_names is not None:
        return [str(name) for name in class_names]
    return [str(index) for index in range(count)]


def save_confusion_matrix_png(
    path: str | Path,
    confusion: np.ndarray,
    class_names: list[str] | None = None,
    normalize: bool = True,
    title: str = "Confusion matrix",
) -> None:
    """Save an annotated confusion-matrix heatmap as a PNG image.

    Cells are coloured by per-true-class proportion (so the diagonal stays
    legible regardless of class support) while the annotation shows the raw
    count. Set ``normalize=False`` to colour by raw counts instead.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    confusion = np.asarray(confusion)
    num_classes = confusion.shape[0]
    names = _resolve_class_names(class_names, num_classes)

    row_sums = confusion.sum(axis=1, keepdims=True)
    normalized = np.divide(
        confusion, row_sums, out=np.zeros(confusion.shape, dtype=float), where=row_sums != 0
    )
    color_data = normalized if normalize else confusion.astype(float)
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total > 0 else 0.0

    with plt.rc_context(_PAPER_STYLE):
        size = max(5.5, min(15.0, 2.2 + num_classes * 0.55))
        fig, ax = plt.subplots(figsize=(size, size * 0.86), constrained_layout=True)
        norm = Normalize(vmin=0.0, vmax=1.0) if normalize else None
        image = ax.imshow(color_data, interpolation="nearest", cmap="Blues", norm=norm)

        ax.set_title(f"{title}  (overall accuracy = {accuracy * 100:.1f}%, N = {total})")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        rotation = 45 if max(len(name) for name in names) > 2 else 0
        ax.set_xticklabels(names, rotation=rotation, ha="right" if rotation else "center")
        ax.set_yticklabels(names)

        ax.grid(False, which="major")
        ax.set_xticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_classes, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=2)

        fontsize = max(5.0, min(9.0, 120.0 / num_classes))
        threshold = 0.55 if normalize else (confusion.max() / 2.0 if confusion.max() > 0 else 0.0)
        for row in range(num_classes):
            for col in range(num_classes):
                count = int(confusion[row, col])
                if count == 0:
                    continue
                text_color = "white" if color_data[row, col] > threshold else "#222222"
                ax.text(col, row, str(count), ha="center", va="center", color=text_color, fontsize=fontsize)

        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        colorbar.outline.set_linewidth(0.6)
        colorbar.ax.tick_params(labelsize=8.5, width=0.6)
        colorbar.set_label("Row-normalised proportion" if normalize else "Count", fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.savefig(output_path)
        plt.close(fig)


def compute_multiclass_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> dict[str, float | None]:
    """Compute one-vs-rest multiclass ROC-AUC scores."""
    from sklearn.metrics import roc_auc_score

    labels = np.arange(num_classes)
    try:
        macro_auc = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
        weighted_auc = float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        macro_auc = None
        weighted_auc = None

    y_one_hot = np.eye(num_classes, dtype=np.int64)[y_true]
    try:
        micro_auc = float(roc_auc_score(y_one_hot.ravel(), probabilities.ravel()))
    except ValueError:
        micro_auc = None

    return {
        "roc_auc_macro": macro_auc,
        "roc_auc_micro": micro_auc,
        "roc_auc_weighted": weighted_auc,
    }


def save_roc_curve_png(
    path: str | Path,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> None:
    """Save styled one-vs-rest ROC curves with per-class and average AUCs."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y_one_hot = np.eye(num_classes, dtype=np.int64)[y_true]

    with plt.rc_context(_PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        fpr_grid = np.linspace(0.0, 1.0, 256)
        interpolated_tpr = np.zeros_like(fpr_grid)
        class_aucs: list[float] = []
        valid_classes = 0
        cmap = plt.get_cmap("tab20")
        line_styles = ("-", "--", "-.", ":")
        for class_id in range(num_classes):
            positives = y_one_hot[:, class_id].sum()
            negatives = y_one_hot.shape[0] - positives
            if positives == 0 or negatives == 0:
                continue
            fpr, tpr, _ = roc_curve(y_one_hot[:, class_id], probabilities[:, class_id])
            class_auc = float(auc(fpr, tpr))
            class_aucs.append(class_auc)
            ax.plot(
                fpr,
                tpr,
                color=cmap(class_id % 20),
                linestyle=line_styles[(class_id // 20) % len(line_styles)],
                linewidth=1.7,
                alpha=0.88,
                label=f"Class {class_id} (AUC = {class_auc:.3f})",
            )
            interpolated_tpr += np.interp(fpr_grid, fpr, tpr)
            valid_classes += 1

        if valid_classes > 0:
            macro_tpr = interpolated_tpr / valid_classes
            macro_tpr[0], macro_tpr[-1] = 0.0, 1.0
            ax.plot(
                fpr_grid,
                macro_tpr,
                color="#D62728",
                linewidth=3.0,
                linestyle="--",
                label=f"Macro-average (AUC = {auc(fpr_grid, macro_tpr):.3f})",
                zorder=5,
            )

        try:
            fpr_micro, tpr_micro, _ = roc_curve(y_one_hot.ravel(), probabilities.ravel())
            ax.plot(
                fpr_micro,
                tpr_micro,
                color="#111111",
                linewidth=3.0,
                label=f"Micro-average (AUC = {auc(fpr_micro, tpr_micro):.3f})",
                zorder=6,
            )
        except ValueError:
            pass

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="#777777",
            linewidth=1.5,
            label="Chance",
            zorder=1,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("False Positive Rate", fontsize=14, fontweight="bold")
        ax.set_ylabel("True Positive Rate", fontsize=14, fontweight="bold")
        ax.set_title(
            f"Multiclass ROC Curves (One-vs-Rest)\n"
            f"{valid_classes} classes | mean class AUC = {np.mean(class_aucs):.3f}",
            fontsize=16,
            fontweight="bold",
        )
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(
            loc="lower right",
            fontsize=8.5 if valid_classes > 10 else 9.5,
            ncol=2 if valid_classes > 10 else 1,
            handlelength=2.4,
            columnspacing=0.9,
            framealpha=0.92,
            shadow=False,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#555555")
        fig.savefig(output_path)
        plt.close(fig)


def save_tsne_png(
    path: str | Path,
    features: np.ndarray,
    labels: np.ndarray,
    max_points: int = 2000,
    seed: int = 42,
    title: str = "t-SNE visualization",
) -> dict[str, float | int | None]:
    """Save a 3D t-SNE visualization and return clustering metrics."""
    if features.shape[0] < 3:
        return {
            "num_samples": int(features.shape[0]),
            "silhouette_feature_space": None,
            "silhouette_tsne_3d": None,
        }

    import matplotlib.pyplot as plt
    from matplotlib import transforms
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if features.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(features.shape[0], size=max_points, replace=False)
        features = features[indices]
        labels = labels[indices]

    # Silhouette score on the high-dimensional features (true clustering quality;
    # computing it on the 3D t-SNE embedding would be misleading).
    silhouette_feature_space: float | None = None
    if len(np.unique(labels)) > 1:
        try:
            silhouette_feature_space = float(silhouette_score(features, labels))
        except ValueError:
            silhouette_feature_space = None

    perplexity = min(30, max(2, (features.shape[0] - 1) // 3))
    embedding = TSNE(
        n_components=3,
        perplexity=perplexity,
        init="pca" if features.shape[1] >= 3 else "random",
        learning_rate="auto",
        random_state=seed,
        max_iter=1000,
    ).fit_transform(features)

    silhouette_tsne_3d: float | None = None
    if len(np.unique(labels)) > 1:
        try:
            silhouette_tsne_3d = float(silhouette_score(embedding, labels))
        except ValueError:
            silhouette_tsne_3d = None

    classes = np.unique(labels)
    qualitative = len(classes) <= 20
    cmap = plt.get_cmap("tab20" if qualitative else "gist_ncar")

    with plt.rc_context(_PAPER_STYLE):
        fig = plt.figure(figsize=(11, 9), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")
        for order, class_id in enumerate(classes):
            mask = labels == class_id
            color = cmap(order) if qualitative else cmap(order / max(1, len(classes) - 1))
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                s=36,
                color=color,
                label=str(int(class_id)),
                alpha=0.72,
                edgecolors="#222222",
                linewidths=0.35,
                depthshade=True,
            )
        feature_text = (
            f"{silhouette_feature_space:.3f}" if silhouette_feature_space is not None else "n/a"
        )
        tsne_text = f"{silhouette_tsne_3d:.3f}" if silhouette_tsne_3d is not None else "n/a"
        ax.set_title(
            f"{title}\n"
            f"Silhouette (feature space): {feature_text} | "
            f"Silhouette (t-SNE 3D): {tsne_text} [visualization only]"
        )
        ax.set_xlabel("t-SNE component 1")
        ax.set_ylabel("t-SNE component 2")
        ax.set_zlabel("t-SNE component 3")
        ax.view_init(elev=24, azim=42)
        ax.grid(True)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor("#f7f8fa")
            axis.pane.set_edgecolor("#d0d4da")
            axis._axinfo["grid"].update(color="#d7dbe0", linestyle="--", linewidth=0.6)
        ax.legend(
            title="Class",
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            bbox_transform=ax.transAxes + transforms.ScaledTranslation(0, -200 / 72, fig.dpi_scale_trans),
            fontsize=7.5,
            title_fontsize=8.5,
            ncol=1 if len(classes) <= 12 else 2,
            handletextpad=0.3,
            columnspacing=0.8,
            borderpad=0.6,
            frameon=True,
            framealpha=0.82,
            facecolor="white",
            edgecolor="#b8bec6",
        )
        fig.savefig(output_path)
        plt.close(fig)
    return {
        "num_samples": int(features.shape[0]),
        "silhouette_feature_space": silhouette_feature_space,
        "silhouette_tsne_3d": silhouette_tsne_3d,
    }


def save_history_plots(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Save train/validation loss, accuracy, macro-F1, and MCC curves."""
    if not rows:
        return
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]

    def series(key: str) -> list[float] | None:
        return [row[key] for row in rows] if key in rows[0] else None

    panels = [
        ("Loss", "train_loss", "val_loss"),
        ("Accuracy", "train_accuracy", "val_accuracy"),
        ("Macro-F1", "train_macro_f1", "val_macro_f1"),
        ("MCC", "train_mcc", "val_mcc"),
    ]
    panels = [panel for panel in panels if series(panel[2]) is not None]
    val_macro_f1 = series("val_macro_f1")
    best_index = int(np.argmax(val_macro_f1)) if val_macro_f1 else None

    with plt.rc_context(_PAPER_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
        flat_axes = axes.ravel()
        for index, (title, train_key, val_key) in enumerate(panels):
            ax = flat_axes[index]
            train_series = series(train_key)
            if train_series is not None:
                ax.plot(epochs, train_series, color=_TRAIN_COLOR, marker="o", markersize=2.5, linewidth=1.6, label="train")
            ax.plot(epochs, series(val_key), color=_VAL_COLOR, marker="s", markersize=2.5, linewidth=1.6, label="validation")
            if best_index is not None:
                ax.axvline(epochs[best_index], color=_BEST_COLOR, linestyle="--", linewidth=1.1, alpha=0.8)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
            ax.legend(frameon=False, loc="best")
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if title == "Macro-F1" and best_index is not None and val_macro_f1 is not None:
                ax.annotate(
                    f"best = {val_macro_f1[best_index]:.3f}\n@ epoch {epochs[best_index]}",
                    xy=(epochs[best_index], val_macro_f1[best_index]),
                    xytext=(0.5, 0.12),
                    textcoords="axes fraction",
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=_BEST_COLOR),
                )

        for index in range(len(panels), len(flat_axes)):
            flat_axes[index].axis("off")

        fig.suptitle("Training history", fontsize=14, fontweight="bold")
        fig.savefig(output_path)
        plt.close(fig)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a model checkpoint and its config."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "config": config}
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)
