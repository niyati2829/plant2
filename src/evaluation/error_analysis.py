"""Qualitative Error Analysis Module.

Provides functions to:
1. Identify misclassified samples with ground truth, prediction, and confidence score.
2. Calculate per-class false positives and false negatives.
3. Identify top confused class pairs.
4. Render and save a publication-quality misclassification gallery.

Author: Crop Disease Detection Project Team
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def find_misclassifications(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    file_paths: List[str],
    class_names: List[str],
) -> List[Dict[str, Any]]:
    """Identify and rank all misclassified instances.

    Args:
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        y_probs: Softmax probabilities of shape (N, C).
        file_paths: Image file paths (relative or absolute).
        class_names: List of class name strings sorted by index.

    Returns:
        List of error dicts sorted by prediction confidence descending.
    """
    errors = []
    for i in range(len(y_true)):
        true_idx = int(y_true[i])
        pred_idx = int(y_pred[i])
        if true_idx != pred_idx:
            pred_conf = float(y_probs[i, pred_idx])
            true_conf = float(y_probs[i, true_idx])
            errors.append({
                "sample_index": i,
                "file_path": file_paths[i],
                "true_class_index": true_idx,
                "true_class_name": class_names[true_idx],
                "pred_class_index": pred_idx,
                "pred_class_name": class_names[pred_idx],
                "pred_confidence": pred_conf,
                "true_confidence": true_conf,
            })

    # Sort by predicted confidence descending (highest confidence mistakes first)
    errors.sort(key=lambda x: x["pred_confidence"], reverse=True)
    return errors


def analyze_confusion_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Identify most frequently confused class pairs and compute FP/FN per class.

    Returns:
        Tuple of (confused_pairs_list, per_class_fp_fn_dict).
    """
    pair_counts = {}
    fp_counts = {name: 0 for name in class_names}
    fn_counts = {name: 0 for name in class_names}

    for true_idx, pred_idx in zip(y_true, y_pred):
        if true_idx != pred_idx:
            true_name = class_names[true_idx]
            pred_name = class_names[pred_idx]
            pair = (true_name, pred_name)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            fn_counts[true_name] += 1
            fp_counts[pred_name] += 1

    sorted_pairs = [
        {"true_class": pair[0], "pred_class": pair[1], "count": count}
        for pair, count in sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    per_class_fp_fn = {}
    for name in class_names:
        per_class_fp_fn[name] = {
            "false_positives": fp_counts[name],
            "false_negatives": fn_counts[name],
        }

    return sorted_pairs, per_class_fp_fn


def plot_misclassification_gallery(
    errors: List[Dict[str, Any]],
    output_path: Path | str,
    root_dir: Path | str = ".",
    max_images: int = 12,
) -> Path:
    """Render a visual gallery of top misclassified leaf images.

    Args:
        errors: List of error dictionaries from find_misclassifications.
        output_path: Where to save the output PNG.
        root_dir: Root directory prepended to relative image paths.
        max_images: Maximum number of images to plot (e.g. 12 for 3x4 grid).

    Returns:
        Saved file Path.
    """
    root_path = Path(root_dir)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not errors:
        # Perfect model fallback
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        ax.text(0.5, 0.5, "Zero misclassifications detected!", ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.savefig(out_file, bbox_inches="tight")
        plt.close(fig)
        return out_file

    n_images = min(len(errors), max_images)
    ncols = 4
    nrows = int(np.ceil(n_images / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), dpi=150)
    axes_flat = np.array(axes).flatten()

    for idx in range(nrows * ncols):
        ax = axes_flat[idx]
        if idx < n_images:
            err = errors[idx]
            img_path = root_path / err["file_path"]
            if img_path.exists():
                try:
                    with Image.open(img_path) as img:
                        ax.imshow(img.convert("RGB"))
                except Exception:
                    ax.text(0.5, 0.5, "Image Load Error", ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "File Missing", ha="center", va="center")

            true_short = err["true_class_name"].replace("Tomato___", "").replace("_", " ")
            pred_short = err["pred_class_name"].replace("Tomato___", "").replace("_", " ")

            title_text = (
                f"True: {true_short}\n"
                f"Pred: {pred_short}\n"
                f"Conf: {err['pred_confidence']*100:.1f}% (True: {err['true_confidence']*100:.1f}%)"
            )
            ax.set_title(title_text, fontsize=9, fontweight="bold", color="#d32f2f", pad=6)
        ax.axis("off")

    plt.suptitle("Top Representative Misclassifications on Holdout Test Set", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    return out_file


# --- Consolidated from src/final_evaluation/error_analysis.py ---
"""In-depth classification error analysis, confusion metrics, and failure characterization."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn


def clean_class_name(name: str) -> str:
    """Format PlantVillage class names for readable display."""
    return name.replace("Tomato___", "").replace("_", " ")


def analyze_classification_errors(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Dict[str, any]:
    """Execute exhaustive error analysis over the dataset.

    Args:
        model: Production PyTorch model.
        dataloader: Holdout test DataLoader.
        class_names: List of class strings in index order.
        device: Evaluation device.

    Returns:
        Dictionary of confusion matrix, per-class metrics, error instances, and confidences.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_confs = []
    all_probs = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    confs_arr = np.array(all_confs)
    probs_arr = np.concatenate(all_probs, axis=0)

    # 1. Confusion Matrix
    cm = confusion_matrix(targets_arr, preds_arr, labels=list(range(len(class_names))))

    # 2. Per-class metrics
    report_dict = classification_report(
        targets_arr,
        preds_arr,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # 3. Identify all error cases
    error_indices = np.where(preds_arr != targets_arr)[0]
    correct_indices = np.where(preds_arr == targets_arr)[0]

    errors = []
    dataset = dataloader.dataset
    has_df = hasattr(dataset, "df")

    for idx in error_indices:
        t_cls = targets_arr[idx]
        p_cls = preds_arr[idx]
        conf = confs_arr[idx]
        img_rel_path = dataset.df.iloc[idx]["image_path"] if has_df else f"sample_{idx}"
        errors.append({
            "sample_index": int(idx),
            "image_path": str(img_rel_path),
            "true_class_idx": int(t_cls),
            "true_class_name": class_names[t_cls],
            "pred_class_idx": int(p_cls),
            "pred_class_name": class_names[p_cls],
            "confidence": float(conf),
            "margin": float(conf - probs_arr[idx, t_cls]),
        })

    # 4. Top Confused Pairs
    confusion_pairs = {}
    for err in errors:
        pair = (err["true_class_name"], err["pred_class_name"])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

    sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

    # 5. High-confidence errors vs low-confidence correct
    high_conf_errors = [e for e in errors if e["confidence"] >= 0.80]
    low_conf_correct = [int(idx) for idx in correct_indices if confs_arr[idx] < 0.70]

    return {
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "errors": errors,
        "top_confusions": sorted_confusions,
        "total_errors": len(errors),
        "total_samples": len(targets_arr),
        "accuracy": float(np.mean(preds_arr == targets_arr)),
        "correct_confs": confs_arr[correct_indices],
        "error_confs": confs_arr[error_indices],
        "high_conf_errors_count": len(high_conf_errors),
        "low_conf_correct_count": len(low_conf_correct),
        "class_names": class_names,
    }


def plot_detailed_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """Plot publication-quality confusion matrix heatmap."""
    clean_names = [clean_class_name(c) for c in class_names]

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)
        fmt = ".2%"
        title = "Normalized Confusion Matrix (Holdout Test Set N=2,724)"
    else:
        cm_display = cm
        fmt = "d"
        title = "Raw Confusion Matrix (Holdout Test Set N=2,724)"

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val = cm_display[i, j]
            txt = f"{val:.1%}" if normalize else str(val)
            color = "white" if val > thresh else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8, weight="bold")

    ax.set_xticks(np.arange(len(clean_names)))
    ax.set_yticks(np.arange(len(clean_names)))

    ax.set_xlabel("Predicted Pathology Class", fontsize=12, weight="bold", labelpad=10)
    ax.set_ylabel("True Ground-Truth Class", fontsize=12, weight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, weight="bold", pad=14)
    plt.xticks(rotation=40, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_error_confidence_distribution(
    correct_confs: np.ndarray,
    error_confs: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot comparative histogram of model confidence on correct vs erroneous classifications."""
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(0.0, 1.0, 25)
    ax.hist(correct_confs, bins=bins, color="#2ecc71", alpha=0.7, label=f"Correct (N={len(correct_confs)})", density=True, edgecolor="black")
    if len(error_confs) > 0:
        ax.hist(error_confs, bins=bins, color="#e74c3c", alpha=0.8, label=f"Errors (N={len(error_confs)})", density=True, edgecolor="black")

    ax.set_xlabel("Predicted Model Confidence", fontsize=11, weight="bold")
    ax.set_ylabel("Probability Density", fontsize=11, weight="bold")
    ax.set_title("Prediction Confidence Distribution: Correct vs Erroneous Predictions", fontsize=13, weight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(framealpha=0.9, fontsize=11)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_top_confusions(
    top_confusions: List[Tuple[Tuple[str, str], int]],
    save_path: Optional[str] = None,
    top_k: int = 8,
) -> plt.Figure:
    """Bar chart of the most frequent misclassification pairs."""
    top_k_items = top_confusions[:top_k]
    if not top_k_items:
        labels = ["None"]
        counts = [0]
    else:
        labels = [f"{clean_class_name(pair[0])} -> {clean_class_name(pair[1])}" for pair, _ in top_k_items]
        counts = [cnt for _, cnt in top_k_items]

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, counts, color="#e67e22", edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Misclassified Samples", fontsize=11, weight="bold")
    ax.set_title(f"Top {len(labels)} Misclassification Pairs (Test Set Errors)", fontsize=13, weight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)

    for i, v in enumerate(counts):
        ax.text(v + 0.1, i, str(v), va="center", weight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig
