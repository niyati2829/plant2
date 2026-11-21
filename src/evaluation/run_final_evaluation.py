"""Run comprehensive final holdout test evaluation and error analysis."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import torch

from src.models.transfer_learning import TomatoResNet18
from src.data.dataset import create_dataloaders
from src.utils.device import get_device
from src.evaluation.error_analysis import (
    analyze_classification_errors,
    plot_detailed_confusion_matrix,
    plot_error_confidence_distribution,
    plot_top_confusions,
)


def run_final_evaluation_pipeline(
    checkpoint_path: Path = PROJECT_ROOT / "artifacts/checkpoints/best_resnet18_weighted.pt",
    splits_dir: Path = PROJECT_ROOT / "data/splits",
    output_dir: Path = PROJECT_ROOT / "artifacts/figures/final_evaluation",
    error_csv: Path = PROJECT_ROOT / "reports/final_error_analysis.csv",
    class_metrics_csv: Path = PROJECT_ROOT / "reports/final_per_class_metrics.csv",
):
    print("=" * 70)
    print("STARTING FINAL HOLDOUT ERROR ANALYSIS PIPELINE (Phase F — REQ-10)")
    print("=" * 70)

    device = get_device(preferred="mps")
    print(f"Device: {device}")

    # 1. Load production model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = TomatoResNet18(num_classes=10, weights=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 2. Test dataloader
    print("Loading test dataloader (N=2,724)...")
    loaders = create_dataloaders(splits_dir=splits_dir, batch_size=64, num_workers=0)
    test_loader = loaders["test"]

    class_mapping_path = splits_dir / "class_mapping.json"
    with open(class_mapping_path, "r") as f:
        class_mapping = json.load(f)
    # Sort class names by integer index
    class_names = [k for k, _ in sorted(class_mapping.items(), key=lambda item: item[1])]

    # 3. Analyze errors
    print("\nRunning full test set error audit...")
    analysis = analyze_classification_errors(
        model=model,
        dataloader=test_loader,
        class_names=class_names,
        device=device,
    )

    total = analysis["total_samples"]
    errors_count = analysis["total_errors"]
    accuracy = analysis["accuracy"]
    rep = analysis["classification_report"]

    print("\n" + "=" * 70)
    print("FINAL HOLDOUT EVALUATION METRICS")
    print("=" * 70)
    print(f"Total Test Samples:     {total}")
    print(f"Correct Predictions:    {total - errors_count}")
    print(f"Misclassifications:     {errors_count}")
    print(f"Test Accuracy:          {accuracy * 100:.2f}%")
    print(f"Test Macro Precision:   {rep['macro avg']['precision']:.4f}")
    print(f"Test Macro Recall:      {rep['macro avg']['recall']:.4f}")
    print(f"Test Macro F1-Score:    {rep['macro avg']['f1-score']:.4f}")
    print(f"Test Weighted F1-Score: {rep['weighted avg']['f1-score']:.4f}")
    print(f"High-Conf Errors (>=80%): {analysis['high_conf_errors_count']}")
    print(f"Low-Conf Correct (<70%):  {analysis['low_conf_correct_count']}")
    print("=" * 70)

    # 4. Save per-class metrics CSV
    per_class_rows = []
    for cls in class_names:
        stats = rep[cls]
        per_class_rows.append({
            "Class_Name": cls,
            "Precision": stats["precision"],
            "Recall": stats["recall"],
            "F1_Score": stats["f1-score"],
            "Support": int(stats["support"]),
        })

    df_per_class = pd.DataFrame(per_class_rows)
    class_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    df_per_class.to_csv(class_metrics_csv, index=False)
    print(f"Saved per-class metrics to: {class_metrics_csv}")

    # 5. Save detailed error list CSV
    df_errors = pd.DataFrame(analysis["errors"])
    error_csv.parent.mkdir(parents=True, exist_ok=True)
    df_errors.to_csv(error_csv, index=False)
    print(f"Saved detailed failure instances ({len(df_errors)} errors) to: {error_csv}")

    # 6. Generate Figures
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_norm_path = output_dir / "confusion_matrix_normalized.png"
    print(f"Saving normalized confusion matrix -> {cm_norm_path}")
    plot_detailed_confusion_matrix(analysis["confusion_matrix"], class_names, save_path=str(cm_norm_path), normalize=True)

    cm_raw_path = output_dir / "confusion_matrix_raw.png"
    print(f"Saving raw confusion matrix -> {cm_raw_path}")
    plot_detailed_confusion_matrix(analysis["confusion_matrix"], class_names, save_path=str(cm_raw_path), normalize=False)

    conf_dist_path = output_dir / "error_confidence_distribution.png"
    print(f"Saving error confidence distribution -> {conf_dist_path}")
    plot_error_confidence_distribution(analysis["correct_confs"], analysis["error_confs"], save_path=str(conf_dist_path))

    top_conf_path = output_dir / "top_confusion_pairs.png"
    print(f"Saving top confusion pairs -> {top_conf_path}")
    plot_top_confusions(analysis["top_confusions"], save_path=str(top_conf_path))

    print("\nFinal evaluation pipeline completed successfully!")
    return analysis


if __name__ == "__main__":
    run_final_evaluation_pipeline()
