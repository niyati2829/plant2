"""Unit tests for holdout error analysis and confusion reporting."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.error_analysis import (
    analyze_classification_errors,
    clean_class_name,
    plot_detailed_confusion_matrix,
    plot_error_confidence_distribution,
    plot_top_confusions,
)


def test_clean_class_name():
    """Verify clean name formatting removes Tomato prefix and replaces underscores."""
    raw = "Tomato___Early_blight"
    cleaned = clean_class_name(raw)
    assert cleaned == "Early blight"


def test_analyze_classification_errors_toy():
    """Analyze errors on a known synthetic toy model and dataset."""
    class ToyClassifier(nn.Module):
        def forward(self, x):
            # Output fixed logits: class 0 gets high logit, except sample 3 gets class 1
            b = x.size(0)
            logits = torch.zeros(b, 3)
            logits[:, 0] = 5.0
            return logits

    # 10 samples: 8 of class 0, 2 of class 1
    x = torch.randn(10, 2)
    y = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=5)

    model = ToyClassifier()
    class_names = ["Class_A", "Class_B", "Class_C"]

    res = analyze_classification_errors(
        model=model,
        dataloader=loader,
        class_names=class_names,
        device=torch.device("cpu"),
    )

    assert res["total_samples"] == 10
    assert res["total_errors"] == 2
    assert res["accuracy"] == 0.8
    assert res["confusion_matrix"].shape == (3, 3)
    assert len(res["errors"]) == 2
    assert res["top_confusions"][0][0] == ("Class_B", "Class_A")
    assert res["top_confusions"][0][1] == 2


def test_plotting_helpers_generate_figures():
    """Verify visualization functions return valid matplotlib Figures."""
    cm = np.array([[10, 2], [1, 15]])
    class_names = ["Tomato___Healthy", "Tomato___Blight"]

    fig_cm = plot_detailed_confusion_matrix(cm, class_names, save_path=None)
    assert fig_cm is not None

    fig_dist = plot_error_confidence_distribution(
        correct_confs=np.array([0.95, 0.98, 0.88]),
        error_confs=np.array([0.65, 0.72]),
        save_path=None,
    )
    assert fig_dist is not None

    fig_top = plot_top_confusions(
        top_confusions=[(("Class_A", "Class_B"), 5), (("Class_C", "Class_B"), 2)],
        save_path=None,
    )
    assert fig_top is not None
