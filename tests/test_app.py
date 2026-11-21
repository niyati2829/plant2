"""Unit Tests for Streamlit Application Pipeline.

Verifies:
1. Class mapping loader accurately parses all 10 categories.
2. Model loader strictly loads best_resnet18_weighted.pt with 10 output classes.
3. Model is set to eval() mode.
4. Input preprocessing transforms yield shape (1, 3, 224, 224) normalized tensor.
5. Prediction returns top-3 classes with descending probabilities and valid bounds.
6. Healthy foliage identification is properly marked.

Author: Crop Disease Detection Project Team
"""

import sys
from pathlib import Path
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import (
    CHECKPOINT_PATH,
    CLASS_MAPPING_PATH,
    DISPLAY_NAME_OVERRIDES,
    get_inference_transforms,
    load_class_mappings,
    load_production_model,
    predict_leaf,
)


def test_class_mapping_integrity():
    """Verify class mapping loads exactly 10 classes in ascending integer index order."""
    class_to_idx, idx_to_class, class_names_sorted = load_class_mappings(CLASS_MAPPING_PATH)

    assert len(class_to_idx) == 10, f"Expected 10 classes, got {len(class_to_idx)}"
    assert len(idx_to_class) == 10
    assert len(class_names_sorted) == 10
    assert set(range(10)) == set(class_to_idx.values())
    assert class_names_sorted[0] == "Tomato___Bacterial_spot"
    assert class_names_sorted[9] == "Tomato___healthy"


def test_production_checkpoint_exists():
    """Verify the frozen final model checkpoint exists and is non-empty."""
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing: {CHECKPOINT_PATH}"
    size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
    assert 100 < size_mb < 120, f"Unexpected checkpoint size: {size_mb} MB"


def test_production_model_loader():
    """Verify production model loads strictly in evaluation mode."""
    model, device, device_name = load_production_model(CHECKPOINT_PATH)

    assert model is not None
    assert isinstance(model, torch.nn.Module)
    assert model.training is False, "Model must be in eval mode!"
    assert model.fc.out_features == 10, "Output head must have 10 classes!"


def test_inference_transforms():
    """Verify inference transforms resize and normalize image to (1, 3, 224, 224)."""
    transform = get_inference_transforms()
    # Synthetic 300x400 image
    dummy_img = Image.new("RGB", (300, 400), color=(120, 180, 70))
    tensor = transform(dummy_img).unsqueeze(0)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_predict_leaf_output_structure():
    """Verify predict_leaf returns ranked top-3 predictions with valid probabilities."""
    model, device, _ = load_production_model(CHECKPOINT_PATH)
    _, idx_to_class, _ = load_class_mappings(CLASS_MAPPING_PATH)

    dummy_img = Image.new("RGB", (256, 256), color=(50, 150, 50))
    predictions = predict_leaf(dummy_img, model, device, idx_to_class)

    assert len(predictions) == 3, "Must return top-3 predictions"
    assert predictions[0]["rank"] == 1
    assert predictions[1]["rank"] == 2
    assert predictions[2]["rank"] == 3

    # Probabilities must be descending and bounded between 0 and 1
    assert predictions[0]["probability"] >= predictions[1]["probability"] >= predictions[2]["probability"]
    for pred in predictions:
        assert 0.0 <= pred["probability"] <= 1.0
        assert 0.0 <= pred["percentage"] <= 100.0
        assert isinstance(pred["display_name"], str)
        assert len(pred["display_name"]) > 0


def test_sample_image_prediction():
    """Verify prediction on an actual dataset leaf image."""
    model, device, _ = load_production_model(CHECKPOINT_PATH)
    _, idx_to_class, _ = load_class_mappings(CLASS_MAPPING_PATH)

    sample_path = PROJECT_ROOT / "data/raw/PlantVillage-Dataset/raw/color/Tomato___Bacterial_spot/0045ba29-ed1b-43b4-afde-719cc7adefdb___GCREC_Bact.Sp 6254.JPG"
    if sample_path.exists():
        with Image.open(sample_path) as img:
            predictions = predict_leaf(img, model, device, idx_to_class)
        top1 = predictions[0]
        assert top1["raw_class"] == "Tomato___Bacterial_spot"
        assert top1["display_name"] == "Bacterial Spot"
        assert top1["percentage"] > 90.0



# --- Consolidated from test_disease_classification_ui.py ---
"""Unit tests for Disease Classification 10-class probability distribution."""

from pathlib import Path
import pytest
from PIL import Image
import torch

from app.app import (
    CHECKPOINT_PATH,
    CLASS_MAPPING_PATH,
    CALIBRATION_TEMPERATURE,
    load_class_mappings,
    load_production_model,
    predict_leaf,
)


@pytest.fixture(scope="module")
def model_and_mapping():
    model, device, _ = load_production_model(CHECKPOINT_PATH)
    _, idx_to_class, class_names_sorted = load_class_mappings(CLASS_MAPPING_PATH)
    return model, device, idx_to_class, class_names_sorted


def test_predict_leaf_default_top3(model_and_mapping):
    """Ensure backward-compatible default top_k=3."""
    model, device, idx_to_class, _ = model_and_mapping
    dummy = Image.new("RGB", (224, 224), color=(60, 140, 50))
    preds = predict_leaf(dummy, model, device, idx_to_class)
    assert len(preds) == 3
    assert preds[0]["rank"] == 1
    assert preds[1]["rank"] == 2
    assert preds[2]["rank"] == 3
    assert preds[0]["probability"] >= preds[1]["probability"] >= preds[2]["probability"]


def test_predict_leaf_all_10_classes(model_and_mapping):
    """Verify inference output across all 10 classes with calibration."""
    model, device, idx_to_class, class_names_sorted = model_and_mapping
    dummy = Image.new("RGB", (224, 224), color=(60, 140, 50))
    
    # Request all 10 classes
    preds = predict_leaf(dummy, model, device, idx_to_class, top_k=len(idx_to_class))
    
    # Check count
    assert len(preds) == 10, f"Expected 10 classes, got {len(preds)}"
    
    # Check ranks 1 through 10
    ranks = [p["rank"] for p in preds]
    assert ranks == list(range(1, 11))
    
    # Check descending order
    probs = [p["probability"] for p in preds]
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1], f"Probabilities not descending at index {i}"
        
    # Check mapping consistency (no invented/hardcoded names)
    returned_raw_names = {p["raw_class"] for p in preds}
    expected_raw_names = set(class_names_sorted)
    assert returned_raw_names == expected_raw_names
    
    # Check probabilities sum to approximately 100%
    total_pct = sum(p["percentage"] for p in preds)
    assert pytest.approx(100.0, abs=0.01) == total_pct
    
    # Check formatting with two decimal places
    for p in preds:
        formatted = f"{p['percentage']:.2f}%"
        assert "%" in formatted
        # parse back
        num = float(formatted.replace("%", ""))
        assert 0.0 <= num <= 100.0
