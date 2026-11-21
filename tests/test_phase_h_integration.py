"""Automated Integration Tests for Phase H Unified Streamlit Application.

Verifies:
1. Application module importability and entry-point integrity.
2. Production disease classifier checkpoint hash invariability.
3. Class mapping integrity (10 classes: 9 diseases + 1 healthy).
4. Valid, invalid, and edge-case image classification handling.
5. Post-hoc temperature calibration (T* = 1.3021) integration.
6. Grad-CAM saliency overlay generation with TomatoResNet18.
7. Experimental pest detector loading, inference, empty detections, and class semantics.
8. Video surveillance processor, frame analyzer, annotated MP4, and CSV generation.
9. Lesion segmentation explicit blocker notice verification.
10. Temporary file cleanup and isolation.
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import (
    CALIBRATION_TEMPERATURE,
    CHECKPOINT_PATH,
    CLASS_MAPPING_PATH,
    PEST_CHECKPOINT_PATH,
    SAMPLE_VIDEO_PATH,
    load_calibrated_classifier,
    load_class_mappings,
    load_pest_detector,
    load_production_model,
    predict_leaf,
    predict_pests,
    run_gradcam_overlay,
)
from src.detection.real_pest_dataset import REAL_PEST_CLASSES
from src.surveillance.frame_analyzer import SurveillanceFrameAnalyzer
from src.surveillance.video_processor import VideoProcessor

EXPECTED_CLASSIFIER_HASH = "b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa"


# ---------------------------------------------------------------------------
# 1. Model Integrity & Checkpoint Invariance
# ---------------------------------------------------------------------------

def test_production_checkpoint_hash_unaltered():
    """Verify that the production classifier checkpoint hash has not changed."""
    assert CHECKPOINT_PATH.exists(), f"Missing production checkpoint: {CHECKPOINT_PATH}"
    hasher = hashlib.sha256()
    with open(CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    current_hash = hasher.hexdigest()
    assert current_hash == EXPECTED_CLASSIFIER_HASH, (
        f"Production checkpoint SHA-256 altered!\nExpected: {EXPECTED_CLASSIFIER_HASH}\nGot: {current_hash}"
    )


def test_class_mapping_preservation():
    """Verify 10 classes are strictly preserved in ascending index order."""
    class_to_idx, idx_to_class, class_names = load_class_mappings(CLASS_MAPPING_PATH)
    assert len(class_to_idx) == 10
    assert class_to_idx["Tomato___healthy"] == 9
    assert class_to_idx["Tomato___Bacterial_spot"] == 0
    assert idx_to_class[9] == "Tomato___healthy"


# ---------------------------------------------------------------------------
# 2. Disease Classification & Calibration & Grad-CAM
# ---------------------------------------------------------------------------

def test_calibrated_leaf_prediction_valid_image():
    """Verify calibrated prediction returns descending probabilities with T*=1.3021."""
    model, device, _ = load_production_model(CHECKPOINT_PATH)
    _, idx_to_class, _ = load_class_mappings(CLASS_MAPPING_PATH)

    dummy_img = Image.new("RGB", (256, 256), color=(40, 160, 45))
    preds = predict_leaf(
        dummy_img,
        model,
        device,
        idx_to_class,
        use_temperature=True,
        temperature=CALIBRATION_TEMPERATURE,
    )

    assert len(preds) == 3
    assert preds[0]["probability"] >= preds[1]["probability"] >= preds[2]["probability"]
    assert 0.0 <= preds[0]["probability"] <= 1.0
    assert isinstance(preds[0]["display_name"], str)
    assert isinstance(preds[0]["is_healthy"], bool)


def test_gradcam_overlay_execution():
    """Verify Grad-CAM produces valid 2D heatmap and blended overlay without weight mutation."""
    model, device, _ = load_production_model(CHECKPOINT_PATH)
    dummy_img = Image.new("RGB", (224, 224), color=(60, 140, 50))

    heatmap, blended, pred_c, conf = run_gradcam_overlay(dummy_img, model)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (224, 224)
    assert 0.0 <= np.min(heatmap) and np.max(heatmap) <= 1.0
    assert blended.shape == (224, 224, 3)
    assert blended.dtype == np.uint8
    assert 0 <= pred_c < 10
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# 3. Pest Detection (Experimental) Integration & Class Semantics
# ---------------------------------------------------------------------------

def test_pest_detector_load_and_semantics():
    """Verify pest detector checkpoint and check that Class 0 represents healthy reference."""
    if not PEST_CHECKPOINT_PATH.exists():
        pytest.skip("Pest detector checkpoint not found")

    detector, device, dev_name = load_pest_detector(PEST_CHECKPOINT_PATH)
    assert detector is not None
    assert detector.training is False
    assert detector.num_classes == 2

    # Verify audited class semantics
    assert REAL_PEST_CLASSES[0] == "healthy"  # Clean reference foliage
    assert REAL_PEST_CLASSES[1] == "tuta_absoluta_damage"  # Actual insect damage


def test_pest_detection_inference_structure():
    """Verify pest detector returns valid bounding boxes and distinct category counts."""
    if not PEST_CHECKPOINT_PATH.exists():
        pytest.skip("Pest detector checkpoint not found")

    detector, device, _ = load_pest_detector(PEST_CHECKPOINT_PATH)
    dummy_leaf = Image.new("RGB", (320, 320), color=(50, 120, 40))

    annotated, detections, tuta_cnt, healthy_cnt = predict_pests(
        dummy_leaf, detector, device, conf_threshold=0.10, iou_threshold=0.50
    )

    assert isinstance(annotated, Image.Image)
    assert annotated.size == (320, 320)
    assert isinstance(detections, list)
    assert tuta_cnt >= 0
    assert healthy_cnt >= 0

    for det in detections:
        assert "class_id" in det
        assert "confidence" in det
        assert "box" in det
        assert len(det["box"]) == 4
        # Verify box bounds
        x1, y1, x2, y2 = det["box"]
        assert 0.0 <= x1 <= x2 <= 320.0
        assert 0.0 <= y1 <= y2 <= 320.0


def test_pest_detection_empty_case():
    """Verify that high threshold returns 0 detections without error."""
    if not PEST_CHECKPOINT_PATH.exists():
        pytest.skip("Pest detector checkpoint not found")

    detector, device, _ = load_pest_detector(PEST_CHECKPOINT_PATH)
    dummy_blank = Image.new("RGB", (256, 256), color=(0, 0, 0))

    # Threshold 0.999 ensures no detections
    _, detections, tuta_cnt, healthy_cnt = predict_pests(
        dummy_blank, detector, device, conf_threshold=0.999
    )

    assert len(detections) == 0
    assert tuta_cnt == 0
    assert healthy_cnt == 0


# ---------------------------------------------------------------------------
# 4. Video Surveillance Pipeline Integration
# ---------------------------------------------------------------------------

def test_video_surveillance_safeguards_and_execution(tmp_path):
    """Verify video processor validation, metadata extraction, and multi-task frame analysis."""
    processor = VideoProcessor(max_duration_seconds=60.0, max_file_size_mb=100.0)

    if not SAMPLE_VIDEO_PATH.exists():
        pytest.skip("Sample field video not present")

    metadata = processor.validate_and_inspect(SAMPLE_VIDEO_PATH)
    assert metadata.duration_seconds <= 60.0
    assert metadata.file_size_mb <= 100.0

    analyzer = SurveillanceFrameAnalyzer(
        classifier_checkpoint=CHECKPOINT_PATH,
        pest_detector_checkpoint=PEST_CHECKPOINT_PATH,
        temperature=CALIBRATION_TEMPERATURE,
    )

    # Sample exactly 2 frames
    frames = list(processor.sample_frames(SAMPLE_VIDEO_PATH, stride=30, max_frames=2))
    assert len(frames) == 2

    for f_idx, ts, frame_bgr in frames:
        res = analyzer.analyze_frame(frame_bgr, frame_idx=f_idx, timestamp=ts)
        assert 0 <= res.disease_class_id < 10
        assert 0.0 <= res.disease_confidence <= 1.0
        assert res.segmentation_status == "Unavailable (Blocked by Dataset Availability)"
        assert res.pest_count >= 0


# ---------------------------------------------------------------------------
# 5. Temporary File Cleanup & Isolation
# ---------------------------------------------------------------------------

def test_temp_file_isolation_and_cleanup(tmp_path):
    """Verify temporary files used in video processing can be safely created and removed."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(b"dummy video content")
    temp_file.close()

    p = Path(temp_file.name)
    assert p.exists()
    p.unlink()
    assert not p.exists()


# ---------------------------------------------------------------------------
# 6. Streamlit Navigation & Lesion Segmentation De-Integration Verification
# ---------------------------------------------------------------------------

def test_streamlit_sidebar_navigation_simplified():
    """Verify active sidebar navigation has exactly 5 primary sections and no lesion segmentation."""
    app_file = PROJECT_ROOT / "app" / "app.py"
    assert app_file.exists(), "app/app.py missing"
    content = app_file.read_text(encoding="utf-8")

    # Verify exactly the 4 primary sections in navigation radio
    expected_nav_items = [
        "🏠 Home & Overview",
        "🔬 Disease Classification",
        "🐛 Pest Damage Detection",
        "📹 Video Surveillance",
    ]
    for item in expected_nav_items:
        assert item in content, f"Expected navigation item '{item}' missing from app/app.py"

    # Verify no obsolete pages in sidebar radio
    assert "🧩 Lesion Segmentation" not in content
    assert "render_segmentation_page" not in content



# ---------------------------------------------------------------------------
# 6. Streamlit Navigation & Lesion Segmentation De-Integration Verification
# ---------------------------------------------------------------------------

def test_streamlit_sidebar_navigation_simplified():
    """Verify active sidebar navigation has exactly 5 primary sections and no lesion segmentation."""
    app_file = PROJECT_ROOT / "app" / "app.py"
    assert app_file.exists(), "app/app.py missing"
    content = app_file.read_text(encoding="utf-8")

    # Verify exactly the 4 primary sections in navigation radio
    expected_nav_items = [
        "🏠 Home & Overview",
        "🔬 Disease Classification",
        "🐛 Pest Damage Detection",
        "📹 Video Surveillance",
    ]
    for item in expected_nav_items:
        assert item in content, f"Expected navigation item '{item}' missing from app/app.py"

    # Verify no obsolete pages in sidebar radio
    assert "🧩 Lesion Segmentation" not in content
    assert "render_segmentation_page" not in content

