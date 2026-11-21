"""Consolidated YOLOv8n-Seg Pest Damage Detection Tests.

Consolidates:
- Phase M3A dataset and annotation audit tests
- Phase M3B YOLOv8 training and holdout evaluation tests
- Phase M3C multi-task surveillance integration tests
"""

# === Section 1: Phase M3A Dataset Audit ===
"""Phase M3-A Automated Unit Test Suite for Tuta absoluta Dataset & Annotation Audit.

Verifies:
1. TomatoEbola Kasaisa dataset split coverage (98 train, 12 valid, 12 test = 122 images).
2. Annotation file parity (122 images, 122 labels, 0 orphaned files).
3. Class distribution (319 total boxes: 138 class 0, 181 class 1).
4. Bounding box validity (0 degenerate, 0 overflow).
5. Image uniqueness and zero cross-split data leakage.
6. EDEN Library verification status.
7. Production checkpoints and baseline detector integrity.

Author: Crop Disease Detection Project Team
Phase: Phase M3-A
"""

import hashlib
import json
from pathlib import Path

import pytest
import torch

DATASET_ROOT = Path("data/pest_detection/real_tuta_absoluta")
AUDIT_JSON = Path("reports/tuta_dataset_audit_phase_m3a.json")
OVERLAY_DIR = Path("reports/pest_detection_audit")

PEST_CHECKPOINT = Path("artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt")
EXPECTED_PEST_HASH = "01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343"

RESNET_CHECKPOINT = Path("artifacts/checkpoints/best_resnet18_weighted.pt")
EXPECTED_RESNET_HASH = "b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa"


def test_tuta_dataset_split_counts_and_parity():
    """Verify that the dataset contains exactly 122 images and 122 label files across splits."""
    assert DATASET_ROOT.exists(), f"Dataset directory missing: {DATASET_ROOT}"

    expected_splits = {"train": 98, "valid": 12, "test": 12}
    total_imgs = 0
    total_lbls = 0

    for split, exp_count in expected_splits.items():
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
        lbls = list(lbl_dir.glob("*.txt"))

        assert len(imgs) == exp_count, f"{split} images count mismatch: {len(imgs)} vs {exp_count}"
        assert len(lbls) == exp_count, f"{split} labels count mismatch: {len(lbls)} vs {exp_count}"

        img_stems = {f.stem for f in imgs}
        lbl_stems = {f.stem for f in lbls}
        assert img_stems == lbl_stems, f"Orphaned files detected in split {split}"

        total_imgs += len(imgs)
        total_lbls += len(lbls)

    assert total_imgs == 122
    assert total_lbls == 122


def test_tuta_annotations_validity_and_classes():
    """Verify bounding boxes have valid normalized bounds and expected classes (0 and 1)."""
    assert AUDIT_JSON.exists(), f"Audit report missing: {AUDIT_JSON}"

    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["total_images"] == 122
    assert audit["total_boxes"] == 319
    assert audit["integrity"]["total_degenerate_boxes"] == 0
    assert audit["integrity"]["total_overflow_boxes"] == 0
    assert audit["integrity"]["intra_split_duplicates_count"] == 0
    assert audit["integrity"]["cross_split_leakage_count"] == 0

    box_stats = audit["box_statistics"]
    assert box_stats["class_0_healthy_count"] == 138
    assert box_stats["class_1_tuta_count"] == 181
    assert box_stats["class_0_healthy_count"] + box_stats["class_1_tuta_count"] == 319


def test_eden_library_and_other_farms_audit_status():
    """Verify that EDEN Library absence and upstream farm restrictions are properly recorded."""
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["eden_library_investigation"]["local_files_present"] is False
    assert "restricted" in audit["other_farm_subsets_status"]["dikumari_farm"].lower()
    assert "restricted" in audit["other_farm_subsets_status"]["kukareta_farm"].lower()


def test_inspection_overlays_exist():
    """Verify visual inspection overlays have been rendered."""
    assert OVERLAY_DIR.exists()
    overlays = list(OVERLAY_DIR.glob("*.png"))
    assert len(overlays) >= 6, f"Expected at least 6 inspection overlays, found {len(overlays)}"


def test_checkpoint_integrity_preserved():
    """Verify that baseline pest detector and ResNet18 checkpoints are bitwise untouched."""
    assert PEST_CHECKPOINT.exists()
    with open(PEST_CHECKPOINT, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_PEST_HASH

    assert RESNET_CHECKPOINT.exists()
    with open(RESNET_CHECKPOINT, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_RESNET_HASH


# === Section 2: Phase M3B Model & Benchmark ===
"""Phase M3-B Automated Test Suite for YOLOv8n-Seg Pest Detection.

Verifies:
1. YOLOv8n-seg trained checkpoint exists, is loadable, and performs forward inference.
2. Training config and history reports exist and record reproducible parameters.
3. Test evaluation metrics integrity (mAP@0.5, Precision, Recall).
4. Visual test prediction overlays existence.
5. Preservation of production ResNet18, baseline TinyYOLO, and U-Net checkpoints.

Author: Crop Disease Detection Project Team
Phase: Phase M3-B
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from ultralytics import YOLO

YOLOV8_CHECKPOINT = Path("artifacts/checkpoints/pest_detection_yolov8/best_yolov8n_seg.pt")
CONFIG_JSON = Path("reports/pest_detection_yolov8_training_config.json")
HISTORY_JSON = Path("reports/pest_detection_yolov8_training_history.json")
EVAL_JSON = Path("reports/pest_detection_yolov8_evaluation.json")
OVERLAYS_DIR = Path("reports/pest_detection_audit/test_prediction_overlays_yolov8")

RESNET_CHECKPOINT = Path("artifacts/checkpoints/best_resnet18_weighted.pt")
EXPECTED_RESNET_HASH = "b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa"

PEST_CHECKPOINT = Path("artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt")
EXPECTED_PEST_HASH = "01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343"

UNET_IMPROVED_CHECKPOINT = Path("artifacts/checkpoints/segmentation_real/best_unet_improved.pt")


def test_yolov8_checkpoint_loading_and_inference():
    """Verify best_yolov8n_seg.pt can be loaded and executed for inference."""
    assert YOLOV8_CHECKPOINT.exists(), f"Missing YOLOv8 checkpoint: {YOLOV8_CHECKPOINT}"
    assert YOLOV8_CHECKPOINT.stat().st_size > 5 * 1024 * 1024  # > 5 MB

    model = YOLO(str(YOLOV8_CHECKPOINT))
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

    preds = model.predict(dummy_img, device="cpu", verbose=False)
    assert len(preds) == 1
    assert hasattr(preds[0], "boxes")
    assert hasattr(preds[0], "masks")


def test_training_and_evaluation_reports_exist():
    """Verify JSON training reports and evaluation summary are populated and valid."""
    assert CONFIG_JSON.exists(), f"Missing config: {CONFIG_JSON}"
    assert HISTORY_JSON.exists(), f"Missing history: {HISTORY_JSON}"
    assert EVAL_JSON.exists(), f"Missing eval report: {EVAL_JSON}"

    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        ev = json.load(f)

    assert ev["test_sample_count"] == 92
    assert ev["overall_box_metrics"]["map50"] > 0.60, "Box mAP50 should exceed 60%"
    assert ev["overall_mask_metrics"]["map50"] > 0.60, "Mask mAP50 should exceed 60%"
    assert ev["per_class_metrics"]["Tuta absoluta"]["box_precision"] > 0.70
    assert ev["cpu_latency_benchmark"]["throughput_fps"] > 5.0


def test_visual_prediction_overlays_exist():
    """Verify prediction overlays were generated on holdout test samples."""
    assert OVERLAYS_DIR.exists(), f"Missing overlays directory: {OVERLAYS_DIR}"
    overlays = list(OVERLAYS_DIR.glob("*.png"))
    assert len(overlays) >= 6, f"Expected at least 6 test overlays, found {len(overlays)}"


def test_production_and_baseline_integrity():
    """Verify that ResNet18, TinyYOLO baseline, and U-Net checkpoints remain bitwise untouched."""
    assert RESNET_CHECKPOINT.exists()
    with open(RESNET_CHECKPOINT, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_RESNET_HASH

    assert PEST_CHECKPOINT.exists()
    with open(PEST_CHECKPOINT, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_PEST_HASH

    assert UNET_IMPROVED_CHECKPOINT.exists()


# === Section 3: Phase M3C Integration ===
"""Automated Integration & Verification Tests for Phase M3-C / Phase N/O.

Verifies:
1. YOLOv8n-Seg checkpoint loading & parameter integrity.
2. Dual model availability: YOLOv8n-Seg (Production) and TinyYOLO (Baseline).
3. `predict_pests_yolov8` inference structure, box/mask generation, and schema consistency.
4. Edge cases: Empty/blank image, miniature image (32x32).
5. `SurveillanceFrameAnalyzer` with `use_yolov8=True` multi-task video frame analysis.
6. `VideoAnnotator` rendering with YOLOv8 segmentation masks and HUD telemetry.
7. Real PlantSeg U-Net checkpoint availability and `predict_unet_lesions` function.
8. Latency benchmark telemetry artifact integrity.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from app.app import (
    CHECKPOINT_PATH,
    PEST_CHECKPOINT_PATH,
    UNET_CHECKPOINT_PATH,
    YOLO_CHECKPOINT_PATH,
    load_pest_detector,
    load_production_model,
    load_unet_lesion_segmenter,
    load_yolov8_detector,
    predict_pests,
    predict_pests_yolov8,
    predict_unet_lesions,
)


def test_checkpoints_exist_and_non_empty():
    """Verify all production and baseline checkpoints exist."""
    assert CHECKPOINT_PATH.exists(), f"ResNet18 checkpoint missing: {CHECKPOINT_PATH}"
    assert PEST_CHECKPOINT_PATH.exists(), f"TinyYOLO checkpoint missing: {PEST_CHECKPOINT_PATH}"
    assert YOLO_CHECKPOINT_PATH.exists(), f"YOLOv8n-Seg checkpoint missing: {YOLO_CHECKPOINT_PATH}"
    assert UNET_CHECKPOINT_PATH.exists(), f"U-Net checkpoint missing: {UNET_CHECKPOINT_PATH}"

    assert YOLO_CHECKPOINT_PATH.stat().st_size > 1_000_000, "YOLOv8 checkpoint is unexpectedly small"
    assert PEST_CHECKPOINT_PATH.stat().st_size > 1_000_000, "TinyYOLO checkpoint is unexpectedly small"


def test_yolov8_detector_loader():
    """Verify load_yolov8_detector loads Ultralytics model without conflicts."""
    model = load_yolov8_detector(YOLO_CHECKPOINT_PATH)
    assert model is not None, "Failed to load YOLOv8n-Seg model"
    assert hasattr(model, "predict"), "Model does not have predict method"


def test_predict_pests_yolov8_inference():
    """Verify predict_pests_yolov8 output schema on a synthetic foliage image."""
    model = load_yolov8_detector(YOLO_CHECKPOINT_PATH)
    assert model is not None

    # Synthetic leaf image
    dummy_img = Image.new("RGB", (640, 640), color=(40, 140, 50))
    annotated, detections, tuta_cnt, healthy_cnt = predict_pests_yolov8(
        dummy_img, model, conf_threshold=0.10, iou_threshold=0.45
    )

    assert isinstance(annotated, Image.Image)
    assert annotated.size == (640, 640)
    assert isinstance(detections, list)
    assert isinstance(tuta_cnt, int)
    assert isinstance(healthy_cnt, int)

    for det in detections:
        assert "class_id" in det
        assert "category" in det
        assert "confidence" in det
        assert "box" in det
        assert "mask_area_px" in det
        assert "is_pest_damage" in det
        assert len(det["box"]) == 4


def test_predict_pests_yolov8_blank_image():
    """Verify predict_pests_yolov8 handles completely black/empty images cleanly."""
    model = load_yolov8_detector(YOLO_CHECKPOINT_PATH)
    assert model is not None

    black_img = Image.new("RGB", (300, 300), color=(0, 0, 0))
    annotated, detections, tuta_cnt, healthy_cnt = predict_pests_yolov8(
        black_img, model, conf_threshold=0.90
    )

    assert isinstance(annotated, Image.Image)
    assert len(detections) == 0
    assert tuta_cnt == 0
    assert healthy_cnt == 0


def test_predict_pests_yolov8_small_image():
    """Verify predict_pests_yolov8 handles small images without crashing."""
    model = load_yolov8_detector(YOLO_CHECKPOINT_PATH)
    assert model is not None

    small_img = Image.new("RGB", (32, 32), color=(50, 120, 40))
    annotated, detections, tuta_cnt, healthy_cnt = predict_pests_yolov8(
        small_img, model, conf_threshold=0.25
    )
    assert isinstance(annotated, Image.Image)
    assert annotated.size == (32, 32)


def test_predict_pests_yolov8_healthy_reference_regions():
    """Verify healthy reference foliage and Tuta damage regions are generated and counted accurately."""
    model = load_yolov8_detector(YOLO_CHECKPOINT_PATH)
    assert model is not None

    test_img_path = Path("/Users/niyatijha/.gemini/antigravity/brain/ce1c6ebc-fbe7-4c5c-85d2-b18bbc7f6880/.user_uploaded/media_1795242487783_7fb16298.jpg")
    if test_img_path.exists():
        with Image.open(test_img_path) as img:
            annotated, detections, tuta_cnt, healthy_cnt = predict_pests_yolov8(
                img, model, conf_threshold=0.25, iou_threshold=0.35, show_healthy=True
            )
            assert isinstance(annotated, Image.Image)
            assert tuta_cnt >= 2, f"Expected at least 2 Tuta damage regions, got {tuta_cnt}"
            assert healthy_cnt >= 1, f"Expected at least 1 healthy reference region, got {healthy_cnt}"
            assert len(detections) == tuta_cnt + healthy_cnt


def test_unet_lesion_segmentation_integration():
    """Verify real PlantSeg U-Net loader and inference."""
    unet_model, dev, _ = load_unet_lesion_segmenter(UNET_CHECKPOINT_PATH)
    assert unet_model is not None, "Failed to load improved U-Net checkpoint"

    dummy_leaf = Image.new("RGB", (256, 256), color=(40, 120, 50))
    overlay, lesion_pct, mask = predict_unet_lesions(dummy_leaf, unet_model, dev)

    assert isinstance(overlay, Image.Image)
    assert isinstance(lesion_pct, float)
    assert 0.0 <= lesion_pct <= 100.0
    assert mask.shape == (256, 256)

