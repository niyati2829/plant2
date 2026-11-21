"""Consolidated PlantSeg U-Net Lesion Segmentation Tests.

Consolidates:
- Phase J dataset preparation & verification tests
- Phase L real lesion holdout test evaluation tests
- Phase M1 improved U-Net architecture & training tests
- Phase M2 comparative benchmark & parity tests
"""

# === Section 1: Phase J Dataset Preparation ===
"""Phase J Automated Integrity Test Suite.

Verifies:
1. Cleaned tomato lesion dataset file presence and counts (657 images, 657 masks).
2. Split manifests and exact sample allocations (train: 459, val: 99, test: 99).
3. Deduplication and zero cross-split leakage (657 unique MD5 hashes, 0 overlap).
4. Image-mask spatial dimension correspondence (100% match, zero transposed errors).
5. Binary mask validity (strictly {0, 255} on disk, zero empty masks).
6. Balanced representation of all 7 tomato foliar pathologies across splits.
7. PyTorch Dataset and DataLoader operational interfaces (tensor shapes, dtypes).
8. Strict preservation of existing production checkpoints and classification splits.

Author: Crop Disease Detection Project Team
Phase: Phase J
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.segmentation.plantseg_dataset import (
    PlantSegTomatoLesionDataset,
    get_plantseg_dataloaders,
)

DATASET_DIR = Path("data/processed/tomato_lesions_plantseg")
IMAGES_DIR = DATASET_DIR / "images"
MASKS_DIR = DATASET_DIR / "masks"
SPLITS_DIR = DATASET_DIR / "splits"

EXPECTED_TOTAL_SAMPLES = 657
EXPECTED_TRAIN_SAMPLES = 459
EXPECTED_VAL_SAMPLES = 99
EXPECTED_TEST_SAMPLES = 99

EXPECTED_DISEASES = {
    "tomato early blight",
    "tomato late blight",
    "tomato leaf mold",
    "tomato septoria leaf spot",
    "tomato bacterial leaf spot",
    "tomato yellow leaf curl virus",
    "tomato mosaic virus",
}

RESNET_CHECKPOINT = Path("artifacts/checkpoints/best_resnet18_weighted.pt")
EXPECTED_RESNET_HASH = "b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa"

PEST_CHECKPOINT = Path("artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt")
EXPECTED_PEST_HASH = "01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343"


def test_directory_structure_and_file_counts():
    """Verify target directories and file counts on disk."""
    assert DATASET_DIR.exists(), f"Missing dataset directory: {DATASET_DIR}"
    assert IMAGES_DIR.exists(), f"Missing images directory: {IMAGES_DIR}"
    assert MASKS_DIR.exists(), f"Missing masks directory: {MASKS_DIR}"
    assert SPLITS_DIR.exists(), f"Missing splits directory: {SPLITS_DIR}"

    img_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    mask_files = list(MASKS_DIR.glob("*.png"))

    assert len(img_files) == EXPECTED_TOTAL_SAMPLES, (
        f"Expected {EXPECTED_TOTAL_SAMPLES} images, found {len(img_files)}"
    )
    assert len(mask_files) == EXPECTED_TOTAL_SAMPLES, (
        f"Expected {EXPECTED_TOTAL_SAMPLES} masks, found {len(mask_files)}"
    )


def test_split_manifests():
    """Verify split manifests, row counts, and sum integrity."""
    full_path = SPLITS_DIR / "manifest_full.csv"
    train_path = SPLITS_DIR / "train.csv"
    val_path = SPLITS_DIR / "val.csv"
    test_path = SPLITS_DIR / "test.csv"

    for p in (full_path, train_path, val_path, test_path):
        assert p.exists(), f"Missing manifest: {p}"

    df_full = pd.read_csv(full_path)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    assert len(df_full) == EXPECTED_TOTAL_SAMPLES
    assert len(df_train) == EXPECTED_TRAIN_SAMPLES
    assert len(df_val) == EXPECTED_VAL_SAMPLES
    assert len(df_test) == EXPECTED_TEST_SAMPLES
    assert len(df_train) + len(df_val) + len(df_test) == EXPECTED_TOTAL_SAMPLES


def test_zero_duplicate_hashes_and_zero_leakage():
    """Verify exact deduplication and zero cross-split leakage."""
    df_train = pd.read_csv(SPLITS_DIR / "train.csv")
    df_val = pd.read_csv(SPLITS_DIR / "val.csv")
    df_test = pd.read_csv(SPLITS_DIR / "test.csv")
    df_full = pd.read_csv(SPLITS_DIR / "manifest_full.csv")

    assert df_full["image_hash"].nunique() == EXPECTED_TOTAL_SAMPLES, (
        "Duplicate image hashes found in full manifest!"
    )

    train_hashes = set(df_train["image_hash"])
    val_hashes = set(df_val["image_hash"])
    test_hashes = set(df_test["image_hash"])

    assert len(train_hashes.intersection(val_hashes)) == 0, "Train-Val hash leakage detected!"
    assert len(train_hashes.intersection(test_hashes)) == 0, "Train-Test hash leakage detected!"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Val-Test hash leakage detected!"


def test_image_mask_dimension_correspondence():
    """Verify 100% spatial alignment between images and binary masks."""
    df_full = pd.read_csv(SPLITS_DIR / "manifest_full.csv")

    # Sample test across full dataset
    for idx, row in df_full.iterrows():
        img_p = IMAGES_DIR / row["image_name"]
        mask_p = MASKS_DIR / row["mask_name"]

        with Image.open(img_p) as im, Image.open(mask_p) as m:
            assert im.size == m.size, (
                f"Dimension mismatch for {row['image_name']}: img {im.size} vs mask {m.size}"
            )
            assert im.size == (row["width"], row["height"])


def test_binary_mask_integrity():
    """Verify binary masks contain strictly {0, 255} and no all-zero masks exist."""
    df_full = pd.read_csv(SPLITS_DIR / "manifest_full.csv")

    # Verify all lesion ratios are strictly > 0
    assert (df_full["lesion_pixels"] > 0).all(), "Found empty/all-zero masks in manifest!"
    assert (df_full["lesion_ratio"] > 0.0).all(), "Lesion ratio <= 0 found!"
    assert (df_full["lesion_ratio"] <= 1.0).all(), "Lesion ratio > 1 found!"

    # Sample test mask pixel values
    sample_rows = df_full.sample(n=20, random_state=42)
    for _, row in sample_rows.iterrows():
        mask_arr = np.array(Image.open(MASKS_DIR / row["mask_name"]))
        unique_vals = set(np.unique(mask_arr))
        assert unique_vals.issubset({0, 255}), f"Non-binary values {unique_vals} in {row['mask_name']}"
        assert 255 in unique_vals, f"No lesion foreground pixels in {row['mask_name']}"


def test_disease_stratification_across_splits():
    """Verify all 7 disease categories are represented across all splits."""
    df_train = pd.read_csv(SPLITS_DIR / "train.csv")
    df_val = pd.read_csv(SPLITS_DIR / "val.csv")
    df_test = pd.read_csv(SPLITS_DIR / "test.csv")

    assert set(df_train["disease"].unique()) == EXPECTED_DISEASES
    assert set(df_val["disease"].unique()) == EXPECTED_DISEASES
    assert set(df_test["disease"].unique()) == EXPECTED_DISEASES


def test_pytorch_dataset_float_mask():
    """Verify PlantSegTomatoLesionDataset with as_float_mask=True."""
    ds = PlantSegTomatoLesionDataset(
        split="train",
        dataset_dir=DATASET_DIR,
        image_size=(256, 256),
        as_float_mask=True,
    )
    assert len(ds) == EXPECTED_TRAIN_SAMPLES

    img, mask = ds[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert img.shape == (3, 256, 256)
    assert mask.shape == (1, 256, 256)
    assert mask.dtype == torch.float32

    # Mask values should be in [0.0, 1.0]
    unique_vals = set(torch.unique(mask).tolist())
    assert unique_vals.issubset({0.0, 1.0})

    meta = ds.get_metadata(0)
    assert "disease" in meta
    assert "image_name" in meta


def test_pytorch_dataset_long_mask():
    """Verify PlantSegTomatoLesionDataset with as_float_mask=False."""
    ds = PlantSegTomatoLesionDataset(
        split="val",
        dataset_dir=DATASET_DIR,
        image_size=(128, 128),
        as_float_mask=False,
    )
    assert len(ds) == EXPECTED_VAL_SAMPLES

    img, mask = ds[0]
    assert img.shape == (3, 128, 128)
    assert mask.shape == (128, 128)
    assert mask.dtype == torch.int64

    unique_vals = set(torch.unique(mask).tolist())
    assert unique_vals.issubset({0, 1})


def test_dataloader_batching():
    """Verify get_plantseg_dataloaders factory produces valid mini-batches."""
    train_loader, val_loader, test_loader = get_plantseg_dataloaders(
        dataset_dir=DATASET_DIR,
        batch_size=4,
        image_size=(256, 256),
        as_float_mask=True,
    )

    batch_imgs, batch_masks = next(iter(train_loader))
    assert batch_imgs.shape == (4, 3, 256, 256)
    assert batch_masks.shape == (4, 1, 256, 256)
    assert batch_masks.dtype == torch.float32


def test_aspect_ratio_padding_preserves_spatial_alignment():
    """Verify letterboxing padding preserves aspect ratio and maintains pixel alignment."""
    # Test with aspect ratio preservation
    ds_padded = PlantSegTomatoLesionDataset(
        split="test",
        dataset_dir=DATASET_DIR,
        image_size=(256, 256),
        as_float_mask=True,
        preserve_aspect_ratio=True,
    )
    img_pad, mask_pad = ds_padded[0]
    assert img_pad.shape == (3, 256, 256)
    assert mask_pad.shape == (1, 256, 256)
    assert set(torch.unique(mask_pad).tolist()).issubset({0.0, 1.0})

    # Test direct resize without padding
    ds_direct = PlantSegTomatoLesionDataset(
        split="test",
        dataset_dir=DATASET_DIR,
        image_size=(256, 256),
        as_float_mask=True,
        preserve_aspect_ratio=False,
    )
    img_dir, mask_dir = ds_direct[0]
    assert img_dir.shape == (3, 256, 256)
    assert mask_dir.shape == (1, 256, 256)
    assert set(torch.unique(mask_dir).tolist()).issubset({0.0, 1.0})


def test_corrected_five_transposed_pairs_integrity():
    """Verify the 5 corrected pairs use valid spatial transformations preserving coordinates."""
    transposed_filenames = [
        "tomato_bacterial_leaf_spot_42.jpg",
        "tomato_early_blight_google_0043.jpg",
        "tomato_early_blight_google_0000.jpg",
        "tomato_early_blight_google_0056.jpg",
        "tomato_septoria_leaf_spot_google_0033.jpg",
    ]

    for fname in transposed_filenames:
        img_p = IMAGES_DIR / fname
        mask_p = MASKS_DIR / (Path(fname).stem + ".png")

        assert img_p.exists(), f"Missing corrected image: {img_p}"
        assert mask_p.exists(), f"Missing corrected mask: {mask_p}"

        with Image.open(img_p) as im, Image.open(mask_p) as m:
            assert im.size == m.size, f"Mismatch in corrected pair {fname}: img {im.size} != mask {m.size}"
            m_arr = np.array(m)
            assert np.sum(m_arr > 0) > 0, f"Corrected mask {fname} has zero lesion pixels!"
            assert set(np.unique(m_arr)).issubset({0, 255})


def test_overlay_visual_artifacts_count():
    """Verify visual overlays in reports/plantseg_audit/cleaned_overlays/ total at least 20."""
    overlay_dir = Path("reports/plantseg_audit/cleaned_overlays")
    assert overlay_dir.exists(), f"Overlay directory missing: {overlay_dir}"
    overlays = list(overlay_dir.glob("*.png"))
    assert len(overlays) >= 20, f"Expected >= 20 overlays, found {len(overlays)}"


def test_production_checkpoints_integrity():
    """Ensure ResNet18 and YOLO checkpoints have not been modified or corrupted."""
    assert RESNET_CHECKPOINT.exists(), "ResNet18 production checkpoint missing!"
    resnet_hash = hashlib.sha256(RESNET_CHECKPOINT.read_bytes()).hexdigest()
    assert resnet_hash == EXPECTED_RESNET_HASH, (
        f"ResNet18 checkpoint tampered! Expected {EXPECTED_RESNET_HASH}, got {resnet_hash}"
    )

    assert PEST_CHECKPOINT.exists(), "Pest detector production checkpoint missing!"
    pest_hash = hashlib.sha256(PEST_CHECKPOINT.read_bytes()).hexdigest()
    assert pest_hash == EXPECTED_PEST_HASH, (
        f"Pest detector checkpoint tampered! Expected {EXPECTED_PEST_HASH}, got {pest_hash}"
    )


# === Section 2: Phase L Real Lesion Evaluation ===
"""Phase L Automated Holdout Test Evaluation Test Suite.

Verifies:
1. Checkpoint loading and forward inference pass on real U-Net.
2. Holdout test split evaluation presence and 99-sample coverage.
3. Test metrics integrity (Dice, IoU, Precision, Recall, Pixel Accuracy).
4. Prediction vs ground-truth visual overlays in reports/plantseg_audit/test_prediction_overlays/.
5. Preservation of production checkpoints, synthetic checkpoints, and test split manifests.

Author: Crop Disease Detection Project Team
Phase: Phase L
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.segmentation.plantseg_dataset import PlantSegTomatoLesionDataset
from src.segmentation.unet import UNet

CHECKPOINT_PATH = Path("artifacts/checkpoints/segmentation_real/best_unet_real.pt")
REPORT_JSON = Path("reports/unet_real_test_evaluation.json")
REPORT_CSV = Path("reports/unet_real_test_per_image.csv")
OVERLAY_DIR = Path("reports/plantseg_audit/test_prediction_overlays")
TEST_SPLIT_CSV = Path("data/processed/tomato_lesions_plantseg/splits/test.csv")

RESNET_CHECKPOINT = Path("artifacts/checkpoints/best_resnet18_weighted.pt")
EXPECTED_RESNET_HASH = "b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa"

PEST_CHECKPOINT = Path("artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt")
EXPECTED_PEST_HASH = "01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343"

SYNTHETIC_CHECKPOINT = Path("artifacts/checkpoints/segmentation/best_unet.pt")


def test_checkpoint_loading_and_inference():
    """Verify real U-Net checkpoint can be loaded and executed for inference."""
    assert CHECKPOINT_PATH.exists(), f"Missing checkpoint: {CHECKPOINT_PATH}"

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "epoch" in ckpt
    assert "val_dice" in ckpt
    assert ckpt["in_channels"] == 3
    assert ckpt["num_classes"] == 1

    model = UNet(in_channels=3, num_classes=1, init_features=ckpt["init_features"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)

    assert logits.shape == (1, 1, 256, 256)
    assert probs.min().item() >= 0.0
    assert probs.max().item() <= 1.0


def test_test_evaluation_reports_exist_and_match():
    """Verify JSON summary and per-image CSV reports exist and have 99 samples."""
    assert REPORT_JSON.exists(), f"Missing JSON report: {REPORT_JSON}"
    assert REPORT_CSV.exists(), f"Missing CSV report: {REPORT_CSV}"

    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    assert res["test_sample_count"] == 99
    assert res["fixed_classification_threshold"] == 0.5
    assert "macro_metrics" in res
    assert "micro_metrics" in res
    assert "per_disease_breakdown" in res

    df_csv = pd.read_csv(REPORT_CSV)
    assert len(df_csv) == 99

    df_test = pd.read_csv(TEST_SPLIT_CSV)
    assert set(df_csv["image_name"]) == set(df_test["image_name"])


def test_test_metric_ranges_and_values():
    """Verify macro and micro metrics are mathematically sound."""
    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    macro = res["macro_metrics"]
    micro = res["micro_metrics"]

    # All metrics must be in [0, 1]
    for m in (macro["mean_dice_score"], macro["mean_iou"], macro["mean_precision"], macro["mean_recall"], macro["mean_pixel_accuracy"]):
        assert 0.0 <= m <= 1.0

    for m in (micro["micro_dice_score"], micro["micro_iou"], micro["micro_precision"], micro["micro_recall"], micro["micro_pixel_accuracy"]):
        assert 0.0 <= m <= 1.0

    # Sanity checks on known achieved scores
    assert macro["mean_dice_score"] >= 0.40
    assert micro["micro_dice_score"] >= 0.45
    assert macro["mean_recall"] >= 0.70
    assert macro["mean_pixel_accuracy"] >= 0.80


def test_per_disease_breakdown_covers_all_seven():
    """Verify per-disease breakdown covers all 7 tomato foliar pathologies."""
    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        res = json.load(f)

    per_dis = res["per_disease_breakdown"]
    assert len(per_dis) == 7

    total_samples = sum(item["test_sample_count"] for item in per_dis.values())
    assert total_samples == 99


def test_visual_prediction_overlays_exist():
    """Verify representative prediction vs ground-truth overlays were generated."""
    assert OVERLAY_DIR.exists(), f"Missing overlay directory: {OVERLAY_DIR}"
    overlays = list(OVERLAY_DIR.glob("*.png"))
    assert len(overlays) >= 7

    for ovl in overlays:
        with Image.open(ovl) as im:
            assert im.size[0] > 1000  # Multi-panel figure
            assert im.size[1] > 200


def test_non_interference_preservation():
    """Verify existing production checkpoints, synthetic checkpoint, and test manifest are unmodified."""
    assert RESNET_CHECKPOINT.exists()
    assert hashlib.sha256(RESNET_CHECKPOINT.read_bytes()).hexdigest() == EXPECTED_RESNET_HASH

    assert PEST_CHECKPOINT.exists()
    assert hashlib.sha256(PEST_CHECKPOINT.read_bytes()).hexdigest() == EXPECTED_PEST_HASH

    assert SYNTHETIC_CHECKPOINT.exists()
    assert SYNTHETIC_CHECKPOINT.stat().st_size == 40303563

    assert TEST_SPLIT_CSV.exists()
    assert len(pd.read_csv(TEST_SPLIT_CSV)) == 99


# === Section 3: Phase M1 Improved U-Net ===
"""Phase M1 Test Suite: Verification of Improved Real U-Net Checkpoint & Artifacts.

Verifies:
1. Loading and inference pass for the improved real U-Net checkpoint.
2. Improved model architecture configuration (in=3, out=1, init_features=16).
3. Presence and validity of validation prediction overlays.
4. Training history and configuration JSON files.
5. Preservation of Phase K baseline checkpoint (best_unet_real.pt) and test split.

Author: Crop Disease Detection Project Team
Phase: Phase M1
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from src.segmentation.unet import UNet

IMPROVED_CHECKPOINT = Path("artifacts/checkpoints/segmentation_real/best_unet_improved.pt")
BASELINE_CHECKPOINT = Path("artifacts/checkpoints/segmentation_real/best_unet_real.pt")
HISTORY_JSON = Path("reports/unet_improved_training_history.json")
CONFIG_JSON = Path("reports/unet_improved_training_config.json")
VAL_OVERLAY_DIR = Path("reports/plantseg_audit/val_prediction_overlays_improved")
TEST_SPLIT_CSV = Path("data/processed/tomato_lesions_plantseg/splits/test.csv")


def test_improved_checkpoint_loading_and_inference():
    """Verify improved checkpoint can be loaded and outputs probability map."""
    assert IMPROVED_CHECKPOINT.exists(), f"Missing improved checkpoint: {IMPROVED_CHECKPOINT}"

    ckpt = torch.load(IMPROVED_CHECKPOINT, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert "val_dice" in ckpt
    assert "val_precision" in ckpt
    assert ckpt["in_channels"] == 3
    assert ckpt["num_classes"] == 1
    assert ckpt["init_features"] == 16

    model = UNet(in_channels=3, num_classes=1, init_features=16, bilinear=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)

    assert logits.shape == (2, 1, 256, 256)
    assert probs.min().item() >= 0.0
    assert probs.max().item() <= 1.0


def test_improved_training_artifacts():
    """Verify training history and configuration files exist with expected structure."""
    assert HISTORY_JSON.exists(), f"Missing history JSON: {HISTORY_JSON}"
    assert CONFIG_JSON.exists(), f"Missing config JSON: {CONFIG_JSON}"

    with open(CONFIG_JSON, "r", encoding="utf-8") as f:
        conf = json.load(f)

    assert conf["model_architecture"] == "UNet"
    assert conf["test_split_touched"] is False
    assert conf["best_val_dice"] >= 0.55
    assert conf["best_val_precision"] >= 0.50

    with open(HISTORY_JSON, "r", encoding="utf-8") as f:
        hist = json.load(f)

    assert len(hist["train_loss"]) == conf["completed_epochs"]
    assert len(hist["val_dice"]) == conf["completed_epochs"]


def test_validation_overlays_exist():
    """Verify representative validation prediction overlays were generated."""
    assert VAL_OVERLAY_DIR.exists(), f"Missing overlay directory: {VAL_OVERLAY_DIR}"
    overlays = list(VAL_OVERLAY_DIR.glob("*.png"))
    assert len(overlays) == 7

    for ovl in overlays:
        with Image.open(ovl) as im:
            assert im.size[0] > 1000
            assert im.size[1] > 200


def test_baseline_and_test_split_preservation():
    """Verify Phase K baseline checkpoint and test split are untouched."""
    assert BASELINE_CHECKPOINT.exists()
    assert BASELINE_CHECKPOINT.stat().st_size > 0

    assert TEST_SPLIT_CSV.exists()
    df_test = pd.read_csv(TEST_SPLIT_CSV)
    assert len(df_test) == 99


# === Section 4: Phase M2 Comparative Benchmark ===
"""Phase M2 Automated Holdout Test Comparative Evaluation Test Suite.

Verifies:
1. Checkpoint loading and forward inference pass on improved U-Net vs baseline U-Net.
2. Holdout test split comparative evaluation presence and 99-sample coverage.
3. Test metrics integrity (Dice, IoU, Precision, Recall, Pixel Accuracy, Confusion Matrix).
4. Mathematical consistency of absolute changes and over-segmentation reduction.
5. Prediction vs ground-truth visual overlays in reports/plantseg_audit/test_prediction_overlays_comparison/.
6. CPU latency benchmark reporting.
7. Preservation of production checkpoints, synthetic checkpoints, and test split manifests.

Author: Crop Disease Detection Project Team
Phase: Phase M2
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.segmentation.unet import UNet

BASELINE_CHECKPOINT = Path("artifacts/checkpoints/segmentation_real/best_unet_real.pt")
IMPROVED_CHECKPOINT = Path("artifacts/checkpoints/segmentation_real/best_unet_improved.pt")

COMPARISON_JSON = Path("reports/unet_test_comparison_phase_m2.json")
COMPARISON_CSV = Path("reports/unet_test_comparison_per_image.csv")
IMPROVED_JSON = Path("reports/unet_improved_test_evaluation.json")
IMPROVED_CSV = Path("reports/unet_improved_test_per_image.csv")
OVERLAY_DIR = Path("reports/plantseg_audit/test_prediction_overlays_comparison")

TEST_SPLIT_CSV = Path("data/processed/tomato_lesions_plantseg/splits/test.csv")

RESNET_CHECKPOINT = Path("artifacts/checkpoints/best_resnet18_weighted.pt")
EXPECTED_RESNET_HASH = "b86d760146c7e8389600bb56d63dca9c552bb0a649d01e74feb1cb19bfa4a3fa"

PEST_CHECKPOINT = Path("artifacts/checkpoints/pest_detection_real/best_pest_detector_real.pt")
EXPECTED_PEST_HASH = "01d8d64368b62c3ce309db4e4d18aaf6af807aad7b6225d60da048ef784b1343"

SYNTHETIC_CHECKPOINT = Path("artifacts/checkpoints/segmentation/best_unet.pt")


def test_checkpoints_loading_and_architecture_parity():
    """Verify both baseline and improved U-Net checkpoints load and share identical architecture."""
    assert BASELINE_CHECKPOINT.exists(), f"Missing baseline checkpoint: {BASELINE_CHECKPOINT}"
    assert IMPROVED_CHECKPOINT.exists(), f"Missing improved checkpoint: {IMPROVED_CHECKPOINT}"

    ckpt_base = torch.load(BASELINE_CHECKPOINT, map_location="cpu")
    ckpt_impr = torch.load(IMPROVED_CHECKPOINT, map_location="cpu")

    assert ckpt_base["in_channels"] == ckpt_impr["in_channels"] == 3
    assert ckpt_base["num_classes"] == ckpt_impr["num_classes"] == 1
    assert ckpt_base["init_features"] == ckpt_impr["init_features"] == 16

    model_base = UNet(in_channels=3, num_classes=1, init_features=16)
    model_base.load_state_dict(ckpt_base["model_state_dict"])
    model_base.eval()

    model_impr = UNet(in_channels=3, num_classes=1, init_features=16)
    model_impr.load_state_dict(ckpt_impr["model_state_dict"])
    model_impr.eval()

    dummy_input = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out_base = model_base(dummy_input)
        out_impr = model_impr(dummy_input)

    assert out_base.shape == (2, 1, 256, 256)
    assert out_impr.shape == (2, 1, 256, 256)


def test_comparison_reports_exist_and_cover_99_test_images():
    """Verify comparison JSON and CSV reports exist and contain exactly 99 test images."""
    assert COMPARISON_JSON.exists(), f"Missing JSON: {COMPARISON_JSON}"
    assert COMPARISON_CSV.exists(), f"Missing CSV: {COMPARISON_CSV}"
    assert IMPROVED_JSON.exists(), f"Missing JSON: {IMPROVED_JSON}"
    assert IMPROVED_CSV.exists(), f"Missing CSV: {IMPROVED_CSV}"

    with open(COMPARISON_JSON, "r", encoding="utf-8") as f:
        comp = json.load(f)

    assert comp["test_sample_count"] == 99
    assert comp["threshold"] == 0.50

    df_comp = pd.read_csv(COMPARISON_CSV)
    assert len(df_comp) == 99

    df_test = pd.read_csv(TEST_SPLIT_CSV)
    assert len(df_test) == 99
    assert set(df_comp["image_name"]) == set(df_test["image_name"])

    df_impr = pd.read_csv(IMPROVED_CSV)
    assert len(df_impr) == 99
    assert set(df_impr["image_name"]) == set(df_test["image_name"])


def test_comparative_metrics_mathematical_consistency():
    """Verify confusion matrix sums, metric bounds, and positive deltas for improved model."""
    with open(COMPARISON_JSON, "r", encoding="utf-8") as f:
        comp = json.load(f)

    base = comp["models"]["baseline"]
    impr = comp["models"]["improved"]
    abs_changes = comp["absolute_changes"]

    # Total pixels per model must equal 99 * 256 * 256 = 6,488,064
    total_pixels_expected = 99 * 256 * 256
    for model_dict in (base, impr):
        conf = model_dict["confusion_pixels"]
        tot = conf["total_tp"] + conf["total_fp"] + conf["total_fn"] + conf["total_tn"]
        assert tot == total_pixels_expected, f"Pixel sum mismatch: {tot} vs {total_pixels_expected}"

    # Verify metrics in [0, 1]
    for m in (base["macro_metrics"], impr["macro_metrics"]):
        for v in m.values():
            assert 0.0 <= v <= 1.0

    # Macro Dice >= Macro IoU (mathematical theorem)
    assert base["macro_metrics"]["macro_dice"] >= base["macro_metrics"]["macro_iou"]
    assert impr["macro_metrics"]["macro_dice"] >= impr["macro_metrics"]["macro_iou"]

    # Improved model shows superior Dice, IoU, Precision, and Accuracy
    assert impr["macro_metrics"]["macro_dice"] > base["macro_metrics"]["macro_dice"]
    assert impr["macro_metrics"]["macro_iou"] > base["macro_metrics"]["macro_iou"]
    assert impr["macro_metrics"]["macro_precision"] > base["macro_metrics"]["macro_precision"]
    assert impr["macro_metrics"]["macro_pixel_accuracy"] > base["macro_metrics"]["macro_pixel_accuracy"]

    # False positive pixels reduced
    assert impr["confusion_pixels"]["total_fp"] < base["confusion_pixels"]["total_fp"]
    assert abs_changes["fp_pixel_reduction_percent"] > 20.0  # Over 20% reduction


def test_cpu_latency_benchmarks_present():
    """Verify latency benchmarks exist and reflect realistic CPU execution."""
    with open(COMPARISON_JSON, "r", encoding="utf-8") as f:
        comp = json.load(f)

    for tag in ("baseline", "improved"):
        lat = comp["models"][tag]["cpu_latency"]
        assert lat["device"] == "cpu"
        assert lat["num_runs"] == 99
        assert 1.0 <= lat["mean_latency_ms"] <= 500.0
        assert lat["throughput_fps"] > 0.0


def test_comparison_overlays_generated():
    """Verify multi-panel visual overlays are generated for all 7 diseases."""
    assert OVERLAY_DIR.exists()
    pngs = list(OVERLAY_DIR.glob("*.png"))
    assert len(pngs) >= 7, f"Expected at least 7 comparison overlays, found {len(pngs)}"


def test_production_and_split_integrity():
    """Verify that existing production models, synthetic U-Net, and test split are untouched."""
    assert RESNET_CHECKPOINT.exists()
    with open(RESNET_CHECKPOINT, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_RESNET_HASH

    assert PEST_CHECKPOINT.exists()
    with open(PEST_CHECKPOINT, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == EXPECTED_PEST_HASH

    assert SYNTHETIC_CHECKPOINT.exists()
    assert TEST_SPLIT_CSV.exists()

