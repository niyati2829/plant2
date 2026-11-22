"""Unified Streamlit Web Application for Tomato Disease and Pest Surveillance System.

Phase H/M3 — 4-Page Navigation Interface:
1. 🏠 Home & Overview: Project overview, 4 capability cards, performance summary, consolidated benchmarks, tech stack, datasets, and limitations.
2. 🔬 Disease Classification: Calibrated ResNet18 (T* = 1.3021), top-3 hypotheses, expandable 10-class probabilities, Grad-CAM, and PlantSeg U-Net foliar lesion segmentation.
3. 🐛 Pest Damage Detection: Production YOLOv8n-Seg instance segmentation for Tuta absoluta foliar feeding damage and healthy reference foliage.
4. 📹 Video Surveillance: Real-time multi-task video pipeline (calibrated ResNet18 + YOLOv8n-Seg, HUD telemetry, and temporal disease timeline).

Author: Crop Disease Detection Project Team
"""

import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops, transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.temperature_scaling import ModelWithTemperature
from src.detection.model import TinyYOLODetector
from src.detection.real_pest_dataset import REAL_PEST_CLASSES
from src.interpretability.grad_cam import GradCAM
from src.interpretability.visualization import overlay_heatmap
from src.models.transfer_learning import TomatoResNet18
from src.segmentation.unet import UNet
from src.surveillance.annotator import VideoAnnotator
from src.surveillance.frame_analyzer import (
    TOMATO_CLASSES,
    FrameAnalysisResult,
    SurveillanceFrameAnalyzer,
)
from src.surveillance.timeline_plotter import plot_surveillance_timeline
from src.surveillance.video_processor import VideoMetadata, VideoProcessor
from src.utils.device import get_device, get_device_info

# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)

# Canonical file paths
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "best_resnet18_weighted.pt"
PEST_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "pest_detection_real" / "best_pest_detector_real.pt"
YOLO_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "pest_detection_yolov8" / "best_yolov8n_seg.pt"
UNET_CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation_real" / "best_unet_improved.pt"
CLASS_MAPPING_PATH = PROJECT_ROOT / "data" / "splits" / "class_mapping.json"
SAMPLE_VIDEO_PATH = PROJECT_ROOT / "data" / "video" / "sample_field_surveillance.mp4"
TIMELINE_FIG_PATH = PROJECT_ROOT / "artifacts" / "figures" / "video" / "disease_timeline.png"

# Calibration temperature determined in Phase F
CALIBRATION_TEMPERATURE = 1.3021

DISPLAY_NAME_OVERRIDES = {
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites (Two-Spotted Spider Mite)",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___healthy": "Healthy Foliage",
}


# ---------------------------------------------------------------------------
# Class Mapping & Utilities (Preserved for compatibility with tests)
# ---------------------------------------------------------------------------

@st.cache_data
def load_class_mappings(mapping_file: Path) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    """Load class mapping JSON and return bidirectional lookup tables."""
    if not mapping_file.exists():
        raise FileNotFoundError(f"Class mapping file not found at: {mapping_file}")

    with open(mapping_file, "r", encoding="utf-8") as f:
        class_to_idx: Dict[str, int] = json.load(f)

    sorted_items = sorted(class_to_idx.items(), key=lambda x: x[1])
    class_names_sorted = [item[0] for item in sorted_items]
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    return class_to_idx, idx_to_class, class_names_sorted


# ---------------------------------------------------------------------------
# Cached Model Loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_production_model(checkpoint_file: Path) -> Tuple[nn.Module, torch.device, str]:
    """Load and initialize the frozen final production model once."""
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Production checkpoint not found at: {checkpoint_file}")

    device = get_device(preferred="mps")
    device_info = get_device_info(preferred="mps")
    device_name = device_info.get("device_name", str(device))

    model = TomatoResNet18(num_classes=10)
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    return model, device, device_name


@st.cache_resource(show_spinner=False)
def load_calibrated_classifier(checkpoint_file: Path, temperature: float = CALIBRATION_TEMPERATURE) -> Tuple[ModelWithTemperature, torch.device, str]:
    """Load production model wrapped with post-hoc temperature scaling."""
    base_model, device, device_name = load_production_model(checkpoint_file)
    calibrated_model = ModelWithTemperature(base_model)
    calibrated_model.set_temperature(temperature)
    calibrated_model.eval()
    return calibrated_model, device, device_name


@st.cache_resource(show_spinner=False)
def load_pest_detector(checkpoint_file: Path) -> Tuple[Optional[TinyYOLODetector], torch.device, str]:
    """Load experimental real pest detector if checkpoint exists."""
    if not checkpoint_file.exists():
        return None, torch.device("cpu"), "Unavailable"

    try:
        device = get_device(preferred="mps")
        device_info = get_device_info(preferred="mps")
        device_name = device_info.get("device_name", str(device))

        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        num_classes = checkpoint.get("num_classes", 2)
        grid_size = checkpoint.get("grid_size", 8)
        num_boxes_per_cell = checkpoint.get("num_boxes_per_cell", 1)
        detector = TinyYOLODetector(
            num_classes=num_classes,
            grid_size=grid_size,
            num_boxes_per_cell=num_boxes_per_cell,
        )
        detector.load_state_dict(checkpoint["model_state_dict"])
        detector.to(device)
        detector.eval()

        for p in detector.parameters():
            p.requires_grad = False

        return detector, device, device_name
    except Exception:
        return None, torch.device("cpu"), "Unavailable"


@st.cache_resource(show_spinner=False)
def load_yolov8_detector(checkpoint_file: Path) -> Optional[Any]:
    """Load production YOLOv8n-Seg model for Tuta absoluta damage segmentation."""
    if not checkpoint_file.exists():
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(str(checkpoint_file))
        return model
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_unet_lesion_segmenter(checkpoint_file: Path) -> Tuple[Optional[nn.Module], torch.device, str]:
    """Load improved real U-Net model trained on PlantSeg for lesion segmentation."""
    if not checkpoint_file.exists():
        return None, torch.device("cpu"), "Unavailable"
    try:
        from src.segmentation.unet import UNet
        device = get_device(preferred="cpu")
        device_info = get_device_info(preferred="cpu")
        device_name = device_info.get("device_name", str(device))

        state = torch.load(checkpoint_file, map_location=device, weights_only=False)
        in_channels = state.get("in_channels", 3)
        num_classes = state.get("num_classes", 1)
        init_features = state.get("init_features", 16)
        model = UNet(in_channels=in_channels, num_classes=num_classes, init_features=init_features)

        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model, device, device_name
    except Exception:
        return None, torch.device("cpu"), "Unavailable"


# ---------------------------------------------------------------------------
# Preprocessing & Inference Pipelines
# ---------------------------------------------------------------------------

def get_inference_transforms() -> transforms.Compose:
    """Deterministic evaluation transforms identical to validation/test pipeline."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def predict_leaf(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
    idx_to_class: Dict[int, str],
    use_temperature: bool = True,
    temperature: float = CALIBRATION_TEMPERATURE,
    top_k: Optional[int] = 3,
) -> List[Dict[str, Any]]:
    """Run model inference and return predictions with calibrated probabilities."""
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    rgb_image = image.convert("RGB")
    image_tensor = transform(rgb_image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        if use_temperature:
            logits = logits / float(temperature)
        probabilities = torch.softmax(logits, dim=1)[0]
        k = len(idx_to_class) if (top_k is None or top_k >= len(idx_to_class)) else max(1, top_k)
        top_probs, top_indices = torch.topk(probabilities, k=min(k, probabilities.shape[0]))

    results = []
    for rank in range(len(top_indices)):
        idx = top_indices[rank].item()
        prob = top_probs[rank].item()
        raw_name = idx_to_class.get(idx, f"Class_{idx}")
        display_name = DISPLAY_NAME_OVERRIDES.get(raw_name, raw_name.replace("Tomato___", "").replace("_", " "))
        is_healthy = ("healthy" in raw_name.lower())

        results.append({
            "rank": rank + 1,
            "raw_class": raw_name,
            "display_name": display_name,
            "probability": prob,
            "percentage": prob * 100.0,
            "is_healthy": is_healthy,
        })

    return results


def run_gradcam_overlay(
    image: Image.Image,
    model: TomatoResNet18,
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Generate Grad-CAM activation heatmap and blended overlay."""
    device = next(model.parameters()).device
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    rgb_image = image.convert("RGB")
    tensor = transform(rgb_image).unsqueeze(0).to(device)
    tensor.requires_grad = True

    target_layer = model.layer4[-1]
    with GradCAM(model, target_layer) as cam:
        heatmap, pred_c, conf = cam.generate_heatmap(tensor, target_class=target_class)

    img_224 = np.array(rgb_image.resize((224, 224), Image.BILINEAR))
    blended = overlay_heatmap(img_224, heatmap, alpha=0.50)

    return heatmap, blended, pred_c, conf


def predict_pests(
    image: Image.Image,
    detector: TinyYOLODetector,
    device: torch.device,
    conf_threshold: float = 0.20,
    iou_threshold: float = 0.50,
) -> Tuple[Image.Image, List[Dict[str, Any]], int, int]:
    """Run experimental pest detection and render annotations (internal benchmark compatibility)."""
    orig_w, orig_h = image.size
    rgb_image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    resized_pil = rgb_image.resize((256, 256), Image.BILINEAR)
    tensor = transform(resized_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = detector.predict_boxes(tensor, conf_threshold=conf_threshold, iou_threshold=iou_threshold)[0]

    boxes_256 = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()
    labels = preds["labels"].cpu().numpy()

    annotated = rgb_image.copy()
    draw = ImageDraw.Draw(annotated)

    detections = []
    tuta_count = 0
    healthy_count = 0

    scale_x = orig_w / 256.0
    scale_y = orig_h / 256.0

    for b, s, l in zip(boxes_256, scores, labels):
        cls_id = int(l)
        conf_val = float(s)

        x1 = max(0.0, float(b[0] * scale_x))
        y1 = max(0.0, float(b[1] * scale_y))
        x2 = min(float(orig_w), float(b[2] * scale_x))
        y2 = min(float(orig_h), float(b[3] * scale_y))

        is_tuta = (cls_id == 1)
        if is_tuta:
            tuta_count += 1
            color = (230, 40, 40)
            label = f"Tuta Damage {conf_val:.2f}"
            cat_name = "Tuta absoluta Feeding Damage"
        else:
            healthy_count += 1
            color = (34, 197, 94)
            label = f"Healthy Foliage {conf_val:.2f}"
            cat_name = "Healthy Reference Foliage"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text_w = len(label) * 8 + 6
        text_h = 18
        draw.rectangle([x1, max(0, y1 - text_h), min(float(orig_w), x1 + text_w), y1], fill=color)
        draw.text((x1 + 3, max(0, y1 - text_h + 2)), label, fill=(255, 255, 255))

        detections.append({
            "class_id": cls_id,
            "category": cat_name,
            "confidence": round(conf_val, 4),
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "is_pest_damage": is_tuta,
        })

    return annotated, detections, tuta_count, healthy_count


def predict_pests_yolov8(
    image: Image.Image,
    model: Any,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    show_healthy: bool = True,
) -> Tuple[Image.Image, List[Dict[str, Any]], int, int]:
    """Run production YOLOv8n-Seg instance segmentation for Tuta absoluta damage and healthy foliage reference."""
    rgb_image = image.convert("RGB")
    orig_w, orig_h = rgb_image.size
    np_img = np.array(rgb_image)

    # Multi-scale inference: evaluates at base resolution (640) and native aspect resolution
    # to accurately preserve both broader foliage context and tiny larval feeding mines.
    scales = [640]
    max_dim = max(orig_w, orig_h)
    if max_dim > 700:
        target_sz = min(1024, int(round(max_dim / 32.0) * 32))
        if target_sz not in scales:
            scales.append(target_sz)

    run_conf = min(0.05, float(conf_threshold))
    candidates = []

    for sz in scales:
        results = model.predict(rgb_image, imgsz=sz, conf=run_conf, iou=iou_threshold, retina_masks=True, verbose=False)[0]
        if results.boxes is None or len(results.boxes) == 0:
            continue
        boxes_xyxy = results.boxes.xyxy.cpu()
        confs = results.boxes.conf.cpu()
        clss = results.boxes.cls.cpu()
        raw_masks = results.masks.xy if (results.masks is not None and getattr(results.masks, "xy", None) is not None) else []

        for i in range(len(boxes_xyxy)):
            m = raw_masks[i] if i < len(raw_masks) and raw_masks[i] is not None and len(raw_masks[i]) >= 3 else None
            candidates.append({
                "box": boxes_xyxy[i],
                "conf": float(confs[i]),
                "cls": int(clss[i]),
                "mask": m,
            })

    detections: List[Dict[str, Any]] = []
    tuta_count = 0
    healthy_region_count = 0

    if not candidates:
        return rgb_image, detections, tuta_count, healthy_region_count

    # Multi-scale deduplication using torchvision batched NMS per class
    all_b = torch.stack([c["box"] for c in candidates])
    all_s = torch.tensor([c["conf"] for c in candidates])
    all_c = torch.tensor([c["cls"] for c in candidates])

    keep_indices = ops.batched_nms(all_b, all_s, all_c, float(iou_threshold)).tolist()
    deduped = [candidates[i] for i in keep_indices]

    active_tuta = []
    active_healthy = []
    near_miss_tuta = []

    # Calibrate healthy reference confidence threshold independently from pest damage slider
    # to guarantee reference healthy lamina is not discarded by pest-specific confidence thresholding,
    # while strictly respecting high-confidence filters when explicitly configured.
    if float(conf_threshold) >= 0.50:
        healthy_conf_threshold = float(conf_threshold)
    else:
        healthy_conf_threshold = max(0.10, min(0.18, float(conf_threshold) * 0.75))

    for item in deduped:
        c_id = item["cls"]
        s_val = item["conf"]
        b_np = item["box"].numpy()
        m = item["mask"]
        entry = {"box": b_np, "conf": s_val, "cls": c_id, "mask": m}

        if c_id == 1:
            if s_val >= conf_threshold:
                active_tuta.append(entry)
            else:
                near_miss_tuta.append(entry)
        else:
            if s_val >= healthy_conf_threshold:
                active_healthy.append(entry)

    if "st" in globals() or "streamlit" in sys.modules:
        try:
            st.session_state["near_miss_tuta"] = near_miss_tuta
        except Exception:
            pass

    # Suppress healthy reference boxes ONLY if they are heavily covered by active Tuta damage (>60% of healthy box) or high IoU
    filtered_healthy = []
    for h_item in active_healthy:
        h_box = h_item["box"]
        h_area = max(1.0, float((h_box[2] - h_box[0]) * (h_box[3] - h_box[1])))
        suppressed = False
        for t_item in active_tuta:
            t_box = t_item["box"]
            ix1 = max(h_box[0], t_box[0])
            iy1 = max(h_box[1], t_box[1])
            ix2 = min(h_box[2], t_box[2])
            iy2 = min(h_box[3], t_box[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            t_area = max(1.0, float((t_box[2] - t_box[0]) * (t_box[3] - t_box[1])))
            iou = inter / (h_area + t_area - inter)
            if (inter / h_area) > 0.60 or iou > 0.50:
                suppressed = True
                break
        if not suppressed:
            filtered_healthy.append(h_item)

    display_items = []
    if show_healthy:
        display_items.extend(filtered_healthy)
    display_items.extend(active_tuta)

    if not display_items:
        return rgb_image, detections, 0, 0

    overlay = np_img.copy()
    has_polys = False

    # 1. First pass: Draw filled polygon masks on overlay
    for item in display_items:
        is_tuta = (item["cls"] == 1)
        m = item["mask"]
        fill_color = (230, 40, 40) if is_tuta else (34, 197, 94)
        if m is not None and len(m) >= 3:
            pts = np.asarray(m, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], fill_color)
            has_polys = True

    if has_polys:
        blended_np = cv2.addWeighted(overlay, 0.45, np_img, 0.55, 0)
        # Draw high-contrast polygon boundary contours
        for item in display_items:
            is_tuta = (item["cls"] == 1)
            m = item["mask"]
            contour_color = (255, 30, 30) if is_tuta else (0, 230, 100)
            if m is not None and len(m) >= 3:
                pts = np.asarray(m, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(blended_np, [pts], isClosed=True, color=contour_color, thickness=2)
        annotated = Image.fromarray(blended_np)
    else:
        annotated = rgb_image.copy()

    draw = ImageDraw.Draw(annotated)

    # 2. Second pass: Draw bounding boxes and labels
    for item in display_items:
        b = item["box"]
        conf_val = item["conf"]
        c_id = item["cls"]
        m = item["mask"]
        is_tuta = (c_id == 1)

        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(orig_w), x2), min(float(orig_h), y2)

        mask_area_px = 0.0
        if m is not None and len(m) >= 3:
            mask_area_px = float(cv2.contourArea(np.asarray(m, dtype=np.float32)))

        if is_tuta:
            tuta_count += 1
            box_color = (230, 40, 40)
            label_text = f"Tuta Damage {conf_val:.2f}"
            category_label = "Tuta absoluta Feeding Damage"
        else:
            healthy_region_count += 1
            box_color = (34, 197, 94)
            label_text = f"Healthy Foliage {conf_val:.2f}"
            category_label = "Healthy Foliage Region"

        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
        text_w = len(label_text) * 8 + 6
        text_h = 18
        draw.rectangle([x1, max(0, y1 - text_h), min(float(orig_w), x1 + text_w), y1], fill=box_color)
        draw.text((x1 + 3, max(0, y1 - text_h + 2)), label_text, fill=(255, 255, 255))

        detections.append({
            "class_id": c_id,
            "category": category_label,
            "class_name": category_label,
            "confidence": round(conf_val, 4),
            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "mask_area_px": round(mask_area_px, 1),
            "is_pest_damage": is_tuta,
        })

    return annotated, detections, tuta_count, healthy_region_count


def predict_unet_lesions(
    image: Image.Image,
    model: nn.Module,
    device: torch.device,
    threshold: float = 0.50,
) -> Tuple[Image.Image, float, np.ndarray]:
    """Run real U-Net inference on tomato leaf to segment disease lesions."""
    rgb_image = image.convert("RGB")
    orig_w, orig_h = rgb_image.size
    np_img = np.array(rgb_image)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(rgb_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask_256 = (probs >= threshold).astype(np.uint8)
    mask_orig = cv2.resize(mask_256, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    lesion_pixels = int(np.sum(mask_orig))
    total_pixels = orig_w * orig_h
    lesion_pct = (lesion_pixels / total_pixels) * 100.0

    overlay = np_img.copy()
    overlay[mask_orig > 0] = [230, 50, 50]
    blended = cv2.addWeighted(overlay, 0.45, np_img, 0.55, 0)
    annotated = Image.fromarray(blended)

    return annotated, lesion_pct, mask_orig


# ---------------------------------------------------------------------------
# Streamlit Application Styling
# ---------------------------------------------------------------------------

def apply_custom_css():
    """Apply premium, modern dark botanical AI dashboard styling for the application."""
    st.markdown(
        """
        <style>
        /* Base application styling */
        .stApp {
            background-color: #0B0F0D;
            color: #F1F5F2;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
        }

        /* Typography & Headers */
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            color: #F1F5F2;
            margin-bottom: 0.3rem;
            letter-spacing: -0.02em;
            line-height: 1.25;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #A5B3A9;
            margin-bottom: 1.5rem;
            font-weight: 400;
            line-height: 1.45;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #F1F5F2 !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }
        p, span, label, li {
            color: #A5B3A9;
        }
        strong, b {
            color: #F1F5F2;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #101713 !important;
            border-right: 1px solid #29372F;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
            color: #F1F5F2 !important;
        }
        section[data-testid="stSidebar"] .stRadio label {
            color: #F1F5F2 !important;
            font-weight: 500;
        }

        /* Cards and Containers */
        .capability-card {
            background-color: #151D18;
            border: 1px solid #29372F;
            border-radius: 10px;
            padding: 1.25rem;
            height: 100%;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            transition: border-color 0.2s ease, transform 0.2s ease;
        }
        .capability-card:hover {
            border-color: #35C979;
            transform: translateY(-2px);
        }
        .capability-card h4 {
            margin-top: 0;
            margin-bottom: 0.5rem;
            color: #F1F5F2 !important;
            font-size: 1.08rem;
            font-weight: 700;
        }
        .capability-card p {
            color: #A5B3A9 !important;
            font-size: 0.88rem;
            line-height: 1.5;
        }
        .capability-card ul {
            color: #A5B3A9 !important;
            font-size: 0.85rem;
            line-height: 1.6;
        }
        .capability-card li {
            color: #A5B3A9 !important;
        }
        .capability-card strong {
            color: #F1F5F2 !important;
        }

        /* Badges */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 4px;
            margin-bottom: 0.5rem;
            letter-spacing: 0.02em;
        }
        .badge-validated {
            background-color: rgba(53, 201, 121, 0.16);
            color: #35C979;
            border: 1px solid rgba(53, 201, 121, 0.35);
        }
        .badge-production {
            background-color: rgba(117, 221, 160, 0.16);
            color: #75DDA0;
            border: 1px solid rgba(117, 221, 160, 0.35);
        }
        .badge-framework {
            background-color: rgba(165, 180, 252, 0.12);
            color: #A5B4FC;
            border: 1px solid rgba(165, 180, 252, 0.3);
        }
        .tech-badge {
            display: inline-block;
            background-color: #1B2620;
            color: #F1F5F2;
            padding: 0.3rem 0.65rem;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 500;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            border: 1px solid #29372F;
        }

        /* Metric Cards */
        div[data-testid="stMetric"] {
            background-color: #151D18;
            border: 1px solid #29372F;
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        div[data-testid="stMetricLabel"] {
            color: #A5B3A9 !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #35C979 !important;
            font-size: 1.65rem !important;
            font-weight: 800 !important;
        }

        /* Informational and Alert Boxes */
        .disclaimer-box {
            background-color: #151D18;
            border: 1px solid #29372F;
            border-left: 4px solid #35C979;
            padding: 0.9rem 1.15rem;
            border-radius: 6px;
            margin-top: 1rem;
            margin-bottom: 1rem;
            font-size: 0.88rem;
            color: #A5B3A9;
            line-height: 1.55;
        }
        .disclaimer-box strong {
            color: #F1F5F2;
        }
        .warning-box {
            background-color: #151D18;
            border: 1px solid #29372F;
            border-left: 4px solid #F2B84B;
            padding: 0.9rem 1.15rem;
            border-radius: 6px;
            margin-top: 0.8rem;
            margin-bottom: 0.8rem;
            font-size: 0.88rem;
            color: #F2B84B;
            line-height: 1.55;
        }
        .warning-box strong {
            color: #F1F5F2;
        }

        /* Streamlit Native Alerts */
        div[data-testid="stAlert"] {
            background-color: #151D18;
            border: 1px solid #29372F;
            border-radius: 8px;
            color: #F1F5F2;
        }

        /* Action Buttons */
        .stButton > button {
            background-color: #1B2620;
            color: #F1F5F2;
            border: 1px solid #29372F;
            border-radius: 6px;
            font-weight: 600;
            padding: 0.5rem 1.1rem;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background-color: #151D18;
            border-color: #35C979;
            color: #35C979;
        }
        .stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #35C979 0%, #2BA663 100%) !important;
            color: #0B0F0D !important;
            font-weight: 700 !important;
            border: none !important;
            box-shadow: 0 2px 10px rgba(53, 201, 121, 0.25);
        }
        .stButton > button[kind="primary"]:hover, .stButton > button[data-testid="baseButton-primary"]:hover {
            background: linear-gradient(135deg, #75DDA0 0%, #35C979 100%) !important;
            color: #0B0F0D !important;
            box-shadow: 0 4px 14px rgba(53, 201, 121, 0.4);
        }
        .stDownloadButton > button {
            background-color: #1B2620;
            color: #75DDA0;
            border: 1px solid #29372F;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stDownloadButton > button:hover {
            background-color: #151D18;
            border-color: #75DDA0;
            color: #F1F5F2;
        }

        /* File Uploader Dropzone */
        section[data-testid="stFileUploaderDropzone"] {
            background-color: #151D18 !important;
            border: 1.5px dashed #29372F !important;
            border-radius: 8px;
            padding: 1.5rem 1rem;
        }
        section[data-testid="stFileUploaderDropzone"]:hover {
            border-color: #35C979 !important;
        }
        section[data-testid="stFileUploaderDropzone"] span {
            color: #A5B3A9 !important;
        }

        /* Progress Bars */
        div[data-testid="stProgressBar"] > div {
            background-color: #1B2620;
            border-radius: 6px;
            height: 10px;
        }
        div[data-testid="stProgressBar"] > div > div {
            background: linear-gradient(90deg, #35C979 0%, #75DDA0 100%) !important;
            border-radius: 6px;
        }

        /* Expanders */
        div[data-testid="stExpander"] {
            background-color: #151D18;
            border: 1px solid #29372F;
            border-radius: 8px;
        }

        /* Dataframe / Tables */
        div[data-testid="stDataFrame"] {
            border: 1px solid #29372F !important;
            border-radius: 8px !important;
            background-color: #151D18 !important;
        }

        /* Form Inputs & Dropdowns */
        div[data-baseweb="select"] > div {
            background-color: #151D18 !important;
            border-color: #29372F !important;
            color: #F1F5F2 !important;
        }
        div[data-baseweb="popover"] ul {
            background-color: #151D18 !important;
            border: 1px solid #29372F !important;
        }
        div[data-baseweb="popover"] li {
            color: #F1F5F2 !important;
        }
        div[data-baseweb="popover"] li:hover {
            background-color: #1B2620 !important;
        }
        div[data-baseweb="input"] > div {
            background-color: #151D18 !important;
            border-color: #29372F !important;
            color: #F1F5F2 !important;
        }
        div[data-baseweb="input"] input {
            color: #F1F5F2 !important;
        }

        /* Controls: Checkbox, Slider, Radio, Selectbox labels */
        .stCheckbox label span, .stSlider label span, .stSelectbox label span, .stRadio label span {
            color: #F1F5F2 !important;
            font-weight: 500;
        }
        div[data-testid="stSlider"] div[data-testid="stTickBar"] {
            color: #A5B3A9 !important;
        }
        div[data-testid="stExpander"] details summary {
            color: #F1F5F2 !important;
        }
        div[data-testid="stExpander"] details summary svg {
            fill: #A5B3A9 !important;
        }

        /* Horizontal Rules / Dividers */
        hr {
            border-color: #29372F !important;
            margin: 1.8rem 0;
        }

        /* Captions */
        div[data-testid="stCaptionContainer"] {
            color: #A5B3A9 !important;
        }

        /* Images and Video Player Containers */
        [data-testid="stImage"] img {
            border-radius: 8px;
            border: 1px solid #29372F;
        }
        [data-testid="stVideo"] {
            border-radius: 8px;
            border: 1px solid #29372F;
            background-color: #101713;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 1: Home & Overview (Consolidated Project Hub)
# ---------------------------------------------------------------------------

def render_home_page():
    """Render consolidated, presentation-ready Home & Overview page."""
    st.markdown('<div class="main-header">Tomato Disease and Pest Surveillance System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Computer Vision for Precision Tomato Crop Health Monitoring</div>', unsafe_allow_html=True)

    st.markdown(
        """
        Welcome to the **Tomato Disease and Pest Surveillance System**. This platform provides automated
        foliar pathology diagnosis, *Tuta absoluta* leafminer damage instance segmentation, and real-time field
        video monitoring to protect tomato crops against yield loss.
        """
    )

    st.markdown("### Core Capabilities")
    c1, c2, c3, c4 = st.columns(4, gap="small")

    with c1:
        st.markdown(
            """
            <div class="capability-card">
                <h4>🌿 Tomato Disease Classification</h4>
                <div><span class="badge badge-validated">Validated Production</span></div>
                <p style="font-size: 0.9rem; color: #A5B3A9; margin-top: 0.5rem;">
                    Calibrated ResNet18 model fine-tuned on the PlantVillage benchmark to diagnose 10 foliar tomato conditions.
                </p>
                <ul style="font-size: 0.86rem; color: #A5B3A9; padding-left: 1.2rem; margin-bottom: 0;">
                    <li><strong>Architecture:</strong> ResNet18 Transfer Learning</li>
                    <li><strong>Holdout Accuracy:</strong> <strong>99.23%</strong> (N = 2,724)</li>
                    <li><strong>Macro F1-Score:</strong> <strong>0.9885</strong></li>
                    <li><strong>Post-Hoc Calibration:</strong> T* = 1.3021 (ECE: 0.36%)</li>
                    <li><strong>Interpretability:</strong> Layer-4 Grad-CAM Saliency</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="capability-card">
                <h4>🐛 Tuta absoluta Damage Detection</h4>
                <div><span class="badge badge-production">YOLOv8n-Seg Production</span></div>
                <p style="font-size: 0.9rem; color: #A5B3A9; margin-top: 0.5rem;">
                    Instance segmentation model trained on verified tomato foliage to localize leafminer feeding damage tunnels and mines.
                </p>
                <ul style="font-size: 0.86rem; color: #A5B3A9; padding-left: 1.2rem; margin-bottom: 0;">
                    <li><strong>Model:</strong> YOLOv8n-Seg (3.39M parameters)</li>
                    <li><strong>Task:</strong> Foliar feeding mine instance segmentation</li>
                    <li><strong>Targets:</strong> Larval feeding mines & necrotic tunnels</li>
                    <li><strong>References:</strong> Healthy foliage baseline regions</li>
                    <li><strong>Benchmark Evaluation:</strong> Detailed below</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="capability-card">
                <h4>🔬 Foliar Lesion Segmentation</h4>
                <div><span class="badge badge-framework">PlantSeg U-Net</span></div>
                <p style="font-size: 0.9rem; color: #A5B3A9; margin-top: 0.5rem;">
                    Pixel-level foliar lesion segmentation delineating necrotic tissue and chlorotic infection boundaries.
                </p>
                <ul style="font-size: 0.86rem; color: #A5B3A9; padding-left: 1.2rem; margin-bottom: 0;">
                    <li><strong>Architecture:</strong> Improved PlantSeg U-Net</li>
                    <li><strong>Task:</strong> Pixel-wise foliar pathology mask prediction</li>
                    <li><strong>Resolution:</strong> 256×256 input with dual-head lesion logits</li>
                    <li><strong>Outputs:</strong> Blended mask overlay, damage fraction (%)</li>
                    <li><strong>Integration:</strong> Available in Disease module</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            """
            <div class="capability-card">
                <h4>📹 Field Video Surveillance</h4>
                <div><span class="badge badge-framework">Field Pipeline</span></div>
                <p style="font-size: 0.9rem; color: #A5B3A9; margin-top: 0.5rem;">
                    Multi-task vision engine integrating calibrated disease classification and YOLOv8n-Seg pest segmentation.
                </p>
                <ul style="font-size: 0.86rem; color: #A5B3A9; padding-left: 1.2rem; margin-bottom: 0;">
                    <li><strong>Engine:</strong> OpenCV Multi-Task Pipeline</li>
                    <li><strong>Task:</strong> Automated foliar disease and pest inspection</li>
                    <li><strong>Temporal Subsampling:</strong> Configurable Stride (k = 1 to 5)</li>
                    <li><strong>Outputs:</strong> Annotated Stream, HUD Telemetry</li>
                    <li><strong>Telemetry:</strong> Temporal Disease & Pest Severity Timelines</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Performance Summary")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric(label="Disease Accuracy", value="99.23%", help="Evaluated on untouched PlantVillage test split (N=2,724)")
    p2.metric(label="Disease Macro F1", value="0.9885", help="Unweighted macro average across all 10 conditions")
    p3.metric(label="Pest Box mAP@0.5", value="71.99%", help="YOLOv8n-Seg on TutaAbsolutaResearch v4 test split")
    p4.metric(label="Pest Mask mAP@0.5", value="70.37%", help="YOLOv8n-Seg instance mask segmentation on test split")

    st.divider()

    st.markdown("### Pest Detection Performance — YOLOv8n-Seg")
    st.caption(
        "Verified model-level evaluation metrics on untouched holdout test split (TutaAbsolutaResearch v4, N = 92 images, 151 polygon annotations). "
        "Precision and recall values reflect offline holdout test benchmark performance, not single-prediction confidence:"
    )

    y_col1, y_col2, y_col3, y_col4 = st.columns(4)
    y_col1.metric("Box mAP@0.5", "71.99%")
    y_col2.metric("Mask mAP@0.5", "70.37%")
    y_col3.metric("Benchmark Precision", "79.76%", help="Model-level precision evaluated on holdout test annotations")
    y_col4.metric("Benchmark Recall", "56.45%", help="Model-level recall evaluated on holdout test annotations")

    st.markdown(
        """
        - **Model Architecture:** YOLOv8n-Seg instance segmentation with multi-scale feature pyramid and Proto mask generation (3.39M parameters).
        - **Target Pathology:** Foliar feeding damage mines and necrotic tunnels created by *Tuta absoluta* larvae.
        - **CPU Latency & Throughput:** 77.99 ms/image (12.82 FPS on standard CPU).
        """
    )

    st.divider()

    st.markdown("### Technology Stack")
    st.markdown(
        """
        <span class="tech-badge">Python 3.12</span>
        <span class="tech-badge">PyTorch 2.x</span>
        <span class="tech-badge">Torchvision</span>
        <span class="tech-badge">Ultralytics YOLOv8</span>
        <span class="tech-badge">OpenCV (cv2)</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">NumPy</span>
        <span class="tech-badge">pandas</span>
        <span class="tech-badge">KaTeX Math</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Project & Dataset Specifications")
    st.markdown(
        """
        - **PlantVillage Dataset:** 18,160 leaf images spanning 10 classes (9 pathologies + healthy). Strict leakage-safe splits: 12,712 train (70%), 2,724 validation (15%), 2,724 test (15%).
        - **TutaAbsolutaResearch v4 Dataset:** 937 high-resolution tomato foliage images with verified *Tuta absoluta* feeding mines and healthy reference polygons. Split: 657 train, 188 validation, 92 test.
        - **PlantSeg Foliar Lesion Benchmark:** Improved U-Net segmenter generating pixel-wise lesion masks and leaf surface damage ratios.
        """
    )

    st.markdown(
        """
        <div class="disclaimer-box">
        <strong>⚠️ Decision-Support Disclaimer:</strong><br>
        This software provides automated decision support derived from deep learning models.
        All phytosanitary and chemical intervention decisions should be verified by a qualified agronomist.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 2: Disease Classification (Standalone Module)
# ---------------------------------------------------------------------------

def render_classification_page(model: TomatoResNet18, device: torch.device, idx_to_class: Dict[int, str]):
    """Render validated tomato leaf disease classification page."""
    st.markdown('<div class="main-header">🔬 Tomato Disease Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Calibrated ResNet18 (T* = 1.3021) across 10 Tomato Conditions</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 1.2], gap="large")

    with col1:
        st.subheader("1. Upload Leaf Image")
        st.caption("💡 **Image Guidance:** For optimal diagnostic accuracy, upload a clear, well-lit photograph focusing on an individual tomato leaf showing visible symptoms against a clean background.")
        if "cls_upload_counter" not in st.session_state:
            st.session_state["cls_upload_counter"] = 0

        uploaded_file = st.file_uploader(
            label="Upload photographic image (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp", "JPG", "JPEG", "PNG", "WEBP"],
            key=f"cls_uploader_{st.session_state['cls_upload_counter']}",
            help="Select a photograph of an individual tomato leaf.",
        )

        show_gradcam = st.checkbox("🔍 Compute Grad-CAM Attention Heatmap", value=False)
        show_lesion_seg = st.checkbox("🔬 Segment Foliar Lesions (PlantSeg U-Net)", value=False)

        uploaded_image = None
        if uploaded_file is not None:
            # Eager byte reading & SHA-256 hashing to guarantee independent processing
            raw_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
            current_cls_file_id = f"{uploaded_file.name}_{file_hash}"

            if st.session_state.get("last_cls_file_id") != current_cls_file_id:
                st.session_state.pop("cls_prediction", None)
                st.session_state.pop("cls_image", None)
                st.session_state["last_cls_file_id"] = current_cls_file_id

            try:
                pil_img = Image.open(io.BytesIO(raw_bytes))
                uploaded_image = ImageOps.exif_transpose(pil_img).convert("RGB")
                uploaded_image.load()
                st.image(
                    uploaded_image,
                    caption=f"Uploaded: {uploaded_file.name} ({uploaded_image.size[0]}×{uploaded_image.size[1]}px)",
                    use_container_width=True,
                )
            except Exception as ie:
                st.error(f"Image decode error: {ie}. Please upload a valid JPG or PNG photograph.")
                uploaded_image = None

            if st.button("🔄 Clear / Upload Different Image", key="btn_clear_cls_img"):
                st.session_state["cls_upload_counter"] += 1
                st.session_state.pop("cls_prediction", None)
                st.session_state.pop("cls_image", None)
                st.session_state.pop("last_cls_file_id", None)
                st.rerun()
        else:
            st.session_state.pop("cls_prediction", None)
            st.session_state.pop("cls_image", None)
            st.session_state.pop("last_cls_file_id", None)

        predict_button = st.button("🚀 Diagnose Disease", type="primary", disabled=(uploaded_image is None))

    with col2:
        st.subheader("2. Diagnostic Results")

        if uploaded_image is None:
            st.info("Upload a tomato leaf photo in the left panel and click **Diagnose Disease**.")
        elif not predict_button and "cls_prediction" not in st.session_state:
            st.info("Image loaded. Click **Diagnose Disease** to execute calibrated inference.")
        else:
            if predict_button:
                with st.spinner("Executing calibrated ResNet18 inference (T* = 1.3021)..."):
                    try:
                        predictions = predict_leaf(uploaded_image, model, device, idx_to_class, top_k=len(idx_to_class))
                        st.session_state["cls_prediction"] = predictions
                        st.session_state["cls_image"] = uploaded_image
                    except Exception as err:
                        st.error(f"Inference error: {err}")
                        predictions = None
            else:
                predictions = st.session_state.get("cls_prediction")

            if predictions:
                top1 = predictions[0]

                is_low_confidence = (top1["probability"] < 0.60)
                if is_low_confidence:
                    st.warning(
                        f"⚠️ **Uncertain Prediction Warning:** The top calibrated confidence is {top1['percentage']:.1f}% (< 60.0%). "
                        "The symptoms may be ambiguous or in an early developmental stage. "
                        "This tentative indication should not be treated as a confirmed diagnosis; agronomist inspection is advised."
                    )
                    st.markdown(f"### Tentative Indication: **{top1['display_name']}** *(Uncertain / Low Confidence)*")
                else:
                    if top1["is_healthy"]:
                        st.success(f"### Predicted Diagnosis: **{top1['display_name']}**")
                    else:
                        st.error(f"### Predicted Pathology: **{top1['display_name']}**")

                st.metric(label="Calibrated Diagnostic Confidence", value=f"{top1['percentage']:.2f}%")

                st.markdown("#### Top-3 Ranked Diagnostic Hypotheses")
                for pred in predictions[:3]:
                    c_name, c_bar, c_pct = st.columns([2.2, 3.0, 1.0])
                    with c_name:
                        icon = "🥇" if pred["rank"] == 1 else ("🥈" if pred["rank"] == 2 else "🥉")
                        st.write(f"{icon} **{pred['display_name']}**")
                    with c_bar:
                        st.progress(float(np.clip(pred["probability"], 0.0, 1.0)))
                    with c_pct:
                        st.write(f"**{pred['percentage']:.1f}%**")

                with st.expander("View All 10 Disease Class Probabilities", expanded=False):
                    st.caption("Calibrated probabilities across all 10 foliar classes supported by ResNet18:")
                    for pred in predictions:
                        r_col, name_col, bar_col, pct_col = st.columns([0.6, 2.6, 3.2, 1.2])
                        with r_col:
                            st.write(f"#{pred['rank']}")
                        with name_col:
                            st.write(f"**{pred['display_name']}**")
                        with bar_col:
                            st.progress(float(np.clip(pred["probability"], 0.0, 1.0)))
                        with pct_col:
                            st.write(f"**{pred['percentage']:.2f}%**")
                    total_prob = sum(p["percentage"] for p in predictions)
                    st.caption(f"**Total Probability Sum:** {total_prob:.2f}% (calibrated via temperature scaling $T^* = {CALIBRATION_TEMPERATURE}$)")

                if show_gradcam:
                    st.divider()
                    st.markdown("#### 🧠 Model Visual Explanation (Grad-CAM)")
                    with st.spinner("Generating gradient-weighted class activation map (layer4)..."):
                        try:
                            heatmap, blended, pred_c, _ = run_gradcam_overlay(uploaded_image, model)
                            g_col1, g_col2 = st.columns(2)
                            with g_col1:
                                st.image(heatmap, caption="Saliency Heatmap (layer4)", use_container_width=True, clamp=True)
                            with g_col2:
                                st.image(blended, caption="Grad-CAM Overlay", use_container_width=True)
                            st.caption("Warm regions (red/yellow) indicate leaf zones that strongly drove the classification decision.")
                        except Exception as ge:
                            st.warning(f"Grad-CAM could not be computed: {ge}")

                if show_lesion_seg:
                    st.divider()
                    st.markdown("#### 🔬 Foliar Lesion Segmentation (PlantSeg U-Net)")
                    st.caption("Pixel-level semantic segmentation identifying necrotic lesion tissue across the lamina:")
                    unet_model, unet_dev, _ = load_unet_lesion_segmenter(UNET_CHECKPOINT_PATH)
                    if unet_model is not None:
                        with st.spinner("Executing PlantSeg U-Net lesion segmentation..."):
                            try:
                                seg_overlay, lesion_pct, binary_mask = predict_unet_lesions(uploaded_image, unet_model, unet_dev)
                                u_col1, u_col2 = st.columns(2)
                                with u_col1:
                                    st.image(uploaded_image, caption="Original Leaf Image", use_container_width=True)
                                with u_col2:
                                    mask_display = (binary_mask * 255).astype(np.uint8)
                                    st.image(mask_display, caption=f"Predicted Binary Lesion Mask ({lesion_pct:.2f}% lesion area)", use_container_width=True, clamp=True)

                                st.image(seg_overlay, caption=f"Lesion Mask Overlay on Leaf ({lesion_pct:.2f}% foliar damage)", use_container_width=True)
                                st.metric("Estimated Foliar Lesion Coverage", f"{lesion_pct:.2f}%")
                                st.caption("Segments necrotic disease lesion tissue across the lamina using pixel-level U-Net. Note: This operates independently from Grad-CAM attention heatmaps.")
                            except Exception as ue:
                                st.warning(f"Lesion segmentation failed: {ue}")
                    else:
                        st.info("PlantSeg U-Net checkpoint not found at `artifacts/checkpoints/segmentation_real/best_unet_improved.pt`.")

                st.divider()
                st.markdown("#### 📋 AI Diagnostic Inspection Summary")
                assessment_label = f"Tentative Indication: {top1['display_name']} (Uncertain / Low Confidence)" if is_low_confidence else f"Predicted: {top1['display_name']}"
                st.markdown(
                    f"""
                    <div class="disclaimer-box" style="margin-top: 0.5rem; margin-bottom: 1rem;">
                        <strong>Analysis Status:</strong> Completed<br>
                        <strong>Diagnostic Assessment:</strong> {assessment_label}<br>
                        <strong>Calibrated Confidence:</strong> {top1['percentage']:.2f}% (T* = {CALIBRATION_TEMPERATURE})<br>
                        <strong>Diagnostic Category:</strong> {'Healthy Tomato Foliage' if top1['is_healthy'] else 'Foliar Pathology'}<br>
                        <strong>Foliar Classes Evaluated:</strong> 10 conditions assessed via calibrated ResNet18<br>
                        <strong>AI Prediction Disclaimer:</strong> This automated assessment is intended for decision-support guidance only. Phytosanitary interventions should be verified by a certified crop consultant.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Page 3: Pest Damage Detection (YOLOv8n-Seg Production)
# ---------------------------------------------------------------------------

def render_pest_detection_page(
    detector: Optional[TinyYOLODetector],
    device: torch.device,
    yolo_detector: Optional[Any] = None,
):
    """Render Tuta absoluta pest damage detection page (YOLOv8n-Seg Production Engine)."""
    st.markdown('<div class="main-header">🐛 Pest Damage Detection (Tuta absoluta)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Foliar Feeding Mine Localization & Instance Segmentation (YOLOv8n-Seg Production)</div>', unsafe_allow_html=True)

    if yolo_detector is None:
        yolo_detector = load_yolov8_detector(YOLO_CHECKPOINT_PATH)

    if yolo_detector is None:
        st.error(f"Production YOLOv8n-Seg checkpoint `{YOLO_CHECKPOINT_PATH}` not found.")
        return

    st.markdown(
        """
        <div class="warning-box">
        <strong>⚠️ Biological Identification Clarification:</strong><br>
        All detected regions represent <strong>foliar feeding damage</strong> (chlorotic/necrotic leaf mines and feeding tunnels created by <em>Tuta absoluta</em> larvae), <strong>NOT living adult insects</strong>.<br>
        Adult moths require pheromone delta traps for direct entomological counts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-box">
        <strong>📌 Detection Semantics & Reference Labels:</strong><br>
        • <strong style="color: #FF6464;">Red Masks / Bounding Boxes:</strong> <strong>Tuta absoluta Feeding Damage</strong> (actual foliar feeding mines and necrotic tunnels).<br>
        • <strong style="color: #35C979;">Green Masks / Bounding Boxes:</strong> <strong>Healthy Foliage / Reference Regions</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.1, 1.2], gap="large")

    with col1:
        st.subheader("1. Upload Foliage Image")
        if "pest_upload_counter" not in st.session_state:
            st.session_state["pest_upload_counter"] = 0

        uploaded_file = st.file_uploader(
            label="Upload photo for pest damage inspection (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp", "JPG", "JPEG", "PNG", "WEBP"],
            key=f"pest_uploader_{st.session_state['pest_upload_counter']}",
        )

        conf_thresh = st.slider(
            "Detection Confidence Threshold",
            min_value=0.10,
            max_value=0.80,
            value=0.25,
            step=0.05,
            help="Minimum confidence for Tuta absoluta damage localization. Note: Foliar disease spots (Septoria, Blight) are not Tuta mines.",
        )

        iou_thresh = st.slider("NMS IoU Threshold", min_value=0.20, max_value=0.80, value=0.45, step=0.05)
        show_healthy = st.checkbox("Show Healthy Reference Foliage (Class 0)", value=True, help="Uncheck to hide background healthy reference boxes and focus strictly on Tuta absoluta damage.")

        uploaded_image = None
        if uploaded_file is not None:
            raw_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(raw_bytes).hexdigest()[:16]
            current_pest_run_key = f"{file_hash}_{conf_thresh:.2f}_{iou_thresh:.2f}_{show_healthy}"

            if st.session_state.get("last_pest_run_key") != current_pest_run_key:
                st.session_state.pop("pest_annotated", None)
                st.session_state.pop("pest_detections", None)
                st.session_state.pop("tuta_count", None)
                st.session_state.pop("healthy_count", None)
                st.session_state["last_pest_run_key"] = current_pest_run_key

            try:
                pil_img = Image.open(io.BytesIO(raw_bytes))
                uploaded_image = ImageOps.exif_transpose(pil_img).convert("RGB")
                uploaded_image.load()
                st.image(uploaded_image, caption=f"Uploaded: {uploaded_file.name} ({uploaded_image.size[0]}×{uploaded_image.size[1]}px)", use_container_width=True)
            except Exception as ie:
                st.error(f"Invalid image format: {ie}")
                uploaded_image = None

            if st.button("🔄 Clear / Upload Different Image", key="btn_clear_pest_img"):
                st.session_state["pest_upload_counter"] += 1
                st.session_state.pop("pest_annotated", None)
                st.session_state.pop("pest_detections", None)
                st.session_state.pop("tuta_count", None)
                st.session_state.pop("healthy_count", None)
                st.session_state.pop("last_pest_run_key", None)
                st.rerun()
        else:
            st.session_state.pop("last_pest_run_key", None)
            st.session_state.pop("pest_annotated", None)
            st.session_state.pop("pest_detections", None)
            st.session_state.pop("tuta_count", None)
            st.session_state.pop("healthy_count", None)

        detect_button = st.button("🔍 Detect Tuta absoluta Damage", type="primary", disabled=(uploaded_image is None))

    with col2:
        st.subheader("2. Detection & Segmentation Output")

        if uploaded_image is None:
            st.info("Upload an image in the left panel and click **Detect Tuta absoluta Damage**.")
        elif not detect_button and "pest_annotated" not in st.session_state:
            st.info("Image ready. Click **Detect Tuta absoluta Damage** to run YOLOv8n-Seg.")
        else:
            if detect_button:
                with st.spinner("Executing YOLOv8n-Seg Instance Segmentation..."):
                    try:
                        annotated, detections, tuta_cnt, healthy_cnt = predict_pests_yolov8(
                            uploaded_image, yolo_detector, conf_threshold=conf_thresh, iou_threshold=iou_thresh, show_healthy=show_healthy
                        )
                        st.session_state["pest_annotated"] = annotated
                        st.session_state["pest_detections"] = detections
                        st.session_state["tuta_count"] = tuta_cnt
                        st.session_state["healthy_count"] = healthy_cnt
                    except Exception as pe:
                        st.error(f"Detection failed: {pe}")
                        detections = None
            else:
                annotated = st.session_state.get("pest_annotated")
                detections = st.session_state.get("pest_detections", [])
                tuta_cnt = st.session_state.get("tuta_count", 0)
                healthy_cnt = st.session_state.get("healthy_count", 0)

            if detections is not None:
                caption_str = "Annotated Regions (Red: Tuta Damage, Green: Healthy Reference Foliage)"
                st.image(annotated, caption=caption_str, use_container_width=True)

                d_col1, d_col2 = st.columns(2)
                d_col1.metric("Detected Tuta Feeding-Damage Regions", f"{tuta_cnt}")
                d_col2.metric("Healthy Foliage Reference Regions", f"{healthy_cnt}")

                max_conf = max([float(d.get("confidence", 0.0)) for d in detections]) if detections else 0.0

                if healthy_cnt == 0:
                    st.caption(
                        "ℹ️ **Foliage Reference Note:** Zero healthy reference regions were identified by the model on this specific foliage surface. "
                        "In the underlying *TutaAbsolutaResearch v4* training distribution, images containing severe or distributed feeding mines are annotated strictly for larval damage (Class 1) rather than residual intact lamina (Class 0). "
                        "Consequently, the model's instance segmentation heads localize active feeding mines without partitioning intact background foliage into artificial healthy boxes; this does NOT imply that the entire leaf is diseased."
                    )
                else:
                    st.caption(
                        "ℹ️ **Foliage Reference Note:** Green bounding boxes and masks indicate unaffected healthy reference foliage; they do NOT represent pest detections."
                    )

                if tuta_cnt > 0 and max_conf < 0.40:
                    st.warning(
                        f"⚠️ **Low-Confidence Symptom Advisory (Peak Region Confidence: {max_conf:.2f}):**\n\n"
                        f"The {tuta_cnt} detected region(s) barely cleared the confidence threshold.\n\n"
                        "• **Agronomic Context:** *Tuta absoluta* larvae create **broad, irregular blotchy feeding mines** (translucent/brown blisters with dark frass). "
                        "In contrast, **numerous small circular spots (1–3 mm) with dark rims or shot-holes** are caused by **foliar fungal/bacterial diseases** (e.g., *Septoria* Leaf Spot, Bacterial Spot) or flea beetle feeding.\n\n"
                        "• **Specialized Scope:** This detector is specialized strictly for *Tuta absoluta* mines."
                    )

                # Near-miss advisory if candidate Tuta mine is just below threshold
                near_misses = st.session_state.get("near_miss_tuta", [])
                if near_misses:
                    best_cand_conf = max([float(item.get("conf", 0.0)) for item in near_misses])
                    prefix = "Additional" if tuta_cnt > 0 else "Candidate"
                    st.info(
                        f"💡 **{prefix} Feeding Damage Detected Below Threshold (Confidence: {best_cand_conf:.2f}):** "
                        f"{len(near_misses)} candidate damage region{'s were' if len(near_misses) > 1 else ' was'} detected with confidence < {conf_thresh:.2f}. "
                        f"Adjust the slider to **{max(0.10, best_cand_conf - 0.01):.2f}** or lower to reveal {'them' if len(near_misses) > 1 else 'it'}."
                    )

                if tuta_cnt == 0 and healthy_cnt == 0:
                    st.info("No regions detected above the selected confidence threshold.")
                elif tuta_cnt == 0 and healthy_cnt > 0:
                    st.info("No Tuta absoluta feeding damage detected on this leaf.")

                # Move raw bounding-box coordinates, mask details and technical detection inventory into an expander
                if detections:
                    with st.expander("Technical Detection Inventory & Region Coordinates", expanded=False):
                        st.caption("Quantitative bounding-box coordinates [xmin, ymin, xmax, ymax], individual detection confidence, and segmented mask areas:")
                        df_det = pd.DataFrame(detections)
                        disp_cols = [c for c in ["category", "confidence", "box", "mask_area_px"] if c in df_det.columns]
                        st.dataframe(df_det[disp_cols], use_container_width=True)

                st.divider()
                st.markdown("#### 📋 AI Pest Inspection Summary")
                tuta_status_text = f"Detected ({tuta_cnt} feeding-damage region{'s' if tuta_cnt != 1 else ''})" if tuta_cnt > 0 else "Not Detected"
                max_c_str = f" (Peak Region Confidence: {max_conf:.2f})" if tuta_cnt > 0 else ""
                st.markdown(
                    f"""
                    <div class="disclaimer-box" style="margin-top: 0.5rem; margin-bottom: 1rem;">
                        <strong>Analysis Status:</strong> Completed<br>
                        <strong>Tuta Feeding Damage:</strong> {tuta_status_text}{max_c_str}<br>
                        <strong>Detected Damage Regions:</strong> {tuta_cnt} localized larval feeding mine{'s' if tuta_cnt != 1 else ''}<br>
                        <strong>Healthy Foliage References:</strong> {healthy_cnt} baseline reference region{'s' if healthy_cnt != 1 else ''}<br>
                        <strong>Biological Clarification:</strong> Localizes larval foliar feeding mines and necrotic tunnels; does NOT represent counts of living adult insects.<br>
                        <strong>Field Verification:</strong> Deploy delta pheromone traps for entomological monitoring of adult moth flight activity.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Page 4: Video Surveillance (Multi-Task Pipeline)
# ---------------------------------------------------------------------------

def render_video_surveillance_page():
    """Render tomato field video surveillance page."""
    st.markdown('<div class="main-header">📹 Tomato Field-Video Surveillance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Task Frame Surveillance (Calibrated ResNet18 + YOLOv8n-Seg Production)</div>', unsafe_allow_html=True)

    st.markdown("### Surveillance Workflow")
    st.markdown(
        """
        1. **Upload Video:** Upload a tomato crop field video (<60s duration, <100MB file size; .mp4, .avi, .mov).
        2. **Configure Temporal Stride:** Sample every $k$-th frame to match compute and frame-rate requirements.
        3. **Multi-Task Inference:** Runs calibrated ResNet18 disease diagnosis and YOLOv8n-Seg pest damage instance segmentation simultaneously.
        4. **Generate Telemetry:** Produces an annotated video stream, temporal disease progression timeline, and downloadable frame log CSV.
        """
    )

    processor = VideoProcessor(max_duration_seconds=60.0, max_file_size_mb=100.0)

    st.subheader("1. Upload Custom Video")
    st.caption("Upload a photographic field video of tomato crops (maximum 60 seconds duration, maximum 100 MB).")

    if "vid_upload_counter" not in st.session_state:
        st.session_state["vid_upload_counter"] = 0

    uploaded_vid = st.file_uploader(
        "Upload tomato field video (.mp4, .avi, .mov)",
        type=["mp4", "avi", "mov", "MP4", "AVI", "MOV"],
        key=f"vid_uploader_{st.session_state['vid_upload_counter']}",
        help="Select a video file under 60 seconds and 100 MB.",
    )

    video_to_process: Optional[Path] = None

    if uploaded_vid is not None:
        raw_vid_bytes = uploaded_vid.getvalue()
        vid_hash = hashlib.sha256(raw_vid_bytes).hexdigest()[:16]
        current_vid_id = f"{uploaded_vid.name}_{vid_hash}"

        if st.session_state.get("last_uploaded_vid_id") != current_vid_id:
            st.session_state.pop("surveillance_results", None)
            st.session_state.pop("surveillance_video_out", None)
            st.session_state.pop("surveillance_timeline_out", None)
            st.session_state.pop("surveillance_total_time", None)
            st.session_state.pop("surveillance_mean_latency", None)
            st.session_state["last_uploaded_vid_id"] = current_vid_id

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_vid.name).suffix)
            tfile.write(raw_vid_bytes)
            tfile.close()
            st.session_state["saved_video_path"] = tfile.name

        saved_path_str = st.session_state.get("saved_video_path")
        if saved_path_str and Path(saved_path_str).exists():
            video_to_process = Path(saved_path_str)

        if st.button("🔄 Clear / Upload Different Video", key="btn_clear_vid"):
            st.session_state["vid_upload_counter"] += 1
            st.session_state.pop("surveillance_results", None)
            st.session_state.pop("surveillance_video_out", None)
            st.session_state.pop("surveillance_timeline_out", None)
            st.session_state.pop("surveillance_total_time", None)
            st.session_state.pop("surveillance_mean_latency", None)
            st.session_state.pop("last_uploaded_vid_id", None)
            st.session_state.pop("saved_video_path", None)
            st.rerun()
    elif SAMPLE_VIDEO_PATH.exists():
        if st.checkbox("Use Sample Benchmark Field Video (300 frames, 10s)", value=False):
            video_to_process = SAMPLE_VIDEO_PATH
            st.info(f"Loaded benchmark field surveillance video: `{SAMPLE_VIDEO_PATH.name}`")
    else:
        st.session_state.pop("surveillance_results", None)
        st.session_state.pop("surveillance_video_out", None)
        st.session_state.pop("surveillance_timeline_out", None)
        st.session_state.pop("surveillance_total_time", None)
        st.session_state.pop("surveillance_mean_latency", None)
        st.session_state.pop("last_uploaded_vid_id", None)
        st.session_state.pop("saved_video_path", None)

    stride = st.slider("Temporal Stride (Process every k-th frame)", min_value=1, max_value=5, value=2, help="Stride k=2 processes 15 FPS from 30 FPS stream.")

    run_button = st.button("🎬 Run Surveillance Analysis", type="primary", disabled=(video_to_process is None))

    if run_button and video_to_process is not None:
        try:
            metadata = processor.validate_and_inspect(video_to_process)
            st.write(
                f"**Video Metadata:** {metadata.width}×{metadata.height} | {metadata.fps:.1f} FPS | "
                f"{metadata.frame_count} frames | {metadata.duration_seconds:.2f}s | {metadata.file_size_mb:.2f} MB"
            )
        except Exception as ve:
            st.error(f"Video validation error: {ve}")
            return

        with st.spinner("Analyzing video frames (Calibrated ResNet18 + Production YOLOv8n-Seg)..."):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            try:
                analyzer = SurveillanceFrameAnalyzer(
                    temperature=CALIBRATION_TEMPERATURE,
                    use_yolov8=True,
                )
                annotator = VideoAnnotator()

                out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                out_path = Path(out_temp.name)
                out_temp.close()

                effective_fps = metadata.fps / float(stride)
                writer = processor.create_writer(out_path, metadata.width, metadata.height, fps=effective_fps, codec="mp4v")

                results: List[FrameAnalysisResult] = []
                frame_latencies: List[float] = []

                sampled_frames = list(processor.sample_frames(video_to_process, stride=stride))
                total_sampled = len(sampled_frames)

                t_start = time.perf_counter()
                for i, (f_idx, ts, frame_bgr) in enumerate(sampled_frames):
                    t0 = time.perf_counter()
                    res = analyzer.analyze_frame(frame_bgr, frame_idx=f_idx, timestamp=ts)
                    t1 = time.perf_counter()
                    frame_latencies.append((t1 - t0) * 1000.0)

                    annotated = annotator.annotate(frame_bgr, res)
                    writer.write(annotated)
                    results.append(res)

                    progress = (i + 1) / float(total_sampled)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed frame {i+1}/{total_sampled} ({res.disease_label}, {res.pest_count} pests)")

                writer.release()
                total_dur = time.perf_counter() - t_start

                # Generate timeline plot
                timeline_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                timeline_path = Path(timeline_temp.name)
                timeline_temp.close()
                plot_surveillance_timeline(results, save_path=timeline_path)

                # Store in session state
                st.session_state["surveillance_video_out"] = str(out_path)
                st.session_state["surveillance_timeline_out"] = str(timeline_path)
                st.session_state["surveillance_results"] = results
                st.session_state["surveillance_total_time"] = total_dur
                st.session_state["surveillance_mean_latency"] = float(np.mean(frame_latencies))

                st.success(f"Surveillance complete: Processed {total_sampled} frames in {total_dur:.2f}s ({np.mean(frame_latencies):.1f} ms/frame)")

            except Exception as se:
                st.error(f"Surveillance execution failure: {se}")
                return

    # Display results if present
    if "surveillance_results" in st.session_state:
        results = st.session_state["surveillance_results"]
        vid_out_p = Path(st.session_state["surveillance_video_out"])
        timeline_p = Path(st.session_state["surveillance_timeline_out"])
        total_time = st.session_state["surveillance_total_time"]
        mean_lat = st.session_state["surveillance_mean_latency"]

        st.subheader("2. Surveillance Telemetry & Findings")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Analyzed Frames", len(results))
        s2.metric("Processing Time", f"{total_time:.2f}s")
        s3.metric("Average Latency", f"{mean_lat:.1f} ms")
        s4.metric("Throughput", f"{1000.0 / mean_lat:.1f} FPS" if mean_lat > 0 else "N/A")

        # Compact AI Surveillance Inspection Summary
        total_pests_detected = sum(r.pest_count for r in results)
        dominant_condition = pd.Series([r.disease_label for r in results]).mode()[0] if results else "N/A"
        st.markdown("#### 📋 AI Video Surveillance Summary")
        st.markdown(
            f"""
            <div class="disclaimer-box" style="margin-top: 0.5rem; margin-bottom: 1rem;">
                <strong>Surveillance Status:</strong> Completed ({len(results)} frames analyzed)<br>
                <strong>Predominant Foliar Condition:</strong> {dominant_condition}<br>
                <strong>Tuta Damage Detection:</strong> {'Detected (' + str(total_pests_detected) + ' cumulative damage regions across frames)' if total_pests_detected > 0 else 'Not Detected'}<br>
                <strong>Actual Pipeline Latency:</strong> {mean_lat:.1f} ms/frame ({1000.0 / mean_lat:.1f} FPS measured throughput)<br>
                <strong>Interpretation Notice:</strong> Automated multi-task video analytics for field survey guidance. Agrochemical treatments require certified agronomist inspection.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Disease & Pest Temporal Surveillance Timeline")
        if timeline_p.exists():
            st.image(str(timeline_p), caption="Temporal Evolution of Foliar Condition and Pest Severity", use_container_width=True)

        st.markdown("#### Annotated Surveillance Video Stream")
        if vid_out_p.exists():
            try:
                st.video(str(vid_out_p))
            except Exception:
                st.warning("Direct in-browser playback unavailable for this codec. Please download the annotated video below.")

            with open(vid_out_p, "rb") as vf:
                st.download_button(
                    label="⬇️ Download Annotated Video (.mp4)",
                    data=vf.read(),
                    file_name="annotated_surveillance_output.mp4",
                    mime="video/mp4",
                )

        # Downloadable Telemetry CSV
        csv_data = []
        for r in results:
            ts_val = getattr(r, "timestamp_seconds", getattr(r, "timestamp", 0.0))
            csv_data.append({
                "frame_idx": r.frame_idx,
                "timestamp_sec": round(ts_val, 3),
                "predicted_disease": r.disease_label,
                "disease_confidence": round(r.disease_confidence, 4),
                "pest_count": r.pest_count,
                "detector_engine": r.pest_detector_type,
            })
        df_telemetry = pd.DataFrame(csv_data)

        st.markdown("#### Frame Telemetry Summary")
        st.dataframe(df_telemetry.head(20), use_container_width=True)

        csv_buffer = df_telemetry.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Frame Telemetry CSV",
            data=csv_buffer,
            file_name="surveillance_telemetry_log.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Main Application Controller (Clean 4-Page Navigation)
# ---------------------------------------------------------------------------

def main():
    """Main application entry point with clean 4-page navigation."""
    st.set_page_config(
        page_title="Tomato Disease and Pest Surveillance System",
        page_icon="🍅",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_css()

    try:
        class_to_idx, idx_to_class, class_names_sorted = load_class_mappings(CLASS_MAPPING_PATH)
    except Exception as e:
        st.error(f"Failed to load class mapping configuration: {e}")
        st.stop()

    try:
        model, device, device_name = load_production_model(CHECKPOINT_PATH)
    except Exception as e:
        st.error(f"Error loading production model: {e}")
        st.stop()

    detector, det_device, det_device_name = load_pest_detector(PEST_CHECKPOINT_PATH)
    yolo_detector = load_yolov8_detector(YOLO_CHECKPOINT_PATH)

    # Sidebar Navigation - exactly 4 primary pages
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f345.svg", width=56)
        st.title("Navigation")

        nav_options = [
            "🏠 Home & Overview",
            "🔬 Disease Classification",
            "🐛 Pest Damage Detection",
            "📹 Video Surveillance",
        ]

        default_index = 0
        if "page" in st.query_params:
            param = str(st.query_params["page"]).lower()
            for idx, opt in enumerate(nav_options):
                if param in opt.lower():
                    default_index = idx
                    break

        page = st.radio(
            "Select System Module:",
            nav_options,
            index=default_index,
        )

    # Route navigation to 4 primary pages
    if page.startswith("🏠 Home"):
        render_home_page()
    elif page.startswith("🔬 Disease"):
        render_classification_page(model, device, idx_to_class)
    elif page.startswith("🐛 Pest"):
        render_pest_detection_page(detector, det_device, yolo_detector=yolo_detector)
    elif page.startswith("📹 Video"):
        render_video_surveillance_page()


if __name__ == "__main__":
    main()
