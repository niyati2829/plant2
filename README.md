# Tomato Leaf Disease and Pest Surveillance System

An integrated deep learning application for automated tomato plant health monitoring, foliar disease diagnosis, pest damage localization, and field video surveillance.

---

## 🌟 Main Features & Capabilities

1. **Tomato Disease Classification (ResNet18):**
   - Calibrated single-leaf disease diagnosis across **10 conditions** (9 pathologies + healthy).
   - Temperature scaling calibration ($T^* = 1.3021$) providing trustworthy confidence scores.
   - Grad-CAM visual attention heatmaps highlighting the exact foliar regions driving the model's prediction.

2. **Pest Damage Detection (YOLOv8n-Seg):**
   - Instance segmentation localizing *Tuta absoluta* (Tomato Leafminer) larval feeding mines and necrotic tunnels.
   - Identifies and highlights background healthy reference foliage in green.
   - Provides polygon masks, bounding boxes, and quantitative damage area metrics.

3. **Lesion Semantic Segmentation (PlantSeg U-Net):**
   - Pixel-level semantic segmentation isolating necrotic disease lesions across the leaf lamina.
   - Computes percentage foliar lesion coverage to estimate severity.

4. **Automated Field Video Surveillance:**
   - Multi-task temporal analysis combining ResNet18 disease diagnosis and YOLOv8n-Seg pest localization across field video frames.
   - Generates an annotated surveillance video stream, temporal health timeline, and downloadable telemetry CSV.

---

## 🧠 Models Used

| Component | Architecture | Checkpoint | Target Task | Performance |
|---|---|---|---|---|
| **Disease Diagnosis** | Calibrated ResNet18 | `artifacts/checkpoints/best_resnet18_weighted.pt` | 10 Tomato Conditions | 99.23% Accuracy, 0.9885 Macro F1 |
| **Pest Localization** | YOLOv8n-Seg | `artifacts/checkpoints/pest_detection_yolov8/best_yolov8n_seg.pt` | *Tuta absoluta* feeding mines | 71.99% Box mAP@0.5, 70.37% Mask mAP@0.5 |
| **Lesion Segmentation** | PlantSeg U-Net | `artifacts/checkpoints/segmentation_real/best_unet_improved.pt` | Necrotic disease lesions | 0.7410 Test Dice Score |

---

## 📁 Project Structure

```text
crop-disease-detection/
├── app/
│   └── app.py                              # Streamlit 4-page integrated web application
├── configs/
│   └── config.yaml                         # Central project configuration
├── notebooks/
│   ├── 01_disease_classification.ipynb     # Notebook 1: Disease Diagnosis (ResNet18)
│   ├── 02_pest_damage_detection.ipynb      # Notebook 2: Pest Localization (YOLOv8n-Seg)
│   └── 03_video_surveillance.ipynb         # Notebook 3: Field Video Surveillance
├── src/
│   ├── models/                             # ResNet18 architecture definition
│   ├── calibration/                        # Temperature scaling calibration
│   ├── interpretability/                   # Grad-CAM heatmap generation
│   ├── detection/                          # Pest detection data & model utilities
│   ├── segmentation/                       # U-Net lesion segmentation architecture & dataset
│   ├── surveillance/                       # Frame analyzer, annotator, video processor
│   ├── data/                               # Dataset loaders and transforms
│   ├── evaluation/                         # Evaluation metrics, error analysis, plotting
│   └── utils/                              # Hardware accelerator (MPS/CUDA/CPU) detection
├── tests/                                  # Automated unit and integration test suite
├── artifacts/
│   ├── checkpoints/                        # Trained model weights
│   └── figures/                            # Evaluation figures & confusion matrices
├── data/
│   ├── splits/                             # PlantVillage stratified split CSV files
│   ├── pest_detection/                     # Tuta absoluta dataset & archive
│   └── video/                              # Benchmark field surveillance recording
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Environment Setup
```bash
# Clone the repository and navigate into the folder
cd crop-disease-detection

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```
Open your browser to `http://localhost:8501` to explore:
- **Home & Overview**: System summary and verified benchmarks.
- **Disease Classification**: Upload leaf photos for calibrated diagnosis, Grad-CAM, and U-Net lesion segmentation.
- **Pest Damage Detection**: Inspect foliar feeding mines and healthy reference foliage.
- **Video Surveillance**: Run multi-task video analysis and generate field progression timelines.

### 3. Run Automated Tests
```bash
pytest tests/ --disable-warnings
```

---

## 📚 Dataset Attribution & References

- **PlantVillage Dataset**: Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.*
- **PlantSeg Dataset**: Semantic segmentation dataset for foliar plant lesions.
- **TutaAbsolutaResearch v4 Dataset**: Roboflow dataset for *Tuta absoluta* tomato leafminer damage localization.
