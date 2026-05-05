# Project Documentation

## AI Vision Pipeline: Object Detection + Segmentation + Metric Depth + 3D Reconstruction

---

**Author:** Jayaprakash A.R.  
**Role:** AI Engineer & Computer Vision Specialist  
**Repository:** [Object-Detection-Segmentation-Metric-Depth-3-D](https://github.com/jayaprakash2207/Object-Detection-Segmentation-Metric-Depth-3-D)  
**License:** MIT  
**Version:** 1.0  
**Date:** May 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Objectives](#2-project-objectives)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Pipeline Modules](#5-pipeline-modules)
   - 5.1 [Module 1 – Object Detection (YOLOv8)](#51-module-1--object-detection-yolov8)
   - 5.2 [Module 2 – Instance Segmentation (SAM)](#52-module-2--instance-segmentation-sam)
   - 5.3 [Module 3 – Metric Depth Estimation (Apple Depth Pro)](#53-module-3--metric-depth-estimation-apple-depth-pro)
   - 5.4 [Module 4 – 3D Coordinate Transformation & Distance Measurement](#54-module-4--3d-coordinate-transformation--distance-measurement)
   - 5.5 [Module 5 – Interactive 3D Visualization (Plotly)](#55-module-5--interactive-3d-visualization-plotly)
   - 5.6 [Module 6 – Custom YOLOv8 Training (Domain-Specific)](#56-module-6--custom-yolov8-training-domain-specific)
   - 5.7 [Module 7 – Flower Detection + Color Analysis (K-Means)](#57-module-7--flower-detection--color-analysis-k-means)
6. [Data Flow & Processing Logic](#6-data-flow--processing-logic)
7. [Installation & Setup](#7-installation--setup)
8. [How to Run](#8-how-to-run)
9. [Output Description](#9-output-description)
10. [Performance & Hardware Requirements](#10-performance--hardware-requirements)
11. [Key Design Decisions](#11-key-design-decisions)
12. [Limitations & Known Issues](#12-limitations--known-issues)
13. [Future Roadmap](#13-future-roadmap)
14. [References](#14-references)

---

## 1. Executive Summary

This project delivers a **multi-stage AI vision pipeline** that transforms a single RGB image into rich, quantifiable spatial intelligence. By chaining together four state-of-the-art deep-learning models — YOLOv8 (object detection), SAM (instance segmentation), Apple Depth Pro (metric depth estimation), and Plotly (3D visualization) — the system produces precise, real-world 3D positions and inter-object distance measurements from any photograph.

In addition to the core pipeline, the project includes:
- Custom YOLOv8 model training on domain-specific datasets (safety/medical equipment and flowers).
- Intelligent color analysis using K-Means clustering to identify the dominant colors of detected objects.

The pipeline is implemented as both a **Google Colab Jupyter Notebook** (for interactive research use) and a **standalone Python script** (for production deployment). All code is open-source under the MIT License.

---

## 2. Project Objectives

| # | Objective |
|---|-----------|
| 1 | Detect and localize all objects in a 2D image with bounding boxes. |
| 2 | Generate precise pixel-level segmentation masks for each detected object. |
| 3 | Estimate absolute metric depth (in meters) at every pixel of the scene. |
| 4 | Transform each detected object's 2D position into a 3D world coordinate (X, Y, Z in meters). |
| 5 | Compute Euclidean distances between any pair of objects in the scene in real-world units. |
| 6 | Render an interactive 3D point-cloud visualization with annotated object markers and distances. |
| 7 | Train custom YOLOv8 detectors for specialized object categories. |
| 8 | Perform automated color classification for detected objects using unsupervised clustering. |

---

## 3. System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                        Input Layer                            │
│              Single RGB Image (JPG / PNG)                     │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  Module 1 – Object Detection                                  │
│  Model: YOLOv8n (Ultralytics)                                 │
│  Output: Bounding boxes [x1, y1, x2, y2], class labels,      │
│          confidence scores                                    │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  Module 2 – Instance Segmentation                             │
│  Model: SAM ViT-B (Meta AI)                                   │
│  Input: Bounding boxes from Module 1                          │
│  Output: Binary masks (H × W bool) per detected object        │
└──────────────┬────────────────────────────┬───────────────────┘
               │                            │
               ▼                            ▼
┌──────────────────────────┐  ┌─────────────────────────────────┐
│  Module 3 – Metric Depth │  │  Module 7 – Color Analysis      │
│  Model: Apple Depth Pro  │  │  (Flower pipeline only)         │
│  Output: Depth map (m),  │  │  K-Means clustering on masked   │
│          focal length    │  │  flower pixels → dominant color  │
└──────────────┬───────────┘  └─────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────────┐
│  Module 4 – 3D Coordinate Transformation                      │
│  Formula: X = (u - cx) × Z / f                                │
│           Y = (v - cy) × Z / f                                │
│           Z = mean depth over SAM mask                        │
│  Output: (X, Y, Z) in metres for each object                  │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  Module 4b – Distance Calculation                             │
│  Formula: d = ‖P₁ − P₂‖₂  (Euclidean, 3D)                    │
│  Output: Distance in meters and centimeters                   │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  Module 5 – Interactive 3D Visualization (Plotly)             │
│  Output: Point cloud + object spheres + distance line (HTML)  │
└───────────────────────────────────────────────────────────────┘

                 ┌──────────────────────────┐
                 │  Module 6 – Custom       │
                 │  YOLOv8 Training         │
                 │  (Parallel branch,       │
                 │   offline training step) │
                 └──────────────────────────┘
```

---

## 4. Technology Stack

| Category | Library / Tool | Version | Role in Project |
|---|---|---|---|
| Language | Python | 3.8+ | Primary implementation language |
| Deep Learning | PyTorch | 2.0+ | Model inference and training backend |
| Object Detection | Ultralytics YOLOv8 | Latest | Real-time object detection |
| Instance Segmentation | Segment Anything (SAM) | Meta ViT-B | Pixel-level object masks |
| Metric Depth | Apple Depth Pro | Latest | Absolute depth in metres |
| Image Processing | OpenCV (cv2) | 4.8+ | Image I/O, drawing, color mapping |
| Visualization | Plotly | Latest | Interactive 3D scene rendering |
| Scientific Computing | NumPy | 2.0+ | Array math and transformations |
| Machine Learning | scikit-learn | Latest | K-Means color clustering |
| Dataset Management | Roboflow | Latest | Dataset download (flowers, custom) |
| Notebook Environment | Google Colab | — | GPU-accelerated interactive runtime |
| YAML Processing | PyYAML | Latest | Training configuration files |

---

## 5. Pipeline Modules

### 5.1 Module 1 – Object Detection (YOLOv8)

**Purpose:** Locate all objects in the image and classify them with confidence scores.

**Model:** `yolov8n.pt` (nano variant, COCO-pretrained). Can be swapped for `yolov8s/m/l/x` for higher accuracy at the cost of speed.

**Confidence threshold:** `0.25` (configurable via `YOLO_CONF`).

**Key function:**
```python
def run_yolo(img_rgb, conf=YOLO_CONF):
    r = yolo_model(img_rgb, conf=conf, verbose=False)[0]
    out = []
    for b in r.boxes:
        out.append({
            'label':      r.names[int(b.cls)],
            'confidence': float(b.conf),
            'bbox':       b.xyxy[0].cpu().numpy().tolist(),  # [x1, y1, x2, y2]
        })
    return out
```

**Output:** A list of detection dictionaries with `label`, `confidence`, and `bbox` (pixel coordinates).

---

### 5.2 Module 2 – Instance Segmentation (SAM)

**Purpose:** For each YOLOv8 bounding box, produce a precise pixel-level binary mask of the object.

**Model:** SAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`, ~375 MB). The predictor is initialized once and reused for all boxes in an image.

**Approach:** Box-prompted prediction — the detected bounding box is passed directly to SAM as a prompt, making it guided and efficient without requiring manual point selection.

**Key function:**
```python
def run_sam(img_rgb, dets):
    sam_predictor.set_image(img_rgb)
    masks = []
    for d in dets:
        box = np.array(d['bbox'], dtype=float)
        m, _, _ = sam_predictor.predict(
            point_coords=None, point_labels=None,
            box=box, multimask_output=False,
        )
        masks.append(m[0].astype(bool))
    return masks
```

**Output:** One boolean `(H × W)` mask per detection.

---

### 5.3 Module 3 – Metric Depth Estimation (Apple Depth Pro)

**Purpose:** Estimate absolute depth (in metres) at every pixel of the input image without requiring any calibration information.

**Model:** Apple Depth Pro (`depth_pro.pt`, ~2.5 GB). Unlike relative depth models, Depth Pro outputs metric distances, making it directly usable for real-world measurement.

**Precision:** Uses `torch.half` (FP16) on GPU for speed; falls back to `torch.float32` on CPU.

**Key function:**
```python
def run_depth(img_rgb):
    pil  = PILImage.fromarray(img_rgb)
    tens = dp_transform(pil).to(device)
    with torch.no_grad():
        pred = dp_model.infer(tens, f_px=None)
    depth_m = pred['depth'].cpu().numpy().astype(np.float32)
    f_px    = float(pred['focallength_px'])
    return depth_m, f_px
```

**Outputs:**
- `depth_m`: Float32 array of shape `(H, W)` — depth in metres per pixel.
- `f_px`: Estimated focal length in pixels (used for the 3D back-projection in Module 4).

---

### 5.4 Module 4 – 3D Coordinate Transformation & Distance Measurement

**Purpose:** Convert each detected object from 2D image coordinates to 3D real-world coordinates, then compute inter-object distances.

**3D Back-Projection Formula (pinhole camera model):**

```
X = (u - cx) × Z / f
Y = (v - cy) × Z / f
Z = mean(depth_m[mask])
```

Where:
- `(u, v)` = pixel centre of the bounding box
- `(cx, cy)` = image principal point (assumed to be image centre)
- `Z` = mean depth over the object's SAM mask (in metres)
- `f` = focal length in pixels (from Depth Pro)

**Euclidean Distance:**
```
d = sqrt((X₁-X₂)² + (Y₁-Y₂)² + (Z₁-Z₂)²)
```

**Key functions:**
```python
def to_3d(u, v, z, img_hw, f_px):
    h, w   = img_hw
    cx, cy = w / 2.0, h / 2.0
    return (u - cx) * z / f_px, (v - cy) * z / f_px, z

def build_objects(dets, masks, depth_m, f_px):
    h, w = depth_m.shape
    objs = []
    for d, mask in zip(dets, masks):
        x1, y1, x2, y2 = d['bbox']
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        z      = float(depth_m[mask].mean()) if mask.any() else 0.0
        objs.append({
            'label':     d['label'],
            'depth_m':   z,
            'xyz':       to_3d(cx, cy, z, (h, w), f_px),
        })
    return objs
```

**Output:** For each detected object: label, 2D center, depth in metres, and 3D world coordinate `(X, Y, Z)`.

---

### 5.5 Module 5 – Interactive 3D Visualization (Plotly)

**Purpose:** Render the full scene in an interactive 3D environment so users can explore the spatial layout of detected objects.

**Visualization elements:**
| Element | Description |
|---|---|
| Point cloud | Sub-sampled scene points coloured with original image colours (stride = 6 pixels) |
| Red sphere | First selected object's 3D position |
| Blue sphere | Second selected object's 3D position |
| Green line | Line connecting the two objects, annotated with the distance in centimetres |

**Key parameters:**
- Sphere radius: `0.04 m`
- Point cloud stride: every 6 pixels (for performance)
- Chart height: 650 px

**Export:** The 3D scene is saved as a standalone interactive HTML file (`scene_3d.html`) that can be viewed in any modern browser without additional dependencies.

---

### 5.6 Module 6 – Custom YOLOv8 Training (Domain-Specific)

**Purpose:** Fine-tune YOLOv8n on custom, domain-specific datasets to detect objects not present in the standard COCO vocabulary.

**Dataset 1 – Safety/Medical Equipment (5 classes):**
- Ambulance, Safety Vest, Stethoscope, Surgical Mask, Ceiling Fan
- 5 annotated training images
- 100 epochs, image size 640×640, batch size 4

**Dataset 2 – Flowers (multiple classes):**
- Sourced via Roboflow (`flowers-0elmu`, version 1)
- 100 epochs, image size 640×640, batch size 16
- Best weights saved to `best.pt` and downloaded from Colab

**Training configuration (programmatically generated `data.yaml`):**
```yaml
path:  /content/<dataset_path>
train: /content/<dataset_path>/train/images
val:   /content/<dataset_path>/valid/images
nc:    <number_of_classes>
names: [<class_1>, <class_2>, ...]
```

**Evaluation metrics reported after training:**
- `mAP50` — mean Average Precision at IoU 0.50
- `mAP50-95` — mean Average Precision averaged over IoU thresholds 0.50–0.95

---

### 5.7 Module 7 – Flower Detection + Color Analysis (K-Means)

**Purpose:** For each detected flower, identify its primary and secondary colors using unsupervised machine learning.

**Process:**
1. The custom flower YOLOv8 model detects flowers in the uploaded image (confidence ≥ 0.5, falls back to 0.1).
2. SAM generates a precise mask for each flower bounding box.
3. Pixels under the mask are extracted. Near-white (`> 245`) and near-black (`< 20`) pixels are filtered out to remove background noise.
4. K-Means clustering (`k=5`) is applied to the remaining RGB pixels.
5. Each cluster center is mapped to the nearest named color from a 22-color reference dictionary using Euclidean distance in RGB space.
6. Results are displayed as a panel: annotated image + 5 color swatches with percentages and RGB values.

**Color reference palette (22 colors):**
Red, Dark Red, Pink, Hot Pink, Orange, Yellow, White, Off White, Purple, Dark Purple, Violet, Lavender, Blue, Light Blue, Green, Light Green, Brown, Cream, Peach, Coral, Magenta, Gray.

**Output:** For each flower — detected class name, confidence, dominant color name, and a full ranked color breakdown with percentages.

---

## 6. Data Flow & Processing Logic

The following table describes the sequential data transformations from input to output:

| Step | Input | Process | Output |
|---|---|---|---|
| 1 | Raw image bytes | OpenCV decode → RGB array | `IMAGE` (H × W × 3, uint8) |
| 2 | `IMAGE` | YOLOv8 inference | `dets` list of `{label, conf, bbox}` |
| 3 | `IMAGE`, `dets` | SAM box-prompted prediction | `masks` list of bool `(H × W)` arrays |
| 4 | `IMAGE` | Depth Pro inference | `depth_m` (H × W, float32), `f_px` (float) |
| 5 | `dets`, `masks`, `depth_m`, `f_px` | Back-projection + depth averaging | `objects` list with `xyz` tuples |
| 6 | `objects[A]`, `objects[B]` | Euclidean distance | `dist_m`, `dist_cm` |
| 7 | `depth_m`, `IMAGE`, `objects` | Plotly graph construction | Interactive HTML visualization |
| 8 | `IMAGE`, flower `dets`, SAM masks | K-Means on masked pixels | Dominant color names + percentages |

---

## 7. Installation & Setup

### Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.10+ |
| GPU | Any CUDA | NVIDIA T4 / RTX 30-series |
| VRAM | 4 GB | 8 GB+ |
| RAM | 8 GB | 16 GB+ |
| Disk space | 10 GB | 15 GB+ |

### Dependency Installation

```bash
# Step 1 — Core libraries
pip install -U ultralytics
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/apple/ml-depth-pro.git
pip install 'numpy>=2.0,<3.0' --force-reinstall

# Step 2 — Supporting libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python plotly scikit-learn roboflow pyyaml pillow matplotlib
```

> **Note:** Run the installation cell first, then **restart the runtime** before proceeding. This is required because SAM/Depth Pro may downgrade NumPy, which breaks PyTorch's expectations.

### Model Checkpoint Downloads

| Model | Size | Source |
|---|---|---|
| SAM ViT-B (`sam_vit_b_01ec64.pth`) | ~375 MB | `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth` |
| Apple Depth Pro (`depth_pro.pt`) | ~2.5 GB | `https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt` |
| YOLOv8n (`yolov8n.pt`) | ~6 MB | Auto-downloaded by Ultralytics |

Both checkpoints are saved to a local `checkpoints/` directory. The download script checks for existing files to avoid re-downloading.

---

## 8. How to Run

### Option A – Google Colab (Recommended for first use)

1. Open `Object Detection + Segmentation + Metric Depth + 3-D.ipynb` in Google Colab.
2. Set runtime: **Runtime → Change runtime type → T4 GPU**.
3. Run **Cell 1** (installs packages). Wait for "Installation complete."
4. **Restart session** (Runtime → Restart session).
5. Skip Cell 1. Run **Cell 2** (downloads model checkpoints, ~2.9 GB).
6. Run **Cell 3** (loads all three models into GPU memory).
7. Run remaining cells in order. Upload an image when prompted.

### Option B – Python Script

```bash
python "Object Detection + Segmentation + Metric Depth + 3-D python.py"
```

### Execution Order of Cells/Sections

| Cell | Description |
|---|---|
| Cell 1 | Package installation (run once, then restart) |
| Cell 2 | Checkpoint download |
| Cell 3 | Model loading (YOLOv8, SAM, Depth Pro) |
| Cell 4 | Utility functions (`decode_b64`, `show_img`, etc.) |
| Cell 5 | Image upload and display |
| Cell 6 | YOLOv8 → SAM → Depth Pro → object list |
| Cell 7 | Distance calculation + Plotly 3D visualization |
| Cell 8 | Export results (depth map PNG, detections PNG, 3D HTML) |
| Cell 9 | Custom YOLOv8 training (safety/medical dataset) |
| Cell 10 | Custom model inference test |
| Cell 11–13 | Flower dataset download + training + validation |
| Cell 14 | Flower detection + K-Means color analysis |

---

## 9. Output Description

The pipeline produces the following outputs:

| Output | Format | Description |
|---|---|---|
| `detections.png` | PNG image | Original image with YOLOv8 bounding boxes and labels overlaid |
| `depth_map.png` | PNG image | False-color depth map using the Plasma colormap (dark = near, bright = far) |
| `scene_3d.html` | HTML file | Fully interactive 3D Plotly scene with point cloud, object spheres, and distance annotation |
| Console table | Terminal / Notebook output | Per-object report: label, confidence, depth in metres, 3D world coordinates |
| Distance report | Terminal / Notebook output | `Object A ↔ Object B: X.XXXX m (XXX.X cm)` |
| `best.pt` | PyTorch model file | Fine-tuned YOLOv8 weights for the custom-trained domain |
| Color analysis panel | Matplotlib figure | Per-flower panel: annotated image + 5 color swatches with names, percentages, and RGB values |

**Example console output:**
```
============================================================
  3 object(s) detected
============================================================
  [0] bottle         conf=0.91  depth=0.842 m  3D=(-0.143, 0.021, 0.842)
  [1] bottle         conf=0.87  depth=0.965 m  3D=( 0.201, 0.019, 0.965)
  [2] cup            conf=0.75  depth=1.102 m  3D=(-0.011, 0.045, 1.102)
============================================================

[0] bottle  <->  [1] bottle: 0.3512 m  (35.1 cm)
```

---

## 10. Performance & Hardware Requirements

### Runtime Benchmarks (Google Colab T4 GPU)

| Stage | Approximate Time |
|---|---|
| YOLOv8 inference | < 0.1 s |
| SAM mask prediction (per object) | 0.3 – 0.8 s |
| Depth Pro inference | 10 – 20 s |
| 3D construction + Plotly render | < 1 s |
| **Total end-to-end** | **~15 – 25 s per image** |

### Accuracy Benchmarks

| Metric | Value |
|---|---|
| YOLOv8n COCO mAP50 | ~52.0% (standard benchmark) |
| SAM mask quality (IoU) | > 0.92 (box-prompted) |
| Depth Pro absolute depth error | < 5 cm (typical indoor scenes) |

### Hardware Scaling

| Hardware | Feasibility | Notes |
|---|---|---|
| NVIDIA T4 GPU (Colab) | ✅ Recommended | 15–25 s / image |
| NVIDIA RTX 30/40-series | ✅ Optimal | < 10 s / image |
| CPU only | ⚠️ Possible | Very slow (minutes per image); FP32 precision |
| Mobile / Edge | ❌ Not supported in current form | Requires quantization |

---

## 11. Key Design Decisions

### 1. Box-prompted SAM instead of automatic segmentation
Using each YOLOv8 bounding box as a prompt to SAM ensures that exactly one mask is generated per detected object. Automatic SAM segmentation would generate hundreds of masks without semantic labels. This design maintains a strict 1-to-1 correspondence between detections and masks throughout the pipeline.

### 2. Mean depth over SAM mask (not bounding box center)
The object's representative depth `Z` is computed as the mean of all depth values within its SAM mask, not just the single pixel at the bounding box center. This produces a more stable and accurate depth estimate, particularly for non-rectangular objects or when the background is partially included in the bounding box.

### 3. Focal length from Depth Pro
Apple Depth Pro internally estimates the camera's focal length as part of its metric depth model. This estimated focal length is reused in the pinhole back-projection formula, ensuring geometric consistency without requiring camera calibration data.

### 4. Sub-sampled point cloud (stride = 6)
To keep the 3D visualization responsive, the point cloud is built by sampling every 6th pixel in both dimensions rather than all ~300,000+ pixels. This reduces rendering load by ~36× while preserving the visual structure of the scene.

### 5. K-Means with background pixel filtering
Before clustering, extreme near-white and near-black pixels are removed to avoid the K-Means centers being pulled toward background or shadow regions. If filtering removes too many pixels (< 50 remaining), the filter is bypassed to ensure clustering always succeeds.

### 6. Two-phase training strategy
For the custom flower dataset, the project downloads from Roboflow and programmatically rewrites `data.yaml` with absolute paths. This is necessary because Colab's file system paths differ from Roboflow's default relative path conventions.

---

## 12. Limitations & Known Issues

| Limitation | Impact | Mitigation |
|---|---|---|
| Single-image processing | No video/stream support in current implementation | Extend with frame-by-frame loop |
| Depth Pro accuracy degrades outdoors | Metric depth less reliable for very large scenes (> 10 m) | Use domain-adapted depth models for outdoor use |
| YOLOv8n misses small objects | Nano model trades accuracy for speed | Switch to `yolov8m` or `yolov8x` for difficult scenes |
| SAM requires full image embedding | Each `set_image()` call re-computes image features (~0.3 s per object) | Cache the image embedding across all objects |
| Colab API (`files.upload/download`) | Not portable to standalone Python environments | Replace with `argparse` or a web API for production |
| Custom training with only 5 images | Underfitting risk on safety/medical dataset | Expand dataset; apply augmentation |
| Roboflow API key embedded in source | Security concern in public repository | Move to environment variable or secrets manager |

---

## 13. Future Roadmap

| Priority | Feature | Description |
|---|---|---|
| High | Real-time video pipeline | Process video streams frame-by-frame with temporal smoothing |
| High | REST API wrapper | Flask/FastAPI endpoint for production deployment |
| Medium | LiDAR sensor fusion | Fuse LiDAR point clouds with Depth Pro estimates for higher accuracy |
| Medium | Multi-camera stereo vision | Improve depth accuracy using stereo disparity |
| Medium | AR/VR export | Output 3D scene in glTF/USD format for headset display |
| Low | Advanced object tracking | Track object identities across video frames (DeepSORT/ByteTrack) |
| Low | Neural radiance fields | Photorealistic 3D reconstruction with NeRF |
| Low | Multi-modal LLM integration | Add natural language querying of detected scene contents |

---

## 14. References

| Resource | Link |
|---|---|
| YOLOv8 (Ultralytics) | https://github.com/ultralytics/ultralytics |
| Segment Anything Model (SAM) | https://github.com/facebookresearch/segment-anything |
| Apple Depth Pro | https://github.com/apple/ml-depth-pro |
| Plotly Python | https://plotly.com/python/ |
| Roboflow | https://roboflow.com/ |
| PyTorch | https://pytorch.org/ |
| OpenCV | https://opencv.org/ |
| scikit-learn K-Means | https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html |

---

*Document prepared by Jayaprakash A.R. — AI Engineer & Computer Vision Specialist*  
*Contact: jayaprakash22072005@gmail.com*
