# Single-Image Metric 3D Reconstruction

This project reconstructs a coloured 3D point cloud from a **single RGB image**, at **metric scale**: every coordinate is in metres. It needs no camera calibration, EXIF data or second view. The reconstruction runs locally and is shown in an interactive web viewer. You can export it as a PLY file.

For a detailed walkthrough of the pipeline, the flying-pixel filter and how UniDepth V2 differs from V1, see [TECHNICAL_NOTES.md](TECHNICAL_NOTES.md). The full technical report is in [REPORT.md](REPORT.md).

## Method

### 1. Depth and camera estimation (UniDepth V2)

We use **UniDepth V2** ([Piccinelli et al.](https://github.com/lpiccinelli-eth/UniDepth)), checkpoint `lpiccinelli/unidepth-v2-vitl14`. It uses a ViT-L/14 (DINOv2) image encoder. From one image, the model predicts:

- a **metric depth map** Z(u, v) in metres;
- a **camera model**: per-pixel ray directions, from which it derives the pinhole intrinsics K (focal lengths fx, fy and principal point cx, cy);
- a **3D point** for each pixel, obtained by back-projecting the depth along those rays:

```
X = Z(u,v) · K⁻¹ [u, v, 1]ᵀ
```

Because the model estimates the camera itself and was trained on metric ground truth, the output has absolute scale. A classical single-view or monocular pipeline can only recover structure up to an unknown scale factor.

We use the model's point output directly. We checked it against our own back-projection with the predicted K, and the depth (z) values match exactly.

### 2. Post-processing (`engine.py`)

1. **Input normalisation.** The image is rotated according to its EXIF orientation. RGBA, palette, greyscale and 16-bit images are converted to 8-bit RGB. Images whose longest side is over 2048 px are downscaled (Lanczos).
2. **Validity mask.** Pixels with non-finite depth, or depth outside (0.001 m, 80 m), are discarded.
3. **Depth-discontinuity filter.** Single-image depth tends to produce *flying pixels*: points that interpolate between a foreground edge and the background behind it. A pixel is removed if its depth differs from any of its 4 neighbours by more than **4% (relative)**. On the example image this removes about 2% of points, all on object boundaries.
4. **Scene measurements.** The extent is the 1st–99th percentile range along each axis, which ignores stray outliers. We also report min, median and max depth.

### 3. Rendering

The point cloud is rendered in the browser with **Plotly.js** (`plotly.graph_objects.Scatter3d`). This draws in **WebGL**, and the figure is embedded in the page with Gradio's `gr.Plot` component.

- **Axes.** Points are stored in the OpenCV camera frame (x right, y down, z forward). For display they are mapped to x → right, z → depth, −y → up, so the scene appears upright. The axes are equal-scaled (`aspectmode="data"`), so distances on screen are not distorted. Ticks are in metres.
- **Colour.** Each point is coloured either with its pixel's RGB value or by depth (Turbo colormap, clipped to the 2nd–98th percentile, with a colour bar in metres).
- **Interaction.** Plotly's orbit camera: drag to rotate, scroll to zoom, right-drag to pan. Hovering a point shows its x / depth / up coordinates and its straight-line distance from the camera.
- **Point budget.** A 640×480 image gives about 300k points. For responsiveness, the viewer draws a uniform random subset (120k by default, adjustable). The exported file always has every point.
- **No re-inference on display changes.** The model output is cached per session. Changing colour mode, point size, point budget or the edge filter only rebuilds the figure.

We chose Plotly over `gr.Model3D` because Gradio's Model3D treats `.ply` files as Gaussian splats and draws raw point clouds at 1 px, with no metric axes or per-point readout.

### 4. Export

The **binary little-endian PLY** has `float x, y, z` (metres, camera frame) and `uchar red, green, blue` for each vertex. It is written directly with NumPy (no Open3D dependency) and opens in MeshLab, CloudCompare, Blender and Open3D.

## Results

Measured on UniDepth's sample indoor image (`examples/indoor_room.png`, 640×480), which comes with ground-truth depth and intrinsics:

| Quantity | Value |
|---|---|
| Depth error vs. ground truth (AbsRel) | 7.6% |
| Focal length error (fx / fy) | +3.2% / +4.5% |
| Reconstructed room extent (w × h × d) | 5.77 × 3.00 × 4.58 m |
| Depth range | 1.67 – 6.90 m |
| Points | 301,188 (after filtering) |
| Inference time | 6–16 s on a laptop CPU (no GPU) |

## Software

| Component | Used for | Version tested |
|---|---|---|
| UniDepth V2 (ViT-L/14) | Metric depth, camera and 3D points | `unidepth` 0.1 (GitHub) |
| PyTorch | Model inference | 2.14.1 (CPU) |
| timm, einops, huggingface-hub | UniDepth dependencies, weight download | timm 1.0.30 |
| Gradio | Web UI | 5.50.0 |
| Plotly | WebGL 3D point-cloud rendering | 7.1.0 |
| NumPy | Filtering, measurements, PLY writing | 2.4.6 |
| Pillow | Image loading and preprocessing | 11.3.0 |
| Python | | 3.11 |

## Project structure

```
engine.py        Model wrapper (UniDepthEngine), preprocessing, edge filter, PLY export
app.py           Gradio interface and Plotly rendering
examples/        Sample input image
outputs/         Exported PLY files (created at runtime)
requirements.txt Python dependencies
```

## Setup

Requires Python 3.10 or 3.11; UniDepth does not support 3.13 or newer yet.

```powershell
conda create -y -n unidepth python=3.11
conda activate unidepth

# PyTorch: choose one
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu      # CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121    # NVIDIA GPU

# UniDepth without its pinned dependencies (triton/xformers are optional and have no Windows wheels)
pip install --no-deps "unidepth @ git+https://github.com/lpiccinelli-eth/UniDepth.git"

pip install -r requirements.txt
```

## Usage

```powershell
conda activate unidepth
python app.py --preload
```

Open http://127.0.0.1:7860, upload an image (or click the example), and click **Reconstruct**. The first run downloads about 1.3 GB of model weights from Hugging Face.

| Option | Effect |
|---|---|
| `--preload` | Load the model before starting the server |
| `--device cpu\|cuda` | Force a device (default: CUDA if available) |
| `--port N` / `--host H` | Server address (default 127.0.0.1:7860) |
| `--share` | Create a temporary public Gradio link |
| `--resolution-level 0..9` | Override UniDepth's internal resolution (lower is faster) |

The startup warnings about xFormers, KNN, `EdgeGuidedLocalSSI` and `NystromAttention` are harmless. They concern training code and GPU kernels that inference doesn't use.

## Limitations

- **Pinhole camera only.** Fisheye, ultra-wide and panoramic images give distorted geometry.
- **One view only.** Surfaces hidden from the camera are missing, so the cloud is a single-view surface, not a closed model.
- **Accuracy varies with the scene.** It is best for indoor rooms and street scenes with familiar objects. It is unreliable for sky, mirrors, glass, textureless close-ups and cropped or zoomed images.
- **Depth error grows with distance.** Depth is capped at 80 m.

## Credits

UniDepth: L. Piccinelli et al., *UniDepth: Universal Monocular Metric Depth Estimation* (CVPR 2024) and *UniDepthV2* (2025). The model weights and the sample image come from the UniDepth repository and are under its CC BY-NC 4.0 licence (non-commercial use only).
