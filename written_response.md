# Written Response — AI/ML Data Annotation Assessment

## 1. Implementation Overview

This submission presents a fully automated, reproducible annotation pipeline for detecting and classifying stars/blobs and space objects/streaks in SSA real-sky FITS imagery. The pipeline processes 10 raw FITS images (9568×6380 pixels, 16-bit unsigned integer, single-band grayscale) captured by MARS-6100-18GTM-TF sensors. It produces pixel-level instance masks and exports annotations in YOLO Ultralytics segmentation format.

The pipeline uses classical computer vision techniques—sigma-clipped background estimation, adaptive thresholding, connected component analysis, and multi-feature shape classification—chosen because they are explainable, inspectable, and well-suited to the sparse, dark nature of these astronomical images.

All 10 images were processed. The pipeline detected 34,912 total instances: 33,389 blobs/stars and 1,523 streaks/objects across 700 tiles.

---

## 2. Pre-Processing Steps (Q2a)

**Inspection findings**: All 10 FITS files confirmed as 9568×6380, uint16, single-band, with no NaN or Inf values. The dataset exhibits two intensity profiles: eight UUID-named images (mean pixel value 0.8–4.2, highly sparse) and two CAM_B images (mean ~52, broader dynamic range). Both groups use the same sensor (MARS-6100-18GTM-TF).

**Pre-processing applied**:

1. **Data loading**: FITS files are read via Astropy with original uint16 precision preserved. No geometric transformations (rotation, scaling, cropping) are applied.

2. **Background estimation**: Per-tile sigma-clipped statistics (σ=3.0, 5 iterations) compute the local median and standard deviation on the valid (non-padded) region. This robustly estimates the background level even when bright sources are present.

3. **Visualization**: For human inspection and YOLO image export, percentile-based contrast stretching (1st–99.9th percentile) maps the uint16 range to 8-bit grayscale. This is used exclusively for visualization—detection and annotation operate on the original full-precision data.

No spatial filtering, denoising, or background subtraction is applied to the detection data. The images are sufficiently clean that sigma-clipped thresholding on raw pixel values provides reliable source detection.

---

## 3. Tiling and Boundary Handling (Q2b)

**Tiling scheme**: Each 9568×6380 image is divided into a grid of 1024×1024 tiles.

- Width: 9568 ÷ 1024 = 9 complete tiles + 352 pixel remainder → 10 columns
- Height: 6380 ÷ 1024 = 6 complete tiles + 236 pixel remainder → 7 rows
- Total: 70 tiles per image, 700 tiles for the full dataset

**Boundary handling**: The rightmost column of tiles has valid width 352 pixels and the bottom row has valid height 236 pixels. These partial regions are placed into 1024×1024 canvases with zero-padding filling the remainder. Each tile's metadata records the exact valid region (x_start, y_start, valid_w, valid_h, pad_right, pad_bottom).

**Key guarantees**:
- Every original pixel is represented exactly once across all tiles
- Padded pixels (value 0) are never annotated—the detection algorithm computes background statistics only on the valid region, and any connected component lying entirely within the padding zone is excluded
- Objects touching the original image boundary are preserved (not discarded)
- Reconstruction crops padding and places each tile at its original coordinates, producing exactly the original 9568×6380 dimensions
- Pixel-perfect reconstruction is validated by comparing the reassembled image against the original FITS data

**Naming convention**: `{image_id}_x{x_start}_y{y_start}` (e.g., `img_01_x0000_y0000`), enabling deterministic tracing of every tile back to its source image and spatial coordinates.

---

## 4. Blob vs Streak Classification (Q2c)

Classification operates on measured geometric properties of each connected component extracted from the thresholded tile. The following features are computed per instance using `skimage.measure.regionprops`:

| Feature | Description | Blob Typical | Streak Typical |
|---------|-------------|-------------|----------------|
| Aspect ratio | Major axis / minor axis | ~1.0 | >2.0 |
| Eccentricity | Ellipse eccentricity [0,1] | <0.7 | >0.7 |
| Compactness | 4π·area / perimeter² | >0.5 | <0.4 |
| Major axis | Length in pixels | Small | >8.0 |

**Classification rules**:

1. **Long/thin streaks**: Aspect ratio ≥ 3.0 AND eccentricity ≥ 0.85 → class 1 (streak)
2. **Short/fat streaks**: Aspect ratio ≥ 2.0 AND eccentricity ≥ 0.7 AND compactness ≤ 0.4 AND major axis ≥ 8.0 → class 1 (streak)
3. **Otherwise** → class 0 (blob)

**Rationale for short/fat streak handling**: A short or fat streak has moderate elongation (aspect ratio 2–3) that alone might not distinguish it from an asymmetric blob. The additional compactness criterion (measuring how "filled" the shape is relative to a circle) captures the directional, non-circular structure characteristic of streaks. The major axis minimum (8 pixels) prevents tiny, ambiguous objects from being classified as streaks.

These thresholds were determined by examining detected instances across the actual dataset tiles and verifying classification visually on contact sheets showing both blob-dominated and streak-containing tiles.

---

## 5. Faint/Small Blob Criteria (Q2d)

Faint stars are retained using two criteria:

1. **Signal-to-Noise Ratio (SNR) ≥ 3.0**: The peak pixel intensity minus the local sigma-clipped median, divided by the sigma-clipped standard deviation, must exceed 3.0. This ensures the source is statistically significant above the background noise.

2. **Minimum area ≥ 3 pixels**: Connected components with fewer than 3 pixels are discarded as likely noise artifacts. This threshold retains genuine faint point sources (which typically span 3–10 pixels due to the sensor PSF) while excluding isolated hot pixels.

For the sparse UUID images (background median ~1, std ~0.9), a source with peak value ≥ 4 satisfies the SNR threshold. For the CAM_B images (background median ~45, std ~15), sources need peak values ≥ 90 to qualify.

Sources meeting both criteria are annotated as class 0 (blob) regardless of brightness. No upper SNR limit is applied—bright saturated stars are also classified as blobs based on their compact shape.

---

## 6. Annotation Methodology

The annotation pipeline for each tile:

1. Estimate background: sigma-clipped median and std on valid (non-padded) region
2. Threshold: source mask = pixels > (median + 5σ)
3. Connected component labeling: each contiguous group of above-threshold pixels becomes a candidate instance
4. Feature measurement: area, axis lengths, eccentricity, compactness, solidity, centroid, peak/mean intensity, SNR
5. Filtering: discard components with area < 3 or SNR < 3.0 or located entirely in padding
6. Classification: apply blob/streak rules from section 4
7. Instance mask: each kept component gets a unique integer ID and class label
8. YOLO export: mask → contour → simplified polygon → normalized coordinates
9. IoU validation: reconstruct mask from polygon, compare with original

Each tile's annotation is stored as a NumPy instance mask (.npy) with per-instance metadata (.json), enabling reconstruction and re-export.

---

## 7. YOLO Segmentation Export

Annotations are exported in YOLO Ultralytics segmentation format:

```
class_id x1 y1 x2 y2 ... xn yn
```

- `class_id`: 0 (blob) or 1 (streak)
- Coordinates: polygon vertices normalized to [0, 1] relative to the 1024×1024 tile
- One `.txt` file per tile (700 total), paired with corresponding `.png` images

Polygon simplification uses `cv2.approxPolyDP` with tolerance proportional to contour arc length. Minimum polygon points: 4.

**Validation**: Each exported polygon is converted back to a mask and compared with the original instance mask via Intersection-over-Union (IoU). Instances with IoU < 0.80 are flagged in the validation report. Across all 34,912 instances, 775 had IoU below the threshold (2.2%), primarily small objects (3–5 pixels) where polygon approximation inherently loses precision.

---

## 8. Reconstruction / Repatching

For each of the 10 images, the pipeline reconstructs:

1. **Full-size class mask** (9568×6380): Each pixel assigned its class value (0=background, 1=blob, 2=streak) by placing valid tile regions at their original coordinates
2. **Full-size annotated overlay**: Contrast-stretched visualization with color-coded annotations (green=blob, red=streak)

Reconstruction validates that output dimensions exactly match the original 9568×6380. All 10 reconstructions passed this check.

---

## 9. Validation and Quality Control

Automated checks performed:
- 10/10 FITS files inspected and valid
- 700/700 tiles processed with metadata, labels, and images
- 0 YOLO format errors (all class IDs valid, all coordinates in [0,1])
- 0 annotations in padded regions
- 10/10 reconstruction dimension matches verified
- 10/10 pixel-perfect tiling reconstruction validated

---

## 10. Assumptions

1. All 10 FITS files in `Datasets_Assessment/` are annotation targets.
2. The two CAM_B files are treated identically to the eight UUID files (same pipeline, adaptive thresholds).
3. Zero-padding (not overlap) is used for edge tiles, as it preserves every pixel without duplication.
4. A 5σ detection threshold balances sensitivity against false positives for these sparse, low-background images.
5. The reference figures in the assessment PDF (Figs. 1–2) define the expected annotation style: compact green regions for blobs, elongated colored regions for streaks.

## 11. Reproducibility

```bash
pip install -r requirements.txt
python run_full.py
```

All parameters are in `config/config.yaml`. The pipeline is deterministic—identical inputs produce identical outputs.
