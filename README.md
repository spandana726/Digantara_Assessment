# Digantara AI/ML Data Annotation Intern - Assessment Submission

## Overview

Automated annotation pipeline for detecting and classifying **blobs/stars** and **streaks/space objects** in SSA (Space Situational Awareness) real-sky FITS imagery. The pipeline produces pixel-level instance masks and exports annotations in YOLO Ultralytics segmentation format.

## Assessment Requirements Addressed

| Requirement | Status |
|-------------|--------|
| A1: Inspect imagery and apply preprocessing | ✅ |
| A2: Divide into 1024×1024 tiles | ✅ |
| A3: Annotate blobs and streaks as two classes | ✅ |
| Q2a: Preprocessing steps documented | ✅ |
| Q2b: Boundary handling explained | ✅ |
| Q2c: Short/fat streak vs blob criteria | ✅ |
| Q2d: Faint/small blob criteria | ✅ |
| Repatched full-size imagery | ✅ |
| YOLO Ultralytics segmentation format | ✅ |
| Custom scripts via GitHub | ✅ |
| Written response (≤4 pages) | ✅ |

## Environment Setup

### Requirements
- Python 3.10+
- Dependencies listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

### Dataset Placement

Place the 10 raw FITS files in:
```
Digantara_Assessment/Datasets_Assessment/Datasets_Assessment/
```

The pipeline expects exactly 10 FITS files with dimensions 9568×6380 pixels.

## How to Run

### Full Pipeline

```bash
python run_full.py
```

This executes all stages sequentially:
1. FITS inspection and validation
2. Image processing (tiling + annotation + YOLO export)
3. Output validation
4. Summary report generation

Estimated runtime: ~20 minutes for all 10 images on a standard machine.

### Individual Stages

Each module in `src/` can be run independently:

```bash
python src/inspect_fits.py     # Inspect FITS files
python src/run_pipeline.py     # Full pipeline
```

## Pipeline Stages

### 1. FITS Inspection (`src/inspect_fits.py`)
Validates all FITS files: dimensions, dtype, value range, NaN/Inf checks. Generates `reports/fits_inspection.txt`.

### 2. Preprocessing (`src/preprocess.py`)
- No geometric modifications to raw data
- Percentile-based contrast stretching (1st–99.9th percentile) for **visualization only**
- Detection and annotation operate on the original uint16 data

### 3. Tiling (`src/tile_images.py`)
- Divides each 9568×6380 image into 1024×1024 tiles
- Grid: 10 columns × 7 rows = 70 tiles per image
- Edge tiles zero-padded to 1024×1024; valid region tracked in metadata
- Pixel-perfect reconstruction validated against original data

### 4. Annotation (`src/annotate.py`)
- **Background estimation**: Sigma-clipped statistics (σ=3.0, 5 iterations) per tile on uint16 data
- **Source detection**: Pixels above `median + 5σ` threshold
- **Connected component labeling**: Each source is a separate instance
- **Feature measurement**: Area, major/minor axis, eccentricity, aspect ratio, compactness, solidity, SNR
- **Classification**: Multi-feature blob vs streak (see criteria below)
- **Padding exclusion**: Detections in padded regions are excluded; boundary objects are preserved

### 5. YOLO Export (`src/masks_to_yolo.py`)
- Instance masks → contours → simplified polygons → normalized [0,1] coordinates
- Format: `class_id x1 y1 x2 y2 ... xn yn`
- Polygon→mask IoU validation for quality control

### 6. Reconstruction (`src/reconstruct.py`)
- Repatches tiles to original 9568×6380 dimensions
- Validates reconstructed dimensions match original
- Generates full-size annotated overlay images

### 7. Visualization (`src/visualize.py`)
- Per-tile overlay: green=blob, red=streak
- Contact sheets for quick inspection
- Full-size reconstructed overlays

### 8. Validation (`src/validate.py`)
- Checks all 700 tiles have metadata, labels, and images
- Validates YOLO format (class IDs, coordinate ranges)
- Verifies reconstruction dimensions
- Generates `reports/validation_report.txt`

## Output Structure

```
output/
├── tiles_vis/          # 8-bit visualization PNGs (700 tiles)
├── masks/              # Instance masks (.npy) + metadata (.json) per tile
├── yolo/
│   ├── images/         # Tile images for YOLO
│   ├── labels/         # YOLO segmentation .txt labels
│   └── dataset.yaml    # YOLO dataset configuration
├── overlays/           # Annotation overlay visualizations
├── reconstructed/      # Full-size repatched imagery (10 images)
└── inspection/         # Contact sheets for review
data/
└── metadata/           # Tile metadata CSVs + image mapping
reports/
├── fits_inspection.txt
├── fits_inspection.json
├── validation_report.txt
├── validation_stats.json
└── pipeline_results.json
```

## Class Mapping

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | blob | Stars and compact point-like sources |
| 1 | streak | Space objects (satellites/debris) appearing as elongated features |

Background is unlabeled (no class assigned).

## Blob vs Streak Classification Criteria

Classification uses multiple measured features per connected component:

**Blob (class 0)**: Compact, approximately circular sources
- Aspect ratio < 2.0
- Low eccentricity

**Streak (class 1)**: Elongated features with directional structure
- **Long streaks**: Aspect ratio ≥ 3.0 AND eccentricity ≥ 0.85
- **Short/fat streaks**: Aspect ratio ≥ 2.0 AND eccentricity ≥ 0.7 AND compactness ≤ 0.4 AND major axis ≥ 8.0 pixels

Thresholds are configurable in `config/config.yaml`.

## Faint/Small Blob Handling

Sources with SNR ≥ 3.0 (peak intensity above background by at least 3× the noise) and area ≥ 3 pixels are retained. This preserves genuine faint stars while filtering single-pixel noise.

## Boundary Handling

- 9568 ÷ 1024 = 9 full + 352 remainder → 10 columns
- 6380 ÷ 1024 = 6 full + 236 remainder → 7 rows
- Edge tiles are zero-padded to 1024×1024
- Valid region is tracked; padding is never annotated
- Boundary objects (touching image edge) are preserved
- Reconstruction crops padding back to original dimensions

## Configuration

All parameters are centralized in `config/config.yaml`:
- Image dimensions and tile size
- Detection thresholds (sigma, min area, SNR)
- Classification thresholds (aspect ratio, eccentricity, compactness)
- YOLO polygon simplification tolerance

## Assumptions

1. All 10 FITS files in `Datasets_Assessment/` are annotation targets (including CAM_B files).
2. Class 0 = blob/star, Class 1 = streak/object (as specified in the assessment).
3. Zero-padding is used for edge tiles (not overlap or resize).
4. Detection operates on original uint16 data; 8-bit images are visualization only.
5. Sigma-clipped statistics provide robust background estimation for these sparse images.
6. The assessment's reference figures (Figs. 1–2 in the PDF) serve as visual guidance for annotation style.

## Limitations

- Automated thresholds may not capture every faint source; visual review recommended for borderline cases.
- Polygon simplification introduces minor shape approximation (IoU typically >0.70).
- Objects crossing tile boundaries are annotated separately in each tile.

## Submission Contents

The ZIP contains:
- `src/` — All pipeline source code
- `config/` — Configuration file
- `output/yolo/` — YOLO segmentation labels and images
- `output/reconstructed/` — Full-size repatched imagery
- `output/overlays/` — Sample annotation overlays
- `output/inspection/` — Contact sheets
- `reports/` — Validation and inspection reports
- `data/metadata/` — Tile metadata and image mapping
- `README.md`, `requirements.txt`, `.gitignore`
- `written_response.md` — Four-page written response
