# Rebar Algorithm

Rebar detection and analysis pipeline for stereo projects. The package can:

- get a SAM mask from prompt points,
- refine that mask to the visible rebar surface using the stereo 3D map,
- detect the rebar grid directly from the refined mask, or
- run the older YOLO knot detection path and fit rebar lines from detected knot centers.

The recommended path for current use is:

```text
prompt points or SAM mask -> SAM mask processing -> 3D plane extraction -> mask-grid detection
```

The legacy YOLO path is still available:

```text
prompt points or SAM mask -> SAM mask processing -> 3D plane extraction -> YOLO knot detection -> line fitting -> optional 3D spatial analysis
```

## Setup With uv

Install `uv` using the official Astral instructions at <https://docs.astral.sh/uv/getting-started/installation/>.

Then set up this project:

```bash
cd /path/to/rebar_algorithm
uv sync
```

This project requires Python 3.10 or newer. If your machine does not already have a compatible Python, install one with uv before syncing:

```bash
uv python install 3.12
uv sync --python 3.12
```

For development and tests:

```bash
uv sync --extra dev
uv run pytest
```

Useful commands:

```bash
uv run rebar-demo --help
uv run rebar-gui
```

## Project Data Requirements

A stereo project is a directory containing image and 3D files for one capture. The examples below use:

```text
/path/to/project/
```

Minimum files depend on the mode you run.

| File | Required for | Format and notes |
| --- | --- | --- |
| `rect_left.jpg` | Required for all pipeline modes | Rectified left image. The code fails strictly if this file is missing. |
| `xyz_map.npz` | Plane extraction; optional 3D attachment in mask-grid | NumPy archive with key `xyz_map`, shape `(H, W, 3)`, values in metres. Required by `StereoProject`. |
| `K.txt` | Metric YOLO line fitting and 3D projection | First line should contain 9 floats for a row-major 3x3 camera intrinsic matrix. An optional second line can contain baseline. |
| `sam_mask.npy` | `--sam-mask` input mode | Precomputed SAM mask. Accepted shapes include 2D mask, `(H, W, 1)`, and RGBA `(H, W, 4)`. The pipeline converts it to binary. |
| `refined_mask.npy` | `--refined-mask` input mode | Plane-refined binary mask. This skips SAM mask processing and plane extraction. |
| `<output>/plane_extraction_results/refined_mask.npy` | `--reuse-refined` input mode | Cached refined mask from a previous run. |
| `<output>/yolo_results/pose_data.json` | YOLO path with `--use-existing` | Existing YOLO detections. If not present, the YOLO server is called. |

External services:

| Service | Required when | Config |
| --- | --- | --- |
| SAM server | Using `--points` or `run_pipeline_from_points()` | `configuration/sam_conf.yaml`; endpoint is `<server_url>/api/segment`. |
| YOLO server | Using `--detector yolo` without reusable results | `configuration/rebar_conf.yaml`; endpoint is `<server_url>/process_pose/`. |

Python dependencies are declared in `pyproject.toml` and locked by `uv.lock`.

## Quick Start

Run from prompt points and detect the grid directly from the refined mask:

```bash
uv run rebar-demo \
  --project /path/to/project \
  --points 836,902 1778,705 \
  --detector mask-grid
```

Run from an existing SAM mask:

```bash
uv run rebar-demo \
  --project /path/to/project \
  --sam-mask /path/to/sam_mask.npy \
  --detector mask-grid
```

Run from an already-refined mask and skip SAM and plane extraction:

```bash
uv run rebar-demo \
  --project /path/to/project \
  --refined-mask /path/to/refined_mask.npy \
  --detector mask-grid
```

Reuse the refined mask from a previous output folder:

```bash
uv run rebar-demo \
  --project /path/to/project \
  --output /path/to/project/rebar_output \
  --reuse-refined \
  --detector mask-grid
```

Use the YOLO path:

```bash
uv run rebar-demo \
  --project /path/to/project \
  --sam-mask /path/to/sam_mask.npy \
  --detector yolo \
  --yolo-url http://localhost:2001
```

Print the resolved pipeline without running it:

```bash
uv run rebar-demo \
  --project /path/to/project \
  --sam-mask /path/to/sam_mask.npy \
  --detector mask-grid \
  --explain
```

By default, output goes to:

```text
<project>/rebar_output/
```

Override it with `--output`.

## Python Usage

Run from a precomputed SAM mask:

```python
from pathlib import Path

import numpy as np

from rebar_algorithm import run_pipeline_auto

project_path = Path("/path/to/project")
output_path = project_path / "rebar_output"
sam_mask = np.load("/path/to/sam_mask.npy")

final_image_path, analysis_json_path = run_pipeline_auto(
    project_path=project_path,
    output_path=output_path,
    sam_mask=sam_mask,
    use_mask_grid_detector=True,
)
```

Run from prompt points through the application helper:

```python
from pathlib import Path

from rebar_algorithm.app_api import run_pipeline_from_points

result = run_pipeline_from_points(
    project_path=Path("/path/to/project"),
    points=[(836, 902), (1778, 705)],
    detector="mask-grid",
)

print(result.final_image_path)
print(result.analysis_json_path)
```

## CLI Inputs

`rebar-demo` requires exactly one input source:

| Option | Meaning |
| --- | --- |
| `--points X,Y ...` | Calls the configured SAM server with positive foreground points. Saves `sam_mask.npy` and `sam_prompt_points.npy`. |
| `--sam-mask path.npy` | Loads a precomputed SAM mask, then optionally refines it with plane extraction. |
| `--refined-mask path.npy` | Loads a plane-refined mask and skips SAM mask processing and plane extraction. |
| `--reuse-refined` | Loads `<output>/plane_extraction_results/refined_mask.npy`. |

Detector selection:

| Option | Meaning |
| --- | --- |
| `--detector mask-grid` | Uses mask geometry to detect line families and intersections. No YOLO server is needed. |
| `--detector yolo` | Sends the segmented image to YOLO, parses knot centers, and fits lines. |

Other useful options:

| Option | Meaning |
| --- | --- |
| `--no-plane` | Disable plane extraction for SAM input. |
| `--plane-threshold 0.03` | Keep 3D points within this distance, in metres, of the fitted plane. |
| `--use-existing` | In YOLO mode, reuse existing `pose_data.json` if available. |
| `--config path.yaml` | Use a custom rebar config file. |
| `--verbose` | Enable debug logging. |

## Pipeline Orchestration

The top-level orchestrator is `src/rebar_algorithm/pipeline.py`.

| Function | Usage | Input | Output |
| --- | --- | --- | --- |
| `run_pipeline_auto()` | Recommended Python API. Loads `configuration/rebar_conf.yaml`, applies explicit overrides, then calls `run_pipeline()`. | `project_path`, `output_path`, `sam_mask`, optional config/flags. | `(final_image_path, analysis_json_path)`. The JSON path can be `None` if YOLO detection fails. |
| `run_pipeline()` | Lower-level API with explicit parameters. Used by `run_pipeline_auto()`. | Same core data, but all runtime options are passed directly. | `(final_image_path, analysis_json_path)`. |
| `_select_visualization_image()` | Internal helper to choose an overlay base image. | Project path. | Path to `rect_left.jpg`; raises `FileNotFoundError` if missing. |
| `_write_refined_input_image()` | Internal helper for `--refined-mask` mode. | Output path, refined mask, base image. | Writes `refined_input_results/segmented_rebar_refined.png`. |

## Pipeline Stages

### Optional Input Stage: SAM From Prompt Points

Module: `src/rebar_algorithm/app_api.py`

| Function | Usage | Input | Output |
| --- | --- | --- | --- |
| `find_project_image()` | Finds the required image for point picking and SAM upload. | `project_path`. | Path to `rect_left.jpg`; raises `FileNotFoundError` if missing. |
| `get_sam_mask()` | Calls the SAM server and caches the returned mask. | `project_path`, prompt points as `(x, y)`, `output_path`. | `(sam_mask, points_xy)` and files `sam_segment_results/sam_mask.npy`, `sam_segment_results/sam_prompt_points.npy`. |
| `run_pipeline_from_points()` | App/GUI helper for point-driven runs. | Project path, points, detector and config options. | `PipelineRunResult` with final image, analysis JSON, output path, SAM mask path, and prompt point path. |

SAM client details are in `src/rebar_algorithm/clients/sam_client.py`.

| Function | Usage | Input | Output |
| --- | --- | --- | --- |
| `SamClient.segment()` | Sends an RGB image and foreground points to the SAM HTTP API. | Image array `(H, W, 3)` RGB uint8; points as `[x, y, label]`. | List of binary masks `(H, W)` uint8. |

### Step 1: SAM Mask Processing

Module: `src/rebar_algorithm/stages/sam_mask.py`

| Function | Usage | Input | Output |
| --- | --- | --- | --- |
| `SamMaskProcessor.load_base_image()` | Loads the required image used for segmentation output and overlays. | `project_path`, `project_id`. | OpenCV BGR image from `rect_left.jpg`; raises if missing or unreadable. |
| `SamMaskProcessor.process_mask()` | Converts SAM output to a binary mask and segmented image. | `sam_mask`, base image, `output_path`, `project_id`. | Dict with `mask_binary`, `segmented_image`, `segmented_image_path`, `mask_path`, `coverage`, `mask_pixels`. |
| `_convert_to_binary_mask()` | Normalizes SAM mask formats. | 2D mask, `(H, W, 1)`, or RGBA `(H, W, 4)`. | Binary uint8 mask. |
| `_resize_mask_if_needed()` | Aligns mask size with the base image. | Binary mask and target `(height, width)`. | Resized or original mask. |
| `_create_segmented_image()` | Greys out non-rebar pixels. | Base image, binary mask. | BGR segmented image. |

Files written:

```text
<output>/sam_segment_results/segmented_rebar.png
<output>/sam_segment_results/rebar_mask.npy
```

### Step 2: 3D Plane Extraction

Module: `src/rebar_algorithm/stages/plane_extraction.py`

| Function | Usage | Input | Output |
| --- | --- | --- | --- |
| `PlaneExtractor.extract_surface_layer()` | Orchestrator-facing wrapper. Saves refined mask and metadata. | Binary SAM mask, `project_path`, `output_path`, optional prompt points. | Dict with refined masks, metadata, output paths, coverage values. |
| `extract_surface_layer_from_sam_mask()` | Convenience function around `PlaneExtractorImpl`. | SAM mask, project path, threshold, optional prompt points. | `(refined_mask, metadata)`. |
| `PlaneExtractorImpl.extract_surface_layer()` | Core implementation. Loads 3D data, fits plane, filters points near plane. | SAM mask and stereo project files. | Refined binary mask and metadata dict. |
| `_fit_plane_svd()` | Fits plane from at least 3 valid prompt seed points. | Seed 3D points. | Plane model `{a, b, c, d}` and normal. |
| `_fit_plane_ransac()` | Fallback robust plane fitting. | Masked 3D points, iteration count, inlier threshold. | Plane model, full inlier mask, normal. |
| `_compute_proximity_envelope()` | Builds a signed-distance envelope around prompt seeds. | Seed 3D points, plane normal, plane distance. | `(near, far)` distances in metres. |
| `_extract_points_near_plane()` | Produces the refined surface-layer mask. | Full point cloud, plane model, normal, image shape, threshold/envelope. | Binary mask at point-cloud resolution. |
| `PlaneExtractor.update_segmented_image()` | Writes refined segmented image. | Output path, refined mask, base image. | Path to refined segmented image. |

Files required:

```text
<project>/xyz_map.npz
<project>/rect_left.jpg
```

Files written:

```text
<output>/plane_extraction_results/refined_mask.npy
<output>/plane_extraction_results/plane_metadata.json
<output>/plane_extraction_results/segmented_rebar_refined.png
```

### Step 3A: Mask-Grid Detection

Module: `src/rebar_algorithm/stages/mask_grid.py`

This is the recommended detector when the refined mask is clean enough. It does not require YOLO.

| Function | Usage | Input | Output |
| --- | --- | --- | --- |
| `MaskGridDetector.detect_grid()` | Orchestrator-facing method. Runs analysis and writes overlays/JSON. | Refined mask, output path, optional image paths and project path. | Dict with `analysis`, `analysis_json_path`, `final_image_path`, `refined_overlay_path`. |
| `MaskGridDetector.analyze_mask()` | Detects grid lines and intersections without writing files. | Refined mask; optional `project_path` for 3D point lookup. | Analysis dict. |
| `_detect_grid_angles()` | Estimates the two dominant rebar directions. | Binary mask. | Two angles in degrees. |
| `_extract_family()` | Extracts one directional line family using rotation and morphology. | Mask, angle, scaled parameters. | Line dicts and directional debug mask. |
| `_fit_line_from_rotated_band()` | Fits a centerline through one band of mask pixels. | Directional mask band. | Line detail dict or `None`. |
| `_compute_intersections()` | Intersects horizontal and vertical fitted lines and validates local mask support. | Mask, line families, tolerance parameters. | List of intersection dicts. |
| `_attach_3d_points()` | Adds 3D coordinates to intersections when possible. | Intersections and `project_path`. | Mutates intersections with `point_3d_m` values. |
| `draw_overlay()` | Draws detected lines and intersections. | BGR image and analysis dict. | Annotated BGR image. |

Optional file used:

```text
<project>/xyz_map.npz
```

Files written:

```text
<output>/mask_grid_results/mask_grid_analysis.json
<output>/mask_grid_results/mask_grid_on_rect_left.png
<output>/mask_grid_results/mask_grid_on_segmented_rebar_refined.png
<output>/mask_grid_results/directional_horizontal_mask.png
<output>/mask_grid_results/directional_vertical_mask.png
```

The JSON contains line counts, line coordinates, intersections, pixel spacing summaries, and optional per-intersection 3D points.

### Step 3B: YOLO Knot Detection

Module: `src/rebar_algorithm/stages/knot_detection.py`

This is the legacy detector path. It requires a YOLO server unless `--use-existing` finds cached results.

| Function | Usage | Input | Output |
| --- | --- | --- | --- |
| `KnotDetector.detect_knots()` | Orchestrator-facing method. Reuses or creates YOLO result files. | Segmented image path, output path, `use_existing`. | Dict with `pose_data_path`, `found_existing`, `knot_count`, and optional raw result map. |
| `_check_existing()` | Finds cached YOLO results. | Expected pose path and output path. | `True` if reusable results are found or copied from legacy location. |
| `_run_detection()` | Calls the YOLO HTTP client. | Image path and output directory. | Dict of extracted ZIP files. |
| `validate_detection_results()` | Confirms `pose_data.json` exists and contains detections. | Pose JSON path. | Boolean. |

YOLO client details are in `src/rebar_algorithm/clients/yolo_client.py`.

| Function | Usage | Input | Output |
| --- | --- | --- | --- |
| `YoloClient.process_image()` | Uploads a segmented image to the YOLO server. | Image path and output directory. | Extracted ZIP file map. |

Expected main file:

```text
<output>/yolo_results/pose_data.json
```

### Step 4: Line Fitting

Module: `src/rebar_algorithm/stages/line_fitting.py`

This stage runs only in the YOLO detector path.

| Function | Usage | Input | Output |
| --- | --- | --- | --- |
| `LineFitter.fit_lines()` | Orchestrator-facing wrapper. | `pose_data.json`, visualization image, output path, optional refined mask and plane metadata. | Dict with analyzer, final image path, analysis JSON path, line counts, and `used_hough`. |
| `LineFittingAnalyzer.process_all()` | End-to-end line fitting workflow. | JSON path, image path, output folder, optional mask/plane/project data. | Boolean success; writes JSON and image unless disabled. |
| `set_input_files()` | Validates input JSON/image and reads image dimensions. | JSON path and image path. | Boolean success. |
| `extract_centers()` | Parses YOLO detections into knot center points. | `pose_data.json`. | Populates `centers`. |
| `fit_lines_from_centers()` | Clusters knot centers into horizontal and vertical rebar lines. | Centers, image shape, optional refined mask. | Dict of fitted line segments. |
| `set_plane_data()` | Loads depth and camera intrinsics for metric plane-space clustering. | Plane metadata, project path or depth path. | Boolean success. |
| `analyze_line_fitting_results()` | Builds counts, line details, and spacing summaries. | Fitted line state. | Analysis dict. |
| `save_analysis_results()` | Writes line fitting JSON. | Current analysis state. | `line_fitting_analysis.json`. |
| `save_visualization()` | Draws and writes line overlays. | Current analyzer state. | `line_fitting_visualization.png`. |
| `create_segmented_overlay()` | Draws fitted lines over the segmented rebar image. | Segmented image path and output path. | `visualization_results/lines_on_segmented_rebar.png` or `None`. |

Parser details are in `src/rebar_algorithm/clients/yolo_parser.py`.

| Function | Usage | Input | Output |
| --- | --- | --- | --- |
| `YoloResultParser.get_centers_coordinates()` | Returns weighted center points from visible YOLO keypoints. | Parsed `pose_data.json`. | List of `(x, y)` points in image pixels. |

Files written:

```text
<output>/line_fitting_results/line_fitting_analysis.json
<output>/line_fitting_results/line_fitting_visualization.png
<output>/visualization_results/lines_on_segmented_rebar.png
```

### Step 5: Optional 3D Spatial Analysis

Module: `src/rebar_algorithm/stages/spatial_analysis.py`

This stage runs only if an external `ai_matcher` object is passed to the Python API. The current repository contains the wrapper, but the line-fitting analyzer's `calculate_3d_spatial_metrics()` is a stub in this extracted package.

| Function | Usage | Input | Output |
| --- | --- | --- | --- |
| `SpatialAnalyzer.calculate_3d_metrics()` | Calls analyzer-level metric calculation when `ai_matcher` exists. | Line fitting analyzer, optional pose data path. | Metrics dict or `None`. |
| `SpatialAnalyzer.get_spacing_uniformity()` | Converts spacing mean/std into uniformity scores. | Metrics dict. | Dict with horizontal, vertical, and overall uniformity. |

### Step 6: Visualization Helpers

Module: `src/rebar_algorithm/stages/visualization.py`

The main pipeline currently creates the segmented overlay through `LineFitter.create_segmented_overlay()` in YOLO mode. The `Visualizer` class and helper functions are available for additional views.

| Function | Usage | Input | Output |
| --- | --- | --- | --- |
| `Visualizer.create_main_visualization()` | Draws analyzer lines over an image. | Analyzer, image path, output path. | `visualization_results/line_fitting_visualization.png`. |
| `Visualizer.create_segmented_overlay()` | Draws analyzer lines over a segmented image. | Analyzer, segmented image path, output path. | `visualization_results/lines_on_segmented_rebar.png`. |
| `Visualizer.create_comparison_view()` | Creates a side-by-side before/after image. | Original image path, result image path, output path. | `visualization_results/comparison_before_after.png`. |
| `draw_spacing_overlay()` | Draws H/V lines and 3D distances between adjacent grid nodes. | Base image, analysis dict, `xyz_map`. | Annotated BGR image. |
| `draw_knot_boxes_overlay()` | Draws knot boxes from YOLO detections or grid intersections. | Base image, analysis dict, YOLO JSON path. | Annotated BGR image. |

## Output Layout

A typical mask-grid run writes:

```text
<output>/
  sam_segment_results/
    sam_mask.npy                  # when using --points
    sam_prompt_points.npy         # when using --points
    rebar_mask.npy
    segmented_rebar.png
  plane_extraction_results/
    refined_mask.npy
    plane_metadata.json
    segmented_rebar_refined.png
  mask_grid_results/
    mask_grid_analysis.json
    mask_grid_on_rect_left.png
    mask_grid_on_segmented_rebar_refined.png
    directional_horizontal_mask.png
    directional_vertical_mask.png
```

A YOLO run additionally writes:

```text
<output>/
  yolo_results/
    pose_data.json
    ...
  line_fitting_results/
    line_fitting_analysis.json
    line_fitting_visualization.png
  visualization_results/
    lines_on_segmented_rebar.png
```

## Configuration

Default config files:

```text
configuration/sam_conf.yaml
configuration/rebar_conf.yaml
```

`sam_conf.yaml` controls the SAM segmentation server used by prompt-point runs.

`rebar_conf.yaml` controls:

- YOLO server URL and timeout,
- whether to reuse existing YOLO annotations,
- whether config default detector is mask-grid or YOLO,
- whether visualization prefers rectified images,
- plane extraction thresholds.

CLI flags override config values for that run.

## Tests

```bash
uv sync --extra dev
uv run pytest
```

The unit tests do not require SAM or YOLO servers. Tests that need a local sample project are skipped when the sample path is missing.
