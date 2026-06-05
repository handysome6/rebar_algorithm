# Scripts

## Overview

| Script | Purpose |
|--------|---------|
| `run_demo.py` | Quick launcher — delegates to `rebar-demo` CLI |
| `verify_against_original.py` | Verifies extracted algorithm against original JetsonReborn GUI output |
| `experiment_yolo_guided_sam.py` | Experiment: auto-generate SAM prompt points from YOLO detections |

## Verification: Extracted Algorithm vs Original GUI

**Script:** `verify_against_original.py`

Validates that the extracted `rebar_algorithm` package reproduces the same results as the original JetsonReborn GUI pipeline on stereo project `222`.

**Method:**
1. Load `rect_left.jpg` from `/Users/andyliu/DCIM/222`
2. Call SAM server with 4 manual positive points: `(1392, 1240)`, `(2138, 1358)`, `(1911, 1981)`, `(2879, 2273)`
3. Run the full pipeline (SAM mask → plane extraction → YOLO → line fitting)
4. Output to `rebar_output_verify/` for comparison with original `rebar_output_sam/`

**Result:** The extracted algorithm faithfully reproduces the original GUI pipeline.

| Metric | Extracted Algorithm | Original GUI |
|--------|-------------------|-------------|
| Horizontal lines | 8 | 8 |
| Vertical lines | 10 | 10 |
| YOLO knot centers | 79 | 75 |
| Plane normal | [0.024, 0.072, 0.997] | [0.023, 0.091, 0.996] |
| Plane distance (d) | -0.8849 | -0.8860 |
| Refined coverage | 20.0% | 20.6% |

The 4 extra YOLO detections (79 vs 75) are edge knots exposed by a slightly different SAM mask boundary. Plane models converge to nearly identical values despite different SAM masks (30.4% vs 22.7% raw coverage).

---

## Experiment: YOLO-Guided SAM Segmentation

**Script:** `experiment_yolo_guided_sam.py`

**Goal:** Eliminate manual SAM prompt points entirely. Instead, detect rebar knots on the raw image first, then use the top knot positions as SAM prompt points.

### Pipeline

```
rect_left.jpg
     │
     ▼
[1] YOLO on raw image ──→ 51 knot detections
     │
[2] Select top-10 points (spatially spread)
     │
     ▼
[3] SAM server ──→ segmentation mask
     │
     ▼
[4] Standard pipeline (plane extraction → YOLO on segmented → line fitting)
```

### Point Selection Strategy

Naive top-k by confidence failed because:
- High-confidence detections clustered in the center-right of the grid
- Edge/corner false positives (e.g., point at (107, 197)) skewed the plane fit

The improved strategy uses **farthest-point sampling with confidence weighting:**

1. **Edge filter:** Discard points within 5% of the image border
2. **Seed:** Start with the highest-confidence point
3. **Greedy expansion:** Each subsequent pick maximises `min_distance_to_selected * confidence`, balancing spatial spread with detection quality

### Results

| Metric | YOLO-guided (naive top-k) | YOLO-guided (spatial spread) | Manual points | Original GUI |
|--------|--------------------------|------------------------------|---------------|-------------|
| SAM coverage | 28.5% | 40.0% | 30.4% | 22.7% |
| Refined coverage | 14.0% | 17.6% | 20.0% | 20.6% |
| YOLO knots (pipeline) | 42 | 77 | 79 | 75 |
| Horizontal lines | 7 | **8** | **8** | **8** |
| Vertical lines | 7 | **10** | **10** | **10** |
| Plane normal | - | [0.020, 0.087, 0.996] | [0.024, 0.072, 0.997] | [0.023, 0.091, 0.996] |
| Plane distance (d) | - | -0.8793 | -0.8849 | -0.8860 |

### Key Findings

1. **Spatial spread is critical.** Naive top-k by confidence produced a biased SAM mask that missed the left side of the grid entirely (7H x 7V). Farthest-point sampling recovered the full 8H x 10V grid.

2. **Refined coverage is lower (17.6% vs ~20%).** The 10 YOLO-derived seed points produce a tighter SVD proximity envelope than 4 manually chosen points. This doesn't hurt detection — 77 knots is close to the 79/75 baseline — but the mask is more aggressively trimmed.

3. **The approach is fully automatic.** No manual point selection needed. YOLO on the raw image provides enough spatial signal to guide SAM segmentation, and the downstream pipeline matches the manual baseline.

### Running

```bash
# Full run (calls YOLO + SAM servers)
uv run python scripts/experiment_yolo_guided_sam.py

# Output goes to /Users/andyliu/DCIM/222/rebar_output_yolo_guided/
```

Test project: stereo project `222` at `/Users/andyliu/DCIM/222`
