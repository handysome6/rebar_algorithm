# AI Overall Instruction

A working document for the rebar perception paper. Two voices:

- **External advice** — condensed from a separate GPT conversation on novelty
  framing and paper structure.
- **Claude's analysis** — written after reading the code in this repo
  (`src/rebar_algorithm/`) on 2026-04-29.

The two sections sometimes disagree. Where they do, the disagreement is
flagged.

---

## 1. The thesis

One sentence the paper has to defend:

> We cluster rebar knots in **metric plane-space**, not pixel-space, and use
> grid topology to recover missing intersections and reject outliers — turning
> noisy SAM/YOLO proposals into measurable, inspection-ready rebar geometry.

English contribution paragraph (drop-in for an abstract):

> This study proposes a geometry-guided perception and measurement framework
> for rebar mesh inspection. Unlike detection-only methods that identify
> individual rebar cues, the framework exploits the planarity, parallelism,
> and grid-like topology of rebar meshes to refine noisy proposals from
> general-purpose vision models such as SAM and YOLO. Through
> plane-constrained filtering, dominant-direction line fitting in metric
> plane-space, and topology-aware grid correction, the framework reconstructs
> measurable rebar lines, intersections, and spacing for engineering
> inspection. Experiments demonstrate improved measurement accuracy and
> robustness over detection-only baselines under cluttered backgrounds,
> occlusions, and false positives.

---

## 2. Risk: the field is already benchmarked

A 2025 paper introduced a rebar detection and instance segmentation
benchmark (ROI-1555, public on Hugging Face) covering multiple rebar
types, viewpoints, layouts, and assembly stages. It evaluates 6 detectors
and 4 instance segmentation methods.

**Implication.** "We also detect rebars / knots / spacing" has near-zero
novelty. The paper has to be about something detection benchmarks don't
already cover: **metric geometry recovery and grid-level structural
reasoning under occlusion and double-layer interference.**

---

## 3. External advice (GPT, condensed)

### 3.1 Reframe as three layers

| Layer | Purpose | Treat as |
|-------|---------|----------|
| Visual perception | "Where might rebars / knots / boundaries be?" — SAM masks, YOLO knots, candidate regions | **Proposal generation**, not the contribution |
| Geometric constraint | Project proposals onto planar / parallel / grid structure; reject inconsistent ones | Core contribution — RANSAC plane, two-direction line fitting, metric reprojection |
| Structural reasoning | Recover rebar line instances, intersections, ordering, spacing, missing/occluded knots | The "CVPR-flavored" piece — topology recovery |

Writing rule: never present the paper as "we used SAM + YOLO + RANSAC."
Frame it as "we use generic vision models only as proposal generators and
rely on explicit geometric and topological constraints to recover a
metrically valid rebar grid."

### 3.2 Pipeline diagram

```
Input: RGB + stereo depth / point cloud
        │
        ▼
Visual Proposal Generation
  (SAM masks + YOLO knots / intersections)
        │
        ▼
Plane-Constrained Filtering
  (RANSAC + projection)
        │
        ▼
Dominant-Direction Line Fitting
  (two-direction, metric plane-space)
        │
        ▼
Topology-Aware Grid Reasoning
  (intersection recovery + outlier removal + ordering)
        │
        ▼
Metric Inspection Output
  (spacing, deviation, missing bars, pass/fail report)
```

### 3.3 Experiments to plan for

**Baselines** — three is enough, five is overkill:

1. YOLO-only (knots → naive nearest-neighbor spacing).
2. SAM-only + skeletonization + line fit.
3. Hough / RANSAC line detection on raw image.

(Optional fourth: YOLO + SAM without geometry.)

**Metrics** — `mAP` is *not* the headline. The headline is metric error.

| Metric | What it measures |
|--------|------------------|
| Intersection precision / recall | Knot detection quality |
| Line F1 | Rebar line instance recovery |
| Spacing MAE / RMSE | Metric measurement error (mm) |
| Missing-bar detection accuracy | Inspection correctness |
| Pass/fail accuracy | End-to-end engineering decision |
| Runtime | Field deployability |

**Ablations** — each row turns off one module to prove a specific failure
mode is fixed by it:

| Variant | Removes | Demonstrates |
|---------|---------|--------------|
| `w/o plane constraint` | Plane filter | Plane reduces background false positives |
| `w/o dominant direction` | Hough-based angle prior | Direction prior helps line recovery |
| `w/o metric projection` | Plane-space clustering | Metric coordinates beat pixel coordinates under perspective |
| `w/o topology correction` | Grid reasoning | Topology recovers occluded intersections |
| `w/o SAM` | Use YOLO only | Segmentation helps line localization |
| `w/o YOLO` | Use SAM only | Knot detection helps intersection localization |

### 3.4 Story line for the introduction

1. Rebar inspection matters for structural safety.
2. On-site rebar meshes have repetitive texture, occlusion, tying-knot
   distractors, and cluttered backgrounds.
3. Existing detection / segmentation methods recognize *local* visual
   cues but cannot directly produce *engineering-grade structural
   measurements*.
4. Rebar meshes have strong geometric and topological regularity:
   coplanarity, two dominant directions, grid intersections, consistent
   spacing.
5. We propose a structure-aware framework that converts noisy visual
   proposals into a metrically consistent grid representation.
6. Experiments on real construction-site data show stability and
   measurement accuracy gains over detection-only baselines.

### 3.5 Five questions you must answer before writing

| # | Question | Current answer |
|---|----------|----------------|
| 1 | What is the input? | RGB + stereo-reconstructed depth / point cloud |
| 2 | What is the output? | Intersections, rebar lines, rebar spacing |
| 3 | How is ground truth obtained? | Manual on-site collection |
| 4 | What is your strongest technical differentiator? | **Open** — see §4.2 |
| 5 | Where is the gain over baselines? | **Open** — needs ablation study |

---

## 4. Claude's analysis (after reading the code)

### 4.1 What's already implemented

Reading `src/rebar_algorithm/`, the project has more of the geometric
machinery than the GPT advice credits:

- `stages/plane_extraction.py` — RANSAC + SVD-from-seeds + proximity
  envelope. Not just "we ran RANSAC" — there's seeded refinement.
- `stages/line_fitting.py:152-166` — knots are projected into a **metric
  plane-space basis**, clustered there with adaptive distance thresholds,
  then back-projected to image coordinates. This is the genuine
  differentiator over a YOLO-pixel-only pipeline.
- `_analyze_rebar_orientation_from_mask` — Hough on the refined mask
  yields a two-angle prior that drives perpendicular-projection
  clustering. Two-direction structure is already there.
- `_classify_lines_hv` — sorts the two line families into horizontal /
  vertical.

### 4.2 The thesis sentence is therefore narrower than the GPT version

GPT's framing ("we exploit planarity / parallelism / grid topology") is
correct but generic. The defensible, code-backed claim is more specific:

> **Clustering knots in metric plane-space — not pixel-space — is what
> makes spacing measurements transferable across viewpoints, distances,
> and perspective.**

That single ablation row (`w/o metric projection`) is the experiment that
most directly proves the contribution.

### 4.3 What's actually missing

Three concrete gaps, ranked by leverage:

**(a) The topology layer is half-built.** `_classify_lines_hv` sorts
lines but does not model the mesh as an indexed `(row, col)` lattice.
There is no module that:

- Reprojects expected intersections from the line equations.
- Imputes missing knots when YOLO drops one due to occlusion.
- Rejects outlier knots by grid consistency rather than by distance.

This is the single highest-leverage thing to add. It is also what makes
the "topology-aware correction" claim *real* rather than aspirational.
Suggested location: `src/rebar_algorithm/stages/grid_topology.py`,
slotted between `line_fitting` and `spatial_analysis`.

**(b) No evaluation harness.** There is no `metrics/` module computing
spacing MAE/RMSE, intersection P/R, line-F1, pass/fail. Without this
every experiment in §3.3 is blocked. 1–2 day build.

**(c) No baseline runners.** `enable_plane_extraction` is a good start
(`pipeline.py:70`), but there is no way to run "YOLO-only direct
spacing" or "SAM-skeleton + Hough" through the same harness. Need
config switches for: plane filter, metric projection, dominant-angle
prior, topology correction. Without them, no ablation table.

### 4.4 Disagreements with the external advice

- **Five baselines is overkill** for a first submission. Three is
  enough: YOLO-only, SAM-skeleton + RANSAC line, and the full pipeline.
  Add Mask2Former / SAM2 only if a reviewer asks.
- **A short acronym is worth picking.** The external advice says avoid
  network-style names. Fine — but "structure-aware framework" is
  forgettable. A 3–5 letter name (e.g. something like *PG-Mesh* for
  *plane-grid mesh*) helps reviewers remember the paper.
- **Don't push topology recovery as the only novelty.** The metric
  plane-space clustering is the contribution that already works. Sell
  it first. Let topology recovery be the second pillar, not the first.

### 4.5 Evaluation suggestion

Evaluate on ROI-1555 (the 2025 benchmark) **even if** the paper's main
results are on a private construction-site dataset. Otherwise reviewers
will dismiss the comparison as setup-dependent. ROI-1555 is the field's
shared yardstick now.

---

## 5. Recommended next steps (concrete, ordered)

1. **Write `src/rebar_algorithm/metrics/`** — spacing MAE/RMSE,
   intersection P/R, line F1, pass/fail. ≤ 2 days. Unblocks all
   experiments.
2. **Add ablation switches to `pipeline.py`** — flags for metric
   projection, dominant-angle prior, topology correction. ≤ 1 day.
3. **Build `stages/grid_topology.py`** — turns the two line families
   into an indexed lattice; imputes missing intersections; flags
   outliers. ~3–5 days. This is the new contribution.
4. **Run YOLO-only and SAM-skeleton baselines** through the harness.
   ≤ 2 days once (1) and (2) exist.
5. **Evaluate on ROI-1555** for comparability.
6. **Then** start writing.

---

## 6. Phrases to avoid in the paper

Avoid framing the paper as a parts list:

> 我们开发了一个钢筋检测系统。
> 我们使用 YOLO 检测绑扎点。
> 我们使用 SAM 分割钢筋。
> 我们使用 RANSAC 提取平面。
> 我们使用 line fitting 计算间距。

These read as an engineering report. Replace with framing that names
*the problem solved by the constraint*, not the library used.

---

## 7. Open questions

- **Q4 (technical differentiator):** §4.2 proposes metric plane-space
  clustering. Confirm with one ablation before committing in writing.
- **Q5 (where is the gain):** unanswered until the metrics module and
  baselines exist. Resolve via §5 step 4.
- **Dataset specification:** scenes, image counts, camera type, working
  distance, viewpoints, rebar gauges, annotation protocol, GT source.
  Needs a written spec before submission.
- **Naming:** pick a short acronym before the first draft.
