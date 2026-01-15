# Fig3 — Sleep transition similarity (combined NPZ pipeline)

This folder contains a small, self-contained pipeline to reproduce the **Fig.3 sleep-transition analysis** from a single “combined” NPZ file (`similarity_trace_combined.npz`).

The idea is:
1) you build (or export) a *combined* NPZ that contains, for each subject/session, a similarity time series and the detected transition indices.
2) you run `python -m scripts.main ...` to generate:
   - group-level transition traces
   - per-subject heatmaps
   - event-wise AUC stats (Real vs Null windows)

---

## What is inside `similarity_trace_combined.npz`

The pipeline expects a NumPy `.npz` with these keys:

### Required keys

- **`sims`**: `float` array of shape **(n_subjects, max_T, n_maps)**
  - Similarity time series (e.g., LC-template similarity) per subject.
  - Time is the second axis.
  - Shorter subjects are **padded with NaNs** up to `max_T`.

- **`state`**: `float` array of shape **(n_subjects, max_T)**
  - State vector aligned to `sims`.
  - Mouse-like coding used by this repo:
    - **1 = WAKE**
    - **3 = NREM**
    - other/unknown values are typically stored as NaN and ignored.

- **`tr`**: `float`
  - TR in **seconds**.
  - The analysis uses this TR for the time axis in plots and in AUC integration.

- **`subjects`**: `object` array of shape **(n_subjects,)**
  - Subject/session identifiers (strings).

- **`transitions_1to3`**: `object` array of length **n_subjects**
  - For each subject, a 1D `int` array with **indices `t`** such that
    `state[t] == 1` and `state[t+1] == 3`.
  - Interpreted as **WAKE → NREM** transitions.

- **`transitions_3to1`**: `object` array of length **n_subjects**
  - For each subject, a 1D `int` array with **indices `t`** such that
    `state[t] == 3` and `state[t+1] == 1`.
  - Interpreted as **NREM → WAKE** transitions.

### Derived / used internally

From the combined NPZ, the code builds per-transition epochs of length:

- **window length in seconds**: `WIN_SECONDS` (configured in `scripts/config/settings.py`)
- **window length in TRs**: `WIN_TRS = round(WIN_SECONDS / tr)`
- each epoch is **±WIN_TRS** around the transition index `t`

Epochs are extracted only when the full window fits within the available (non-NaN) portion of each subject.

---

## What the code does

### 1) Epoch extraction from the combined NPZ

`scripts/io/from_combined.py` provides `build_epochs_from_combined_npz(...)`.

It:
- loads `sims`, `subjects`, `tr` and transition indices
- builds two collections of epochs:
  - **(1,3)** : WAKE → NREM
  - **(3,1)** : NREM → WAKE
- output format used everywhere else:

```python
all_subject_epochs[(a,b)] = [ subj_epochs_0, subj_epochs_1, ... ]
# each subj_epochs_i has shape (n_events_i, T, n_maps)
# where T = 2*WIN_TRS + 1
```

### 2) Group-level plots

`scripts/plots/group_plots.py` (`plot_all_group_level`) creates “paper-style” plots:
- mean trace across all epochs
- SEM shading
- optional per-event z-scoring (recommended; enabled in `main.py`)

### 3) Group subject heatmaps

`scripts/plots/subject_heatmaps.py` (`plot_group_subject_heatmaps`) creates heatmaps:
- one row per subject
- value = subject-average epoch (average across that subject’s events)
- separate heatmaps per transition and per map

### 4) Event-wise AUC stats (Real vs Null)

`scripts/stats/auc_eventwise.py` (`auc_eventwise`) computes, for each transition and each map:

- **Real distribution**: AUC of |z| for each real transition epoch
- **Null distribution**: AUC of |z| computed on random windows sampled from `sims`

For each map, the code reports:
- KS test p-value
- Mann–Whitney U p-value
- Cliff’s delta effect size

Outputs:
- `auc_eventwise/AUC_EVENTWISE_<a>to<b>.svg`
- `auc_eventwise/auc_eventwise_stats.csv`

**Important:** TR is taken from the combined NPZ key `tr` (or can be overridden by passing `tr_s`).

---

## How to run

### Run everything (AUC + plots + heatmaps)

```bash
python -m scripts.main \
  --combined "/path/to/similarity_trace_combined.npz" \
  --output-dir "/path/to/output_dir"
```

### Skip plots (run only AUC)

```bash
python -m scripts.main \
  --combined "/path/to/similarity_trace_combined.npz" \
  --output-dir "/path/to/output_dir" \
  --skip-plots
```

You can customize AUC integration limits:

```bash
python -m scripts.main \
  --combined "/path/to/similarity_trace_combined.npz" \
  --output-dir "/path/to/output_dir" \
  --t0 -10 --t1 30 --n-rand 5000
```

---

## Outputs

All outputs go under `--output-dir` (or default `RESULTS_DIR`):

- Group plots:
  - `MOUSE_1to3_combinedmaps.svg`
  - `MOUSE_3to1_combinedmaps.svg`

- Heatmaps:
  - `subject_heatmaps_group/1to3_MAP*_group_heatmap.svg`
  - `subject_heatmaps_group/3to1_MAP*_group_heatmap.svg`

- AUC eventwise:
  - `auc_eventwise/AUC_EVENTWISE_1to3.svg`
  - `auc_eventwise/AUC_EVENTWISE_3to1.svg`
  - `auc_eventwise/auc_eventwise_stats.csv`

---

## Notes on mouse vs human

This pipeline is intentionally **species-agnostic** as long as the combined NPZ follows the schema above.

For humans, you can export a mouse-compatible combined NPZ by mapping the hypnogram states to:
- WAKE → `1`
- NREM → `3`
(and storing the detected transition indices in `transitions_1to3` / `transitions_3to1`).

---

## Cross-species projection (mouse → human NIfTI)

In addition to the sleep-transition analyses, this folder includes a utility script to project **mouse ROI-wise LC map values** onto **human MNI-space atlases** using a predefined mouse–human ROI mapping table.

This step was used to generate the human-space LC map visualizations associated with Fig.3–4.

### Purpose

The script:
1. Takes ROI-wise mouse map values (e.g., `Map_1`, `Map_2` from SVD templates).
2. Uses a mouse–human ROI correspondence table.
3. Fills standard human atlases (MNI 2 mm) with the corresponding values.
4. Outputs:
   - A 4D NIfTI file (one volume per map).
   - A ROI-wise CSV summary.

### Script location

```
scripts/cross_species/MOUSE_to_HUMAN_NIFTI.py
```

### Required inputs

- **Mouse ROI map CSV**  
  `roi_by_map_values.csv`  
  (rows = mouse ROI acronyms, columns = Map_1, Map_2, ...)

- **Mouse–human mapping table** (CSV or XLSX)  
  Must contain columns:
  - `Atlas`
  - `Mapped_ROI` (human ROI name)
  - `Animal_ROI` (mouse ROI acronyms)
  - `Index` (atlas label indices)

- **Human atlases in MNI space (2 mm)**  
  Standard FSL atlases can be used, typically found in:

```
$FSLDIR/data/atlases
```

Supported atlases:
- HarvardOxford Cortical
- HarvardOxford Subcortical
- Morel thalamic atlas

### Example command

```bash
python scripts/cross_species/MOUSE_to_HUMAN_NIFTI.py \
  --roi-csv "/path/to/roi_by_map_values.csv" \
  --mapping-csv "/path/to/Selected_ROI_HUMAN_MOUSE.xlsx" \
  --atlas-dir "$FSLDIR/data/atlases" \
  --out-nifti "/path/to/Mouse_to_Human_MAPS.nii.gz" \
  --out-csv "/path/to/Mouse_to_Human_MAPS_ROIwise.csv"
```

### Outputs

- **4D NIfTI**: human MNI-space maps (one volume per LC map)  
- **CSV summary**: ROI-wise numerical values used to fill the atlas

### Notes

- This step is independent from the NPZ-based sleep-transition pipeline.
- It is provided for cross-species visualization and interpretation of LC-related spatial maps.
- Atlas files are not included in the repository and must be obtained separately (e.g., via FSL).
