# Figure 1 — Optogenetic LC time-locked fMRI analysis

This folder contains the **exact analysis pipeline used to generate Figure 1** of the paper, starting from ROI-based optogenetic fMRI time-locked datasets and ending with SVD-derived spatial maps.

---

## Overview of the pipeline

The Figure 1 workflow consists of **three sequential steps**:

1. **Generation of time-locked ROI datasets** (performed upstream, not included here)
2. **Construction of a core LC temporal template**
3. **SVD-based extraction of spatial maps and figures**

Only steps **2** and **3** are implemented in this folder.

---

## Input data (expected format)

All scripts operate on **ROI × time CSV files**, where:

- Rows = ROI acronyms (e.g. `VPM`, `BLA`, `RSPd`, …)
- Columns = time (in seconds, numeric, starting at 0)
- Values = mean z-scored fMRI response time-locked to optogenetic LC stimulation

Each CSV corresponds to **one stimulation frequency**.

Required inputs:
- `Opto_fMRI_dataset_3Hz_*.csv`
- `Opto_fMRI_dataset_5Hz_*.csv`
- `Opto_fMRI_dataset_15Hz_*.csv`

These files are produced by the preprocessing pipeline described in the Methods section of the paper.

---

## Step 2 — Core LC template construction

Script:
```
02_build_core_LC_template.py
```

This script identifies a set of **core ROIs** that show a *consistent temporal response shape* across stimulation frequencies and builds a canonical LC-locked temporal template.

### What the script does

1. Loads the 3Hz / 5Hz / 15Hz ROI × time CSVs
2. Keeps the fixed time window **0–30 s**
3. Removes ROIs containing NaNs in **any** condition
4. Computes ROI-wise similarity across conditions using **raw (non-amplitude-normalized) correlations**
5. Selects *core ROIs* using a similarity threshold (default: `0.75`)
6. Builds the **core LC template** as the mean response across conditions
7. Saves:
   - `core_LC_template.csv` (ROI × time)
   - `core_rois.txt`
   - `roi_similarity_scores.csv`
   - Optional NIfTI representations (if atlas inputs are provided)
---

## Step 3 — SVD-based spatial map extraction

Script:
```
03_svd_maps.py
```

This script performs the dimensionality reduction and spatial mapping shown in Figure 1.

### Core logic

1. Load `core_LC_template.csv`
2. Keep the fixed **0–30 s** window
3. **Z-score each ROI across time**
4. Perform SVD on the matrix `X = time × ROI`
5. Select the first `K` modes based on cumulative explained variance
6. Define **dominant temporal phases** using:
   ```
   |U[:, i] × Σ[i]|
   ```
   (absolute contribution; sign is ignored for dominance)
7. Estimate one spatial map per mode using regression within its dominant phase
8. Resolve the **global sign of each map** based on the dominant ROIs (paper-consistent rule)
9. Save final figures and optional NIfTI maps

### Outputs

- `spatial_maps.csv`  
- `roi_by_map_values.csv`  
- Figures (PNG), exactly matching those in the paper:
  - ROI contribution histograms
  - Temporal projection heatmaps
- Optional NIfTI maps (unilateral and bilateral)

The script **does not** save intermediate diagnostics or metadata that are not shown in the paper.

---

## Reproducibility guarantees

- Fixed time window (0–30 s)
- No stochastic steps
- No user-tunable options affecting the figures
- Identical inputs → identical outputs

This folder therefore represents a **reproducible snapshot** of the Figure 1 analysis.

---

## Dependencies

Required:
- Python ≥ 3.9
- numpy
- pandas
- matplotlib

Optional (for NIfTI export):
- nibabel
- ABI mouse brain parcellation NIfTI
- ABI ROI table (Excel)

---

## Citation

If you reuse this code, please cite the associated paper.

