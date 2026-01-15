# Figure 2 — Photometry (urethane) ↔ fMRI

Code to reproduce **Figure 2** panels for the project `fig2_photometry-urethane-fmri`.

Panels:
- **Fig2a** — UMAP embedding of LC-burst–triggered brain dynamics (ROI time series)
- **Fig2b** — Ridgeline plot of burst-aligned ROI waveforms (selected ROIs)
- **Fig2c** — Group burst-aligned SVD-map similarity (Map1/Map2) + Global Signal (GS) statistics
- **QC Fig2c** —  Control analysis: LC–Map coupling vs Global Signal (no figures).

> Tip: write all generated figures into `outputs/` and do not track them with git.

---

## Folder structure

```
fig2_photometry-urethane-fmri/
├─ scripts/
│  ├─ fig2a_umap.py
│  ├─ fig2b_ridgeline_waves.py
│  ├─ fig2c_similarity_gs.py
│  └─ qc_similarity_gs_lc_controls.py
├─ config/
├─ outputs/
├─ requirements.txt
└─ README.md
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Inputs and naming conventionsa

### Burst index files
`<base>_burst_indices.txt` (volume indices)

### ROI time series (Fig2a)
`<base>_EPI_HP_ROI.txt`, shape `(n_volumes, n_rois)`

### Structured dataset CSV (Fig2b)
Rows = ROIs, columns = time (seconds)

### ROI list (Fig2b)
One ROI acronym per line

### ABI parcellation table (Fig2b)
`ABI_template_2021_Full_ROIs_corrected.xlsx`

### fMRI runs + maps (Fig2c)
- 4D fMRI `.nii.gz`
- 4D maps NIfTI (Map1, Map2, ...)

### LC photometry (QC)
`<base>.txt`, length = n_volumes, aligned to fMRI TR.

---

## Running the scripts

### Fig2a — UMAP

```bash
python scripts/fig2a_umap.py   --roi_dir /path/to/ROI_urethane_LC   --burst_dir /path/to/LC_burst_outputs_full_dataset   --out_dir outputs
```

Output: `outputs/UMAP_Fig2_paper.svg`

---

### Fig2b — Ridgeline

```bash
python scripts/fig2b_ridgeline_waves.py   --input data/fMRI_urethane_timelocked_dataset.csv   --roi_list data/Paper_selected_roi.txt   --parcellation data/ABI_template_2021_Full_ROIs_corrected.xlsx   --tmin -40 --tmax 80   --out outputs/Fig2B_ridgeline.png
```

---

### Fig2c — Similarity + GS

```bash
python scripts/fig2c_similarity_gs.py   --fmri_dir /path/to/nifti_urethane_LC   --burst_dir /path/to/LC_burst_outputs_full_dataset   --maps_path /path/to/SVD_Derived_template.nii.gz   --out_dir outputs/Fig2C   --tr 1.2   --win_sec 60
```

---

### QC — LC–Map vs Global Signal controls

```bash
python scripts/qc_similarity_gs_lc_controls.py \
  --fmri_dir /path/to/nifti_urethane_LC \
  --burst_dir /path/to/LC_burst_outputs_full_dataset \
  --maps_path /path/to/SVD_Derived_template.nii.gz \
  --lc_dir /path/to/photometry_list_fMRI \
  --out_dir outputs/qc_fig2c \
  --tr 1.2 \
  --smooth_sec 8 \
  --n_perm 1000
```

This script produces run-level TSV tables only (no plots) and reproduces the statistics reported in Fig.2c and Supplementary QC.

---

## Reproducibility notes

UMAP results depend on:
- selected runs
- number of valid bursts
- ROIs removed due to NaNs
- QC statistics are computed only on runs where fMRI, LC photometry, and burst indices are all available.

---

## License

See `LICENSE`.
