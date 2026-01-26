# A Conserved Locus Coeruleus fMRI Signature of Brain-State Transitions across Sleep, Anesthesia, and Wakefulness — Figure Code

This repository contains the **analysis code used to generate the main figures** for the manuscript:

**A Conserved Locus Coeruleus fMRI Signature of Brain-State Transitions across Sleep, Anesthesia, and Wakefulness**

The code is **figure-oriented and intentionally minimal**: each figure folder reproduces the corresponding analyses and plots from precomputed intermediate data (ROI time series, similarity traces, transition indices).

---

## Repository structure

```
paper_figures/
└── figures/
    ├── fig1_opto_time_locked/             # Optogenetic LC time-locked fMRI (template + SVD maps)
    ├── fig2_photometry-urethane-fmri/     # LC photometry (urethane) ↔ fMRI (UMAP, ridgeline, similarity+GS)
    ├── fig3_4_sleep_transitions/          # Sleep transitions (combined NPZ pipeline)
    └── fig5_human_restingstate/           # Human awake resting-state LC-map event-wise analysis
```

Each folder contains a dedicated `README.md` describing expected inputs, assumptions, and outputs.

---

## Figure summaries

### Figure 1 — Optogenetic LC time-locked fMRI
Builds a **core LC temporal template** from ROI×time datasets across stimulation frequencies and extracts **SVD-derived spatial maps** used in the figure.

### Figure 2 — Photometry (urethane) ↔ fMRI
Reproduces:
- **UMAP** embedding of burst-aligned ROI dynamics  
- **Ridgeline** visualization of burst-aligned ROI waveforms  
- **Group LC-map similarity** and **Global Signal (GS)** statistics (+ QC controls)

### Figures 3–4 — Sleep transition similarity (combined NPZ pipeline)
Runs a small pipeline operating on a **single combined NPZ** containing:
- per-subject similarity time series  
- state vectors  
- transition indices (WAKE→NREM and NREM→WAKE)

Outputs include group traces, subject heatmaps, and **event-wise AUC** statistics (real transitions vs random-window null).

### Figure 5 — Human resting-state (awake)
Performs event-wise analyses from **precomputed similarity traces** and LC event indices:
- peak-aligned averaging  
- event-wise AUC vs random-window null  

---

## Fig. 3–4 combined NPZ specification (only for `fig3_4_sleep_transitions/`)

The pipeline expects a `similarity_trace_combined.npz` with:

- `sims`: `(n_subjects, max_T, n_maps)` float, padded with NaNs  
- `state`: `(n_subjects, max_T)` float, mouse-like coding: **1=WAKE, 3=NREM**, NaN ignored  
- `tr`: float, TR in seconds  
- `subjects`: `(n_subjects,)` object array of strings  
- `transitions_1to3`: object array (len `n_subjects`) of int arrays (indices `t` where `state[t]=1` and `state[t+1]=3`)  
- `transitions_3to1`: object array (len `n_subjects`) of int arrays (indices `t` where `state[t]=3` and `state[t+1]=1`)  

Epochs are extracted as **±WIN_SECONDS** around each transition, where `WIN_SECONDS` is defined in  
`fig3_4_sleep_transitions/scripts/config/settings.py`.

---

## Installation

Dependencies are lightweight (NumPy, SciPy, Pandas, Matplotlib); some figures may require optional neuroimaging libraries for NIfTI export.

---

## Data availability

Each figure folder documents the expected input formats (e.g., ROI×time CSV/TXT, similarity traces, event indices, combined NPZ).

---

## How to run

See the `README.md` inside each figure folder for exact commands.

---

## License

See `LICENSE` in the repository root.

---
