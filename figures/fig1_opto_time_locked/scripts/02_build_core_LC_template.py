#!/usr/bin/env python3
"""
Build core_LC_template.csv from time-locked opto-fMRI datasets (ROI x time).

This script:
  1) Loads 3 CSVs (e.g., 3Hz/5Hz/15Hz), each ROI x time (columns are seconds).
  2) Optionally averages duplicated ROI names (bilateral averaging if present).
  3) Trims the time window (fixed: 0..30 s).
  4) Removes ROIs with any NaNs in ANY dataset.
  5) Computes cross-condition ROI similarity using diagonal correlation on raw ROI timecourses:
       sim_3_5, sim_3_15, sim_5_15  -> similarity_mean (per ROI)
  6) Selects "core ROIs" with similarity_mean > threshold.
  7) Builds core template by averaging the 3 datasets (ROI-wise mean across conditions).
  8) Saves:
       - core_LC_template.csv (ROI-wise mean across conditions)
       - core_rois.txt (list of selected ROIs)
       - roi_similarity_scores.csv (per-ROI similarity table)
      Optionally, writes NIfTI outputs if requested.

Notes:
  - Time window is fixed 0..30 s (inclusive), not exposed as an argument.
  - This script matches the numeric scale of the original template (no z-scoring).
  - Any normalization should be done downstream (e.g., SVD step).
  - Optional NIfTI outputs require nibabel and ROI parcellation + ROI table inputs.
  - Designed as a paper companion step between:
      01_make_time_locked_dataset.py  -> Opto_fMRI_dataset_*_left/right.csv
      03_svd_maps.py                  -> final Fig.1 outputs
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd


def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    # Ensure time columns can be interpreted as floats (seconds)
    df.columns = df.columns.astype(float)
    return df


def average_duplicated_roi_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    If ROI names are duplicated (e.g., bilateral averaging already merged names),
    average duplicates by index.
    """
    if not df.index.duplicated().any():
        return df
    dup = df[df.index.duplicated(keep=False)]
    avg = dup.groupby(dup.index).mean()
    others = df[~df.index.duplicated(keep=False)]
    out = pd.concat([others, avg]).sort_index()
    return out


def trim_time_window(df: pd.DataFrame, tmin: float, tmax: float) -> pd.DataFrame:
    cols = df.columns.astype(float)
    keep = (cols >= tmin) & (cols <= tmax)
    return df.loc[:, keep]


def diagonal_similarity(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.Series:
    """
    Per-ROI similarity between two datasets as the diagonal of ROIxROI correlation matrix.
    Requires identical ROI order and names in both dataframes.
    """
    if not df_a.index.equals(df_b.index):
        raise ValueError("ROI indices do not match between datasets. Ensure same ROIs in same order.")
    # corrcoef on rows: shape (2R x 2R), diag of cross-block
    corr = np.corrcoef(df_a.values, df_b.values)[:len(df_a), len(df_a):]
    return pd.Series(np.diag(corr), index=df_a.index, name="diag_corr")


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each row (ROI) across timepoints (columns).
    If std == 0 for a row, that row is set to zeros.
    """
    means = df.mean(axis=1)
    stds = df.std(axis=1, ddof=0)
    stds_replaced = stds.replace(0, np.nan)
    zscored = df.sub(means, axis=0).div(stds_replaced, axis=0)
    zscored = zscored.fillna(0.0)
    return zscored


def write_nifti_outputs(core_template: pd.DataFrame, parcellation_nii: str, roi_table_xlsx: str,
                        roi_name_col: int, roi_label_col: int, out_dir: str) -> None:
    import nibabel as nib

    # Load parcellation image
    img = nib.load(parcellation_nii)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    header = img.header

    # Load ROI table
    roi_table = pd.read_excel(roi_table_xlsx)
    # Extract ROI names and labels
    roi_names = roi_table.iloc[:, roi_name_col].astype(str).values
    roi_labels = roi_table.iloc[:, roi_label_col].values

    # Map ROI names in core_template to labels in roi_table
    # Build dict: roi_name -> label
    roi_name_to_label = dict(zip(roi_names, roi_labels))

    # Confirm all core_template ROIs are in roi_name_to_label
    missing_rois = [roi for roi in core_template.index if roi not in roi_name_to_label]
    if missing_rois:
        _eprint(f"[ERROR] The following core ROIs are missing from ROI table: {missing_rois}")
        sys.exit(1)

    # Prepare 4D data array: (X, Y, Z, time)
    shape_4d = data.shape + (core_template.shape[1],)
    data_4d = np.zeros(shape_4d, dtype=np.float32)

    # Fill 4D array: for each ROI, fill voxels with the ROI timecourse
    for roi_name in core_template.index:
        label = roi_name_to_label[roi_name]
        mask = data == label
        if not np.any(mask):
            _eprint(f"[WARNING] ROI label {label} for ROI '{roi_name}' not found in parcellation image.")
            continue
        # Fill all voxels of this ROI with the timecourse vector
        # Broadcast timecourse to voxels
        timecourse = core_template.loc[roi_name].values
        data_4d[mask, :] = timecourse

    # Save 4D NIfTI
    out_4d_path = os.path.join(out_dir, "core_LC_template_4D.nii.gz")
    nib.save(nib.Nifti1Image(data_4d, affine, header), out_4d_path)

    # Save 3D mean image (mean over time)
    data_3d = np.mean(data_4d, axis=3)
    out_3d_path = os.path.join(out_dir, "core_LC_template_image.nii.gz")
    nib.save(nib.Nifti1Image(data_3d, affine, header), out_3d_path)

    print(f"[OK] Saved NIfTI 4D: {out_4d_path}")
    print(f"[OK] Saved NIfTI 3D mean: {out_3d_path}")


def main():
    p = argparse.ArgumentParser(description="Create core_LC_template.csv from 3 opto time-locked datasets.")
    p.add_argument("--csv_3hz", required=True, help="Path to Opto_fMRI_dataset_3Hz_*.csv (ROI x time).")
    p.add_argument("--csv_5hz", required=True, help="Path to Opto_fMRI_dataset_5Hz_*.csv (ROI x time).")
    p.add_argument("--csv_15hz", required=True, help="Path to Opto_fMRI_dataset_15Hz_*.csv (ROI x time).")
    p.add_argument("--out_dir", required=True, help="Output directory.")
    p.add_argument("--similarity_thresh", type=float, default=0.75,
                   help="Core ROI threshold on mean diagonal similarity (default: 0.75).")
    p.add_argument("--average_duplicates", action="store_true",
                   help="If set, average duplicated ROI names (by index) before analysis.")
    p.add_argument("--verbose", action="store_true", help="If set, print summary.")
    p.add_argument("--parcellation_nii", type=str,
                   help="Path to parcellation NIfTI (required for --write_nifti).")
    p.add_argument("--roi_table_xlsx", type=str,
                   help="Path to ROI table Excel file (required for --write_nifti).")
    p.add_argument("--roi_table_name_col", type=int, default=1,
                   help="Column index for ROI names in ROI table (default: 1).")
    p.add_argument("--roi_table_label_col", type=int, default=0,
                   help="Column index for ROI labels in ROI table (default: 0).")
    p.add_argument("--write_nifti", action="store_true",
                   help="If set, write core template NIfTI outputs (requires nibabel and ROI inputs).")
    args = p.parse_args()

    TMIN = 0.0
    TMAX = 60.0

    for path in [args.csv_3hz, args.csv_5hz, args.csv_15hz]:
        if not os.path.isfile(path):
            _eprint(f"[ERROR] Input file not found: {path}")
            sys.exit(1)

    if args.write_nifti:
        if not args.parcellation_nii or not args.roi_table_xlsx:
            _eprint("[ERROR] --write_nifti requires --parcellation_nii and --roi_table_xlsx arguments.")
            sys.exit(1)
        if not os.path.isfile(args.parcellation_nii):
            _eprint(f"[ERROR] Parcellation NIfTI file not found: {args.parcellation_nii}")
            sys.exit(1)
        if not os.path.isfile(args.roi_table_xlsx):
            _eprint(f"[ERROR] ROI table Excel file not found: {args.roi_table_xlsx}")
            sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    df3 = load_csv(args.csv_3hz)
    df5 = load_csv(args.csv_5hz)
    df15 = load_csv(args.csv_15hz)

    # Optional duplicate averaging
    if args.average_duplicates:
        df3 = average_duplicated_roi_names(df3)
        df5 = average_duplicated_roi_names(df5)
        df15 = average_duplicated_roi_names(df15)

    # Trim time
    df3 = trim_time_window(df3, TMIN, TMAX)
    df5 = trim_time_window(df5, TMIN, TMAX)
    df15 = trim_time_window(df15, TMIN, TMAX)

    # Keep only ROIs with no NaNs in ANY dataset
    valid = ~(df3.isna().any(axis=1) | df5.isna().any(axis=1) | df15.isna().any(axis=1))
    df3 = df3.loc[valid]
    df5 = df5.loc[valid]
    df15 = df15.loc[valid]

    # Ensure identical ROI ordering
    common = df3.index.intersection(df5.index).intersection(df15.index)
    df3 = df3.loc[common].sort_index()
    df5 = df5.loc[common].sort_index()
    df15 = df15.loc[common].sort_index()

    # Similarities (diagonal correlations) on RAW (non z-scored) ROI timecourses
    sim_3_5 = diagonal_similarity(df3, df5).rename("sim_3_5")
    sim_3_15 = diagonal_similarity(df3, df15).rename("sim_3_15")
    sim_5_15 = diagonal_similarity(df5, df15).rename("sim_5_15")

    sim_table = pd.concat([sim_3_5, sim_3_15, sim_5_15], axis=1)
    sim_table["similarity_mean"] = sim_table.mean(axis=1)

    # Select core ROIs
    core_rois = sim_table.index[sim_table["similarity_mean"] > args.similarity_thresh].tolist()

    if args.verbose:
        print(f"[INFO] Time window fixed: {TMIN:.1f}..{TMAX:.1f} s")
        print(f"[INFO] Valid ROIs after NaNx filtering: {len(common)}")
        print(f"[INFO] Core ROIs (mean similarity > {args.similarity_thresh}): {len(core_rois)}")
        print("[INFO] Similarity and template are computed on RAW (non z-scored) ROI timecourses to match the original template numeric scale.")

    if len(core_rois) == 0:
        _eprint("[ERROR] No core ROIs selected. Lower --similarity_thresh or check inputs.")
        sys.exit(1)

    # Core datasets and template (mean across conditions) on RAW data
    df3_core = df3.loc[core_rois]
    df5_core = df5.loc[core_rois]
    df15_core = df15.loc[core_rois]
    core_template = (df3_core + df5_core + df15_core) / 3.0

    # Save outputs
    out_csv = os.path.join(args.out_dir, "core_LC_template.csv")
    core_template.to_csv(out_csv)

    out_rois = os.path.join(args.out_dir, "core_rois.txt")
    with open(out_rois, "w") as f:
        for r in core_rois:
            f.write(f"{r}\n")

    out_scores = os.path.join(args.out_dir, "roi_similarity_scores.csv")
    sim_table.to_csv(out_scores)

    if args.verbose:
        print(f"[OK] Saved: {out_csv}")
        print(f"[OK] Saved: {out_rois}")
        print(f"[OK] Saved: {out_scores}")

    if args.write_nifti:
        write_nifti_outputs(core_template, args.parcellation_nii, args.roi_table_xlsx,
                            args.roi_table_name_col, args.roi_table_label_col, args.out_dir)


if __name__ == "__main__":
    main()