#!/usr/bin/env python3
"""
SVD-based temporal progression and spatial map extraction (paper companion)

Workflow:
  1) Load ROI x time CSV (columns are seconds).
  2) Keep time window [0, T_MAX].
  3) Z-score each ROI across time.
  4) SVD on X = (time x ROI) matrix: X = U S V^T
  5) Keep first K modes (cumulative EVR >= VAR_THRESHOLD, capped by MAX_MODES)
  6) Define per-timepoint "dominant mode" based on |U[:,i]*S[i]| contribution (dominance is always determined using absolute contributions; no option).
  7) For each mode i, regress ROI activity on contribution vector within its dominant phase
  8) Save:
     - spatial_maps.csv (ROI x K maps)
     - roi_by_map_values.csv (same as spatial_maps.csv)
     - dominant_indices.json (time indices per mode)
     - figures (PNG):
         combined_roi_contribution_histograms.png
         Map_<k>_projection_heatmap.png
     - Optional: NIfTI maps per Map_<k>.nii.gz and Map_<k>_bilateral.nii.gz

Inputs:
  - core_LC_template.csv (ROI x time), columns must be parseable as float seconds.

NIfTI export (optional):
  - Provide --parcellation_nii and --roi_table plus mapping columns.
  - ROI table can be xlsx or csv; you specify the column indices (0-based) for ROI name and ROI label.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional NIfTI
try:
    import nibabel as nib  # type: ignore
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False


# -----------------------------
# Utils
# -----------------------------

def _eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def load_and_prepare(csv_path: str, t_max: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load ROI x time CSV, keep columns within [0, t_max], z-score per ROI across time."""
    df = pd.read_csv(csv_path, index_col=0)
    df.columns = df.columns.astype(float)
    df = df.loc[:, (df.columns >= 0.0) & (df.columns <= t_max)]
    time_s = df.columns.values

    # Z-score per ROI (row-wise)
    df_proc = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df_proc, time_s


def svd_decompose(df_proc: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SVD of Time x ROI matrix; return U, S, Vt, EVR."""
    X = df_proc.T.values  # (T x R)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S2 = S ** 2
    evr = S2 / (S2.sum() + 1e-12)
    return U, S, Vt, evr


def choose_num_modes(evr: np.ndarray, var_threshold: float, max_modes: int) -> int:
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, var_threshold) + 1)
    k = max(1, min(k, len(evr), max_modes))
    return k


def temporal_contributions(U: np.ndarray, S: np.ndarray, k: int) -> np.ndarray:
    """Absolute temporal contributions used for dominance: |U[:,i] * S[i]| (fixed)."""
    C = U[:, :k] * S[:k][None, :]
    return np.abs(C)


def find_dominance_phases(C: np.ndarray) -> np.ndarray:
    return np.argmax(C, axis=1)


def spatial_regression_maps(
    df_proc: pd.DataFrame,
    U: np.ndarray,
    S: np.ndarray,
    phases: np.ndarray,
    k: int,
    min_timepoints_per_phase: int
) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    """
    For each mode i, select timepoints where it is dominant, then regress ROI activity on a_i = U[:,i]*S[i].
    w_i = (X^T a) / (a^T a)
    """
    roi_names = df_proc.index.to_list()
    X = df_proc.T.values  # (T x R)

    maps: Dict[str, np.ndarray] = {}
    idx_by_mode: Dict[int, List[int]] = {}

    for i in range(k):
        idx = np.where(phases == i)[0]
        if idx.size < min_timepoints_per_phase:
            continue

        a = (U[idx, i] * S[i]).reshape(-1, 1)  # (T_phase x 1)
        denom = float((a.T @ a).squeeze())
        if denom == 0:
            continue

        X_phase = X[idx, :]  # (T_phase x R)
        w = (X_phase.T @ a) / denom  # (R x 1)

        maps[f"Map_{i+1}"] = w.squeeze()
        idx_by_mode[i] = idx.tolist()

    maps_df = pd.DataFrame(maps, index=roi_names)
    return maps_df, idx_by_mode


def save_png(fig, path: str, dpi: int = 300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _load_roi_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, header=None)
    return pd.read_csv(path, header=None)


def export_nifti_maps(
    maps_df: pd.DataFrame,
    parcellation_nii: str,
    roi_table_path: str,
    roi_table_name_col: int,
    roi_table_label_col: int,
    out_dir: str,
    also_bilateral: bool = True,
):
    if not _HAS_NIB:
        _eprint("[INFO] nibabel not available; skipping NIfTI export.")
        return
    if not (parcellation_nii and roi_table_path):
        _eprint("[INFO] parcellation_nii or roi_table_path not provided; skipping NIfTI export.")
        return
    if not (os.path.exists(parcellation_nii) and os.path.exists(roi_table_path)):
        _eprint("[INFO] Parcellation or ROI table not found on disk; skipping NIfTI export.")
        return

    roi_table = _load_roi_table(roi_table_path)
    if roi_table_name_col >= roi_table.shape[1] or roi_table_label_col >= roi_table.shape[1]:
        _eprint("[INFO] ROI table column indices out of range; skipping NIfTI export.")
        return

    # Map ROI name -> list of labels (potentially multiple rows with same name for L/R)
    name_to_labels: Dict[str, List[int]] = {}
    for _, row in roi_table.iterrows():
        name = str(row.iloc[roi_table_name_col]).strip()
        try:
            label = int(row.iloc[roi_table_label_col])
        except Exception:
            continue
        if name not in name_to_labels:
            name_to_labels[name] = []
        name_to_labels[name].append(label)

    parc_img = nib.load(parcellation_nii)
    parc_data = parc_img.get_fdata()
    affine = parc_img.affine
    hdr = parc_img.header

    # If not bilateral: use first matching label only (still deterministic)
    for map_name in maps_df.columns:
        vol = np.zeros_like(parc_data, dtype=np.float32)
        for roi_name, weight in maps_df[map_name].items():
            labels = name_to_labels.get(str(roi_name), [])
            if not labels:
                continue
            # "single" version: fill ONLY first label
            lab = labels[0]
            vol[parc_data == lab] = float(weight)

        out_path = os.path.join(out_dir, f"{map_name}.nii.gz")
        nib.save(nib.Nifti1Image(vol, affine, header=hdr), out_path)

        if also_bilateral:
            vol_bi = np.zeros_like(parc_data, dtype=np.float32)
            for roi_name, weight in maps_df[map_name].items():
                labels = name_to_labels.get(str(roi_name), [])
                for lab in labels:  # fill all matches (e.g., left+right)
                    vol_bi[parc_data == lab] = float(weight)
            out_bi = os.path.join(out_dir, f"{map_name}_bilateral.nii.gz")
            nib.save(nib.Nifti1Image(vol_bi, affine, header=hdr), out_bi)


# -----------------------------
# Deterministic map sign orientation
# -----------------------------

def _orient_map_sign_by_top_rois(
    maps_df: pd.DataFrame,
    df_ref: pd.DataFrame,
    top_n: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Make SVD map signs deterministic based on the reference (core template) timecourses.

    Rule (per map):
      - Take the top `top_n` ROIs by |weight|.
      - For each ROI, compute sign(mean(timecourse over 0..T_MAX)) from df_ref.
      - Compare with sign(weight). If most of the informative ROIs disagree, flip the entire map.

    Returns
    -------
    maps_df_out : DataFrame
        Possibly sign-flipped maps.
    flips : dict
        Map name -> +1 (kept) or -1 (flipped)
    """
    maps_df_out = maps_df.copy()
    flips: Dict[str, int] = {}

    # Precompute reference signs for ROIs present in df_ref
    # sign_ref[roi] = sign(mean over time window)
    ref_means = df_ref.mean(axis=1)
    ref_sign = np.sign(ref_means).replace(0, np.nan)  # treat exact zeros as non-informative

    for map_name in maps_df_out.columns:
        w = maps_df_out[map_name]
        # Top ROIs by absolute weight
        top_rois = w.abs().sort_values(ascending=False).head(top_n).index.tolist()

        agreements = []
        for roi in top_rois:
            if roi not in ref_sign.index:
                continue
            s_ref = ref_sign.loc[roi]
            if pd.isna(s_ref):
                continue
            s_w = np.sign(w.loc[roi])
            if s_w == 0:
                continue
            agreements.append(float(s_ref) * float(s_w))  # +1 agree, -1 disagree

        # If not enough info, keep default orientation
        if len(agreements) == 0:
            flips[map_name] = +1
            continue

        # Majority vote: if sum < 0 => more disagreement => flip
        vote = np.sum(agreements)
        if vote < 0:
            maps_df_out[map_name] = -maps_df_out[map_name]
            flips[map_name] = -1
        else:
            flips[map_name] = +1

    return maps_df_out, flips




def plot_combined_roi_histograms(maps_df: pd.DataFrame, out_dir: str, max_rois_label: int = 120):
    """
    For each map, plot ROI weights as bars. If there are many ROIs, x-labels become unreadable;
    we keep them but you can adjust max_rois_label or turn them off easily.
    """
    roi_names = maps_df.index.tolist()
    all_weights = maps_df.values.flatten()
    global_ymin = float(np.nanmin(all_weights))
    global_ymax = float(np.nanmax(all_weights))

    num_maps = maps_df.shape[1]
    fig, axes = plt.subplots(num_maps, 1, figsize=(14, 4 * num_maps), sharex=True, sharey=True)
    if num_maps == 1:
        axes = [axes]

    for j, map_name in enumerate(maps_df.columns):
        ax = axes[j]
        w = maps_df[map_name].values

        # Diverging color map based on weight magnitude (match old behavior)
        norm = (w - global_ymin) / (global_ymax - global_ymin + 1e-9)
        colors = plt.cm.RdBu_r(norm)

        ax.bar(range(len(roi_names)), w, width=0.8, color=colors)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylim([global_ymin, global_ymax])
        ax.set_ylabel("Weight (a.u.)")
        ax.set_title(f"ROI contributions for {map_name}")

        if j == num_maps - 1:
            if len(roi_names) <= max_rois_label:
                ax.set_xticks(range(len(roi_names)))
                ax.set_xticklabels(roi_names, rotation=90, fontsize=7)
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

    plt.tight_layout()
    save_png(fig, os.path.join(out_dir, "combined_roi_contribution_histograms.png"))


def plot_projection_heatmaps(time_s: np.ndarray, df_proc: pd.DataFrame, maps_df: pd.DataFrame, out_dir: str):
    """
    Heatmap of projection amplitude for each map over time.
    H = X @ W -> (T x K), plot each column as 1xT heatmap.
    """
    X = df_proc.T.values  # (T x R)
    W = maps_df.values    # (R x K)
    H = X @ W             # (T x K)

    global_vmin = float(np.min(H))
    global_vmax = float(np.max(H))

    for i, map_name in enumerate(maps_df.columns):
        fig = plt.figure(figsize=(10, 2.5))
        plt.imshow(H[:, i].reshape(1, -1), aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=global_vmin, vmax=global_vmax)
        plt.colorbar(label="Projected amplitude (a.u.)")
        xt = np.linspace(0, len(time_s) - 1, 6, dtype=int)
        plt.xticks(xt, [f"{time_s[ii]:.0f}" for ii in xt])
        plt.yticks([])
        plt.xlabel("Time (s)")
        plt.title(f"Temporal projection onto {map_name}")
        plt.tight_layout()
        save_png(fig, os.path.join(out_dir, f"{map_name}_projection_heatmap.png"))


# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="SVD-based map extraction from core_LC_template.csv")
    p.add_argument("--csv", required=True, help="Path to core_LC_template.csv (ROI x time).")
    p.add_argument("--out_dir", required=True, help="Output directory for figures/CSVs/maps.")
    p.add_argument("--t_max", type=float, default=30.0, help="Max time (s) to keep (default: 30).")

    p.add_argument("--var_threshold", type=float, default=0.95, help="Cumulative EVR threshold (default: 0.95).")
    p.add_argument("--max_modes", type=int, default=6, help="Maximum number of modes to keep (default: 6).")
    p.add_argument("--min_timepoints_per_phase", type=int, default=5,
                   help="Minimum timepoints for a dominant phase to fit a map (default: 5).")

    # Optional NIfTI export
    p.add_argument("--parcellation_nii", default=None, help="Parcellation NIfTI with integer ROI labels (optional).")
    p.add_argument("--roi_table", default=None, help="ROI mapping table (xlsx/csv) (optional).")
    p.add_argument("--roi_table_name_col", type=int, default=1, help="0-based column index for ROI name (default: 1).")
    p.add_argument("--roi_table_label_col", type=int, default=0, help="0-based column index for ROI label (default: 0).")
    p.add_argument("--no_bilateral", action="store_true", help="If set, do not save *_bilateral.nii.gz maps.")

    p.add_argument("--verbose", action="store_true", help="Print progress messages.")
    args = p.parse_args()

    if not os.path.isfile(args.csv):
        _eprint(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.verbose:
        print("[1/6] Loading and preprocessing…")
    df_proc, time_s = load_and_prepare(args.csv, args.t_max)

    if args.verbose:
        print(f"      Data shape (ROI x time): {df_proc.shape}")

    if args.verbose:
        print("[2/6] SVD decomposition…")
    U, S, Vt, evr = svd_decompose(df_proc)
    k = choose_num_modes(evr, args.var_threshold, args.max_modes)

    if args.verbose:
        print(f"      Keeping K={k} modes (cumulative EVR={evr[:k].sum():.3f})")

    if args.verbose:
        print("[3/6] Temporal contributions and dominance…")
    C = temporal_contributions(U, S, k)
    phases = find_dominance_phases(C)

    if args.verbose:
        print("[4/6] Spatial regression maps…")
    maps_df, idx_by_mode = spatial_regression_maps(
        df_proc=df_proc,
        U=U, S=S,
        phases=phases,
        k=k,
        min_timepoints_per_phase=args.min_timepoints_per_phase
    )

    if maps_df.empty:
        _eprint("[ERROR] No spatial maps produced. Try lowering --min_timepoints_per_phase or adjusting thresholds.")
        sys.exit(1)

    # --- Deterministic sign orientation (per map) based on top-3 ROIs and core template sign ---
    # Load reference (raw core template values; not z-scored here)
    df_ref = pd.read_csv(args.csv, index_col=0)
    df_ref.columns = df_ref.columns.astype(float)
    df_ref = df_ref.loc[:, (df_ref.columns >= 0.0) & (df_ref.columns <= args.t_max)]

    # Align reference to maps_df ROI order (intersection only)
    common_rois = maps_df.index.intersection(df_ref.index)
    maps_df = maps_df.loc[common_rois]
    df_ref = df_ref.loc[common_rois]

    maps_df, sign_flips = _orient_map_sign_by_top_rois(maps_df, df_ref, top_n=3)

    if args.verbose:
        for mn, sf in sign_flips.items():
            if sf == -1:
                print(f"[INFO] Sign flip applied to {mn} based on top-3 ROI reference sign.")

    # Save core tables
    maps_csv = os.path.join(args.out_dir, "spatial_maps.csv")
    maps_df.to_csv(maps_csv)

    roi_map_csv = os.path.join(args.out_dir, "roi_by_map_values.csv")
    maps_df.to_csv(roi_map_csv)

    # Meta + indices

    with open(os.path.join(args.out_dir, "dominant_indices.json"), "w") as f:
        json.dump({f"Mode_{i+1}": idx for i, idx in idx_by_mode.items()}, f, indent=2)

    # Figures (PNG)
    figs_dir = os.path.join(args.out_dir, "figures_png")
    os.makedirs(figs_dir, exist_ok=True)

    if args.verbose:
        print("[5/6] Saving figures (PNG)…")
    plot_combined_roi_histograms(maps_df, figs_dir)
    plot_projection_heatmaps(time_s, df_proc, maps_df, figs_dir)

    # Optional NIfTI export
    if args.parcellation_nii and args.roi_table:
        nifti_dir = os.path.join(args.out_dir, "nifti_maps")
        os.makedirs(nifti_dir, exist_ok=True)
        if args.verbose:
            print("[6/6] Exporting NIfTI maps…")
        export_nifti_maps(
            maps_df=maps_df,
            parcellation_nii=args.parcellation_nii,
            roi_table_path=args.roi_table,
            roi_table_name_col=args.roi_table_name_col,
            roi_table_label_col=args.roi_table_label_col,
            out_dir=nifti_dir,
            also_bilateral=(not args.no_bilateral),
        )
    else:
        if args.verbose:
            print("[6/6] NIfTI export skipped (no --parcellation_nii/--roi_table provided).")

    if args.verbose:
        print("[DONE] Outputs written to:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()