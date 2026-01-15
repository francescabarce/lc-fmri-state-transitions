#!/usr/bin/env python3
"""
fig2c_similarity_gs.py — Group burst-aligned SVD-map similarity + Global Signal (Fig.2c)
---------------------------------------------------------------------------------------

Goal (Fig.2c):
- Compute per-run timecourses:
  (1) Map similarity (spatial regression / dot-product) for Map1 & Map2
  (2) Global Signal (GS) from the SAME voxel mask
- Align Map1/Map2/GS to LC burst indices (t=0) and plot group mean ± SEM.

Critical methodological note :
- Similarity is computed exactly as:
    map_vec = zscore(map within mask)
    X_vox = zscore(fMRI time series within mask, per voxel across time)
    similarity(t) = dot(X_vox(:, t), map_vec) / (Nvox - 1)
  i.e., each voxel is normalized across time before computing the dot-product at each timepoint.

Inputs (recommended repo-style paths)
------------------------------------
- fMRI runs (4D NIfTI .nii.gz) in a folder
- Burst indices (volume indices) per run: <base>_burst_indices.txt
- Merged maps NIfTI (4D): Map_1, Map_2, ... in last dim

Outputs
-------
- Fig2C_group_burst_aligned_similarity_gs.svg
- (optional) cache: per-run similarity_zscores + gs_zscore in output/Similarity/

Also prints
-----------
- GS statistics: R^2(Map ~ GS) and signed corr(Map, GS) per map (across runs).

Example repo:
-------

python scripts/fig2c_similarity_gs.py \
  --fmri_dir ./nifti_urethane_LC \
  --burst_dir ./LC_burst_outputs \
  --maps_path ./SVD_Derived_template.nii.gz \
  --lc_dir ./photometry_list_fMRI \
  --example_base VZE000126_run-16 \
  --out_dir ./Figure2 \
  --tr 1.2 \
  --win_sec 60
"""

import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr

plt.rcParams["svg.fonttype"] = "none"


# ==========================
# ====== HELPERS ===========
# ==========================
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w) / float(w)
    return np.convolve(x, k, mode="same")


def compute_global_signal(data_2d: np.ndarray) -> np.ndarray:
    """
    data_2d: (n_vox_in_mask, n_vols)
    """
    return np.nanmean(data_2d, axis=0)


def compute_similarity_timecourse(map_idx: int,
                                  maps_4d: np.ndarray,
                                  mask_3d: np.ndarray,
                                  data_2d: np.ndarray) -> np.ndarray:
    """
    map_idx: which map in last dim
    maps_4d: (X,Y,Z,n_maps)
    mask_3d: boolean mask selecting voxels
    data_2d: (n_vox_in_mask, n_vols) flattened fmri data within mask

    Similarity(t) = dot(zscore_vox_over_time(vol(t)), map_vec) / (Nvox - 1)
    where zscore_vox_over_time indicates that each voxel is z-scored across time (within-run).
    """
    map_vec = maps_4d[..., map_idx][mask_3d]
    if np.all(map_vec == 0):
        return np.zeros(data_2d.shape[1], dtype=float)

    # z-score each voxel across time (within-run)
    # X is (vox, vols); normalize along the time axis for each voxel
    X = data_2d.astype(float).copy()  # (vox, vols)
    m = np.mean(X, axis=1, keepdims=True)
    s = np.std(X, axis=1, keepdims=True)
    s[s == 0] = 1.0
    Xz = (X - m) / s

    # dot per volume (timepoint)
    nvox = map_vec.size
    sim = (map_vec @ Xz) / float(max(nvox - 1, 1))
    return sim


def linear_r2(y: np.ndarray, x: np.ndarray) -> float:
    """R^2 of y explained by x using linear regression with intercept."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    if y.size < 5:
        return np.nan
    A = np.column_stack([x, np.ones_like(x)])
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - (ss_res / ss_tot)


def signed_corr(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    if y.size < 5:
        return np.nan
    r, _ = pearsonr(y, x)
    return float(r)


# ==========================
# ========= MAIN ===========
# ==========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fmri_dir", required=True,
                    help="Folder with 4D fMRI NIfTI files (.nii.gz).")
    ap.add_argument("--burst_dir", required=True,
                    help="Folder with <base>_burst_indices.txt files.")
    ap.add_argument("--maps_path", required=True,
                    help="4D NIfTI containing SVD maps (e.g., merged maps).")
    ap.add_argument("--out_dir", default="outputs/fig2",
                    help="Output directory (will be created).")
    ap.add_argument("--tr", type=float, default=1.2, help="TR in seconds.")
    ap.add_argument("--win_sec", type=float, default=60.0,
                    help="Half-window in seconds for burst alignment (±win_sec).")
    ap.add_argument("--smooth_sec", type=float, default=0.0,
                    help="Optional smoothing (moving average) in seconds. 0 disables.")
    ap.add_argument("--cache_timeseries", action="store_true",
                    help="If set, saves per-run similarity_zscores and gs_zscore in out_dir/Similarity/.")
    ap.add_argument(
        "--lc_dir",
        default=None,
        help=(
            "Optional folder with LC photometry time series per run as <base>.txt (length = n_vols). "
            "If provided, a single-subject example panel will be generated."
        ),
    )
    ap.add_argument(
        "--example_base",
        default="VZE000126_run-16",
        help="Base run-id for the single-subject example (e.g., VZE000126_run-16).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sim_cache_dir = os.path.join(args.out_dir, "Similarity")
    if args.cache_timeseries:
        os.makedirs(sim_cache_dir, exist_ok=True)

    # Window in volumes
    pre = int(round(args.win_sec / args.tr))
    post = int(round(args.win_sec / args.tr))
    win = pre + post + 1
    time_axis = (np.arange(win) - pre) * args.tr

    smooth_win = int(round(args.smooth_sec / args.tr)) if args.smooth_sec > 0 else 1

    # Load maps + build mask
    maps_img = nib.load(args.maps_path)
    maps = maps_img.get_fdata()
    if maps.ndim != 4:
        raise RuntimeError("maps_path must be a 4D NIfTI with maps in last dimension.")
    n_maps = maps.shape[3]
    if n_maps < 2:
        raise RuntimeError("Need at least 2 maps (Map1, Map2) for Fig.2c.")
    mask = np.any(maps != 0, axis=3)

    # Containers for burst-aligned segments
    map_segments = [[] for _ in range(2)]  # Map1, Map2 only
    gs_segments = []

    # GS stats per run
    gs_stats = {1: {"r2": [], "r": []}, 2: {"r2": [], "r": []}}
    n_runs_used = 0

    fmri_files = sorted([f for f in os.listdir(args.fmri_dir) if f.endswith(".nii.gz")])
    if len(fmri_files) == 0:
        raise RuntimeError(f"No .nii.gz files found in {args.fmri_dir}")

    example = None

    for fmri_file in fmri_files:
        base = "_".join(fmri_file.split("_")[:2])  # keep your convention
        burst_path = os.path.join(args.burst_dir, f"{base}_burst_indices.txt")
        if not os.path.exists(burst_path):
            continue

        # Load fMRI
        fmri_4d = nib.load(os.path.join(args.fmri_dir, fmri_file)).get_fdata()
        if fmri_4d.ndim != 4:
            continue
        n_vols = fmri_4d.shape[3]

        # Flatten masked voxels
        data_2d = fmri_4d.reshape(-1, n_vols)[mask.flatten(), :]  # (vox, vols)

        # Compute GS (same mask voxels), zscore
        gs = zscore(compute_global_signal(data_2d))
        gs = moving_average(gs, smooth_win)
        gs = zscore(gs)

        # Compute similarity for Map1/Map2, then zscore per run
        sim1 = compute_similarity_timecourse(0, maps, mask, data_2d)
        sim2 = compute_similarity_timecourse(1, maps, mask, data_2d)

        sim1 = moving_average(sim1, smooth_win)
        sim2 = moving_average(sim2, smooth_win)
        sim1z = zscore(sim1)
        sim2z = zscore(sim2)

        # Store traces for a single-subject example (if requested)
        if args.lc_dir is not None and base == args.example_base:
            example = {
                "base": base,
                "sim1z": sim1z.copy(),
                "sim2z": sim2z.copy(),
                "gs": gs.copy(),
                "n_vols": n_vols,
            }

        # Cache (optional)
        if args.cache_timeseries:
            np.savetxt(os.path.join(sim_cache_dir, f"{base}_global_signal_zscore.txt"), gs)
            np.savetxt(os.path.join(sim_cache_dir, f"{base}_similarity_zscores.txt"),
                       np.column_stack([sim1z, sim2z]),
                       header="Map1 Map2")

        # GS stats per map (run-level)
        gs_stats[1]["r2"].append(linear_r2(sim1, gs))
        gs_stats[1]["r"].append(signed_corr(sim1, gs))
        gs_stats[2]["r2"].append(linear_r2(sim2, gs))
        gs_stats[2]["r"].append(signed_corr(sim2, gs))

        # Burst alignment
        # Burst indices are sometimes stored as floats (e.g., 1.5000000e+01).
        # Avoid NumPy float->int parsing deprecation by loading as float then casting.
        bursts = np.loadtxt(burst_path, dtype=float)
        bursts = np.atleast_1d(bursts).astype(np.int64)

        used_any = False
        for b in bursts:
            if b < pre or (b + post) >= n_vols:
                continue
            map_segments[0].append(sim1z[b - pre: b + post + 1])
            map_segments[1].append(sim2z[b - pre: b + post + 1])
            gs_segments.append(gs[b - pre: b + post + 1])
            used_any = True

        if used_any:
            n_runs_used += 1

    # Convert to arrays
    map_segments = [np.array(m) for m in map_segments]
    gs_segments = np.array(gs_segments) if len(gs_segments) else np.empty((0, win))

    if map_segments[0].size == 0 or map_segments[1].size == 0 or gs_segments.size == 0:
        raise RuntimeError("No burst-aligned segments collected. Check burst indices, fmri_dir, naming, and window.")

    # Group mean + SEM
    means = [np.nanmean(m, axis=0) for m in map_segments]
    sems = [np.nanstd(m, axis=0, ddof=1) / np.sqrt(m.shape[0]) for m in map_segments]

    gs_mean = np.nanmean(gs_segments, axis=0)
    gs_sem = np.nanstd(gs_segments, axis=0, ddof=1) / np.sqrt(gs_segments.shape[0])

    # ----------- Print GS stats -----------
    print("\n====================================================")
    print("GLOBAL SIGNAL STATISTICS (run-level; same mask voxels)")
    print(f"Runs used (>=1 burst in-window): {n_runs_used}")
    print(f"Total bursts used: {gs_segments.shape[0]}")
    print("----------------------------------------------------")
    for m in [1, 2]:
        r2 = np.array(gs_stats[m]["r2"], dtype=float)
        rr = np.array(gs_stats[m]["r"], dtype=float)
        r2 = r2[np.isfinite(r2)]
        rr = rr[np.isfinite(rr)]
        print(f"Map {m}:")
        if r2.size:
            print(f"  R^2(Map ~ GS): mean={np.mean(r2):.3f} | median={np.median(r2):.3f} | n={r2.size}")
        else:
            print("  R^2(Map ~ GS): n/a")
        if rr.size:
            print(f"  signed corr(Map, GS): mean={np.mean(rr):.3f} | median={np.median(rr):.3f} | n={rr.size}")
        else:
            print("  signed corr(Map, GS): n/a")
    print("====================================================\n")

    # ----------- Plot (Fig.2c) -----------
    fig = plt.figure(figsize=(10, 3.8))
    ax = plt.gca()
    ax.set_facecolor("#F0F0F0")

    # Keep your paper colors (matches your earlier figure style)
    colors = ["#0D0887", "#F89441"]  # Map1, Map2

    for i in [0, 1]:
        ax.plot(time_axis, means[i], color=colors[i], linewidth=2.2, label=f"Map {i+1}")
        ax.fill_between(time_axis, means[i] - sems[i], means[i] + sems[i],
                        color=colors[i], alpha=0.25)

    # Removed GS curve for the group plot as per instructions
    # ax.plot(time_axis, gs_mean, color="black", linewidth=2.0, alpha=0.85, label="Global signal")
    # ax.fill_between(time_axis, gs_mean - gs_sem, gs_mean + gs_sem, color="black", alpha=0.15)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Time from burst (s)", fontsize=14)
    ax.set_ylabel("Map Regression (z)", fontsize=14)
    ax.set_title("Fig.2c — Group burst-aligned SVD map expression and Global Signal", fontsize=14)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    out_png = os.path.join(args.out_dir, "Fig2C_group_burst_aligned_similarity.svg")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"[OK] Saved Fig.2c PNG: {out_png}")

    # ----------- Single-subject example (Fig.2c inset/panel) -----------
    if args.lc_dir is not None:
        if example is None:
            print(
                f"[WARN] --lc_dir provided but no example traces were captured for base='{args.example_base}'. "
                "Check --example_base and file naming."
            )
        else:
            lc_path = os.path.join(args.lc_dir, f"{example['base']}.txt")
            if not os.path.exists(lc_path):
                print(f"[WARN] LC file not found for example: {lc_path}")
            else:
                lc = np.squeeze(np.loadtxt(lc_path))
                if lc.ndim != 1:
                    lc = lc.reshape(-1)
                if lc.size != example["n_vols"]:
                    # Resize defensively to match fMRI length (same behavior as your earlier scripts)
                    lc = np.resize(lc, example["n_vols"])

                lc = zscore(moving_average(lc, smooth_win))
                t = np.arange(example["n_vols"]) * args.tr

                fig_ex = plt.figure(figsize=(10, 3.2))
                ax_ex = plt.gca()
                ax_ex.set_facecolor("#F0F0F0")

                colors = ["#0D0887", "#F89441"]
                ax_ex.plot(t, example["sim1z"], color=colors[0], linewidth=2.0, label="Map 1")
                ax_ex.plot(t, example["sim2z"], color=colors[1], linewidth=2.0, label="Map 2")
                ax_ex.plot(t, lc, color="#8E0000", linewidth=2.5, label="LC")

                ax_ex.set_xlabel("Time (s)")
                ax_ex.set_ylabel("z")
                ax_ex.set_title(f"Fig.2c — Single-subject example: {example['base']}")
                ax_ex.legend(frameon=False, ncol=3)
                ax_ex.grid(alpha=0.25)

                out_ex = os.path.join(args.out_dir, f"Fig2C_single_subject_example_{example['base']}.png")
                fig_ex.tight_layout()
                fig_ex.savefig(out_ex, dpi=300)
                plt.close(fig_ex)
                print(f"[OK] Saved single-subject example PNG: {out_ex}")


if __name__ == "__main__":
    main()