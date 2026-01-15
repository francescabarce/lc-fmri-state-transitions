#!/usr/bin/env python3
"""
qc_similarity_gs_lc_controls.py — QC: LC–Map coupling vs Global Signal (GS)
-----------------------------------------------------------------------------------------

Goal
----
Reproduce the same run-level numbers produced by the original script
`Similarity_frames_Opto_urethane_FC_GS.py`, but in a clean CLI form.

Per run, this script computes:
1) Map similarity time courses (spatial regression / dot-product)
   - map_vec = zscore(map within mask)                       [z-score across voxels]
   - voxel_ts(v) = zscore(fMRI voxel v across time)          [z-score across time]
   - similarity(t) = dot(voxel_ts(:, t), map_vec) / (Nvox-1)

2) Global signal (GS)
   - GS_raw(t) = mean BOLD across the same mask voxels
   - IMPORTANT (to match original numbers):
       * For Map~GS stats (R^2 and signed corr), use GS_raw = zscore(GS_raw) with NO smoothing.
       * For partial correlation LC–Map | GS, use GS_s = zscore(moving_average(GS_raw)).

3) LC–Map zero-lag correlation (r, p_perm)
   - Pearson correlation between:
       sim_s = zscore(moving_average(sim))
       lc_s  = zscore(moving_average(lc))
   - p_perm via phase-randomization surrogate test (preserves autocorrelation).

4) Map ~ GS shared variance (R^2) + signed corr(Map, GS)
   - Using sim (UNsmoothed) and GS_raw (UNsmoothed), as in the original script.

5) Partial correlation LC–Map | GS (r_partial, p_perm_partial)
   - Compute residuals after regressing out GS_s from both sim_s and lc_s.
   - Permutation test via phase-randomization on LC residuals.

Outputs (in --out_dir)
----------------------
- run_level_LC_map_corr.tsv
- run_level_map_gs_stats.tsv
- run_level_LC_map_partialcorr_GS.tsv

No plots. No Similarity/ cache. No Summary/.
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import nibabel as nib
from scipy.stats import pearsonr, zscore, ttest_1samp


# ==========================
# ====== HELPERS ===========
# ==========================

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x
    k = np.ones(int(w), dtype=float) / float(w)
    return np.convolve(x, k, mode="same")


def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomize a 1D signal (preserves amplitude spectrum)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    X = np.fft.rfft(x)
    amps = np.abs(X)
    phases = np.angle(X)

    rand_phases = rng.uniform(0, 2 * np.pi, size=phases.size)
    rand_phases[0] = phases[0]  # preserve DC
    if n % 2 == 0 and phases.size > 1:  # preserve Nyquist for even length
        rand_phases[-1] = phases[-1]

    Xs = amps * np.exp(1j * rand_phases)
    return np.fft.irfft(Xs, n=n)


def compute_global_signal(data_2d: np.ndarray) -> np.ndarray:
    """data_2d: (n_vox_in_mask, n_vols)."""
    return np.nanmean(data_2d, axis=0)


def compute_similarity_timecourse(
    map_idx: int, maps_4d: np.ndarray, mask_3d: np.ndarray, data_2d: np.ndarray
) -> np.ndarray:
    """Match original: zscore map across voxels; zscore each voxel across time."""
    map_vec = maps_4d[..., map_idx][mask_3d]
    if np.all(map_vec == 0) or map_vec.size < 3:
        return np.zeros(data_2d.shape[1], dtype=float)

    # z-score each voxel across time (vox, vols)
    X = data_2d.astype(float)
    m = np.mean(X, axis=1, keepdims=True)
    s = np.std(X, axis=1, keepdims=True)
    s[s == 0] = 1.0
    Xz = (X - m) / s

    nvox = map_vec.size
    return (Xz.T @ map_vec) / float(max(nvox - 1, 1))


def linear_r2(y: np.ndarray, x: np.ndarray) -> float:
    """R^2 of y explained by x using linear regression with intercept."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y = y[ok]
    x = x[ok]
    if y.size < 5:
        return np.nan

    A = np.column_stack([x, np.ones_like(x)])
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1.0 - (ss_res / ss_tot))


def signed_corr(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y = y[ok]
    x = x[ok]
    if y.size < 5:
        return np.nan
    r, _ = pearsonr(y, x)
    return float(r)


def regress_out(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residuals of y after regressing out x (with intercept)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)

    resid = np.full_like(y, np.nan, dtype=float)
    if np.sum(ok) < 5:
        return resid

    A = np.column_stack([x[ok], np.ones(np.sum(ok))])
    beta, _, _, _ = np.linalg.lstsq(A, y[ok], rcond=None)
    yhat = A @ beta
    resid[ok] = y[ok] - yhat
    return resid


def corr_with_phase_perm(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_perm: int = 1000,
) -> tuple[float, float]:
    """Pearson r with phase-randomization p-value (two-sided). Randomizes y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.size < 5:
        return np.nan, 1.0

    r0, _ = pearsonr(x, y)
    null = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        y_surr = phase_randomize(y, rng)
        null[i], _ = pearsonr(x, y_surr)

    p = (1.0 + np.sum(np.abs(null) >= abs(r0))) / float(n_perm + 1)
    return float(r0), float(p)


def infer_base_from_filename(fname: str) -> str:
    """Keep first 2 underscore-separated fields (same convention as original)."""
    parts = fname.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return os.path.splitext(fname)[0].replace(".nii", "")


def save_tsv(path: str, header: list[str], rows: list[list[object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


# ==========================
# ==== GROUP SUMMARY HELPERS
# ==========================

def fisher_mean_p(rvec: np.ndarray) -> tuple[float, float]:
    """Return (mean_r, group_p) where group_p is a 1-sample t-test on Fisher-z vs 0."""
    rvec = np.asarray(rvec, dtype=float)
    rvec = rvec[np.isfinite(rvec)]
    if rvec.size < 2:
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(rvec, -0.999999, 0.999999))
    _, pval = ttest_1samp(z, 0.0)
    return float(np.mean(rvec)), float(pval)


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


# ==========================
# ========= MAIN ===========
# ==========================

def main() -> None:
    ap = argparse.ArgumentParser(description="QC controls: LC–Map coupling vs Global Signal (GS).")
    ap.add_argument("--fmri_dir", required=True, help="Folder with 4D fMRI NIfTI files (.nii.gz).")
    ap.add_argument("--burst_dir", required=True, help="Folder with <base>_burst_indices.txt files (used as run inclusion filter).")
    ap.add_argument("--maps_path", required=True, help="4D NIfTI containing SVD maps.")
    ap.add_argument("--lc_dir", required=True, help="Folder with LC photometry time series per run as <base>.txt.")
    ap.add_argument("--out_dir", default="outputs/qc_fig2c", help="Output directory (created).")
    ap.add_argument("--tr", type=float, default=1.2, help="TR in seconds.")
    ap.add_argument("--smooth_sec", type=float, default=8.0, help="Smoothing window (moving average) in seconds.")
    ap.add_argument("--n_perm", type=int, default=1000, help="Number of phase-randomization permutations.")
    ap.add_argument("--seed", type=int, default=12345, help="RNG seed for permutations.")
    ap.add_argument("--n_maps", type=int, default=2, help="How many maps (starting from Map1) to analyse. Default 2.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    smooth_win = max(1, int(round(args.smooth_sec / args.tr)))

    # Load maps + mask
    maps_img = nib.load(args.maps_path)
    maps = maps_img.get_fdata()
    if maps.ndim != 4:
        raise RuntimeError("maps_path must be a 4D NIfTI with maps in last dimension.")
    n_maps_total = maps.shape[3]
    n_maps = int(min(max(args.n_maps, 1), n_maps_total))
    if n_maps < 1:
        raise RuntimeError("Need at least 1 map.")

    mask = np.any(maps != 0, axis=3)

    fmri_files = sorted([f for f in os.listdir(args.fmri_dir) if f.endswith(".nii.gz")])
    if len(fmri_files) == 0:
        raise RuntimeError(f"No .nii.gz files found in {args.fmri_dir}")

    corr_rows: list[dict] = []
    gs_rows: list[dict] = []
    partial_rows: list[dict] = []
    n_runs_used = 0

    for fmri_file in fmri_files:
        base = infer_base_from_filename(fmri_file)
        burst_path = os.path.join(args.burst_dir, f"{base}_burst_indices.txt")
        lc_path = os.path.join(args.lc_dir, f"{base}.txt")

        # Match original behavior: require burst file AND LC file for inclusion
        if not os.path.exists(burst_path):
            continue
        if not os.path.exists(lc_path):
            continue

        fmri_4d = nib.load(os.path.join(args.fmri_dir, fmri_file)).get_fdata()
        if fmri_4d.ndim != 4:
            continue
        n_vols = fmri_4d.shape[3]

        data_2d = fmri_4d.reshape(-1, n_vols)[mask.flatten(), :]

        # GS_raw: z-scored across time, NO smoothing (matches original for Map~GS)
        gs_raw = zscore(compute_global_signal(data_2d))

        # LC: load and resize if needed
        lc = np.squeeze(np.loadtxt(lc_path)).astype(float).reshape(-1)
        if lc.size != n_vols:
            lc = np.resize(lc, n_vols)

        # LC_s for correlations (matches original)
        lc_s = zscore(moving_average(lc, smooth_win))

        # GS_s for partial correlation only (matches original)
        gs_s = zscore(moving_average(gs_raw, smooth_win))

        # Similarity per map (UNsmoothed raw similarity)
        sims = []
        for mi in range(n_maps):
            sim = compute_similarity_timecourse(mi, maps, mask, data_2d)
            sims.append(sim)

        # Run-level stats per map
        for mi in range(n_maps):
            sim = sims[mi]

            # (A) LC–Map zero-lag corr: smooth+zscore(sim) vs smooth+zscore(lc)
            sim_s = zscore(moving_average(sim, smooth_win))
            r0, p0 = corr_with_phase_perm(sim_s, lc_s, rng, n_perm=args.n_perm)
            corr_rows.append({"base": base, "map": mi + 1, "r": r0, "p_perm": p0})

            # (C) Map~GS: use UNSMOOTHED sim and UNSMOOTHED gs_raw (as in original)
            r2 = linear_r2(sim, gs_raw)
            rgs = signed_corr(sim, gs_raw)
            gs_rows.append({"base": base, "map": mi + 1, "r2_map_gs": r2, "r_map_gs": rgs})

            # (B) Partial corr LC–Map | GS: regress out GS_s from sim_s and lc_s
            sim_res = regress_out(sim_s, gs_s)
            lc_res = regress_out(lc_s, gs_s)
            r_part, p_part = corr_with_phase_perm(sim_res, lc_res, rng, n_perm=args.n_perm)
            partial_rows.append({"base": base, "map": mi + 1, "r_partial": r_part, "p_perm_partial": p_part})

        n_runs_used += 1

    if n_runs_used == 0:
        raise RuntimeError("No runs processed. Check naming, burst files, LC files, and paths.")

    out_corr = os.path.join(args.out_dir, "run_level_LC_map_corr.tsv")
    out_gs = os.path.join(args.out_dir, "run_level_map_gs_stats.tsv")
    out_part = os.path.join(args.out_dir, "run_level_LC_map_partialcorr_GS.tsv")

    save_tsv(out_corr, ["base", "map", "r", "p_perm"],
             [[r["base"], r["map"], r["r"], r["p_perm"]] for r in corr_rows])
    save_tsv(out_gs, ["base", "map", "r2_map_gs", "r_map_gs"],
             [[r["base"], r["map"], r["r2_map_gs"], r["r_map_gs"]] for r in gs_rows])
    save_tsv(out_part, ["base", "map", "r_partial", "p_perm_partial"],
             [[r["base"], r["map"], r["r_partial"], r["p_perm_partial"]] for r in partial_rows])


    print("\n====================================================")
    print("SUMMARY: LC coupling to SVD map similarity")
    print("  (A) LC–Map zero-lag corr  : r, p_perm")
    print("  (B) LC–Map partial | GS   : r_partial, p_perm_partial")
    print("  (C) Map ~ GS shared var   : R^2")
    print("====================================================\n")

    for m in range(1, n_maps + 1):
        # --- Original LC–Map ---
        r_m = np.array([r["r"] for r in corr_rows if r["map"] == m], dtype=float)
        p_m = np.array([r["p_perm"] for r in corr_rows if r["map"] == m], dtype=float)
        good = np.isfinite(r_m) & np.isfinite(p_m)
        r_m = r_m[good]
        p_m = p_m[good]

        # --- Partial LC–Map | GS ---
        pr_m = np.array([r["r_partial"] for r in partial_rows if r["map"] == m], dtype=float)
        pp_m = np.array([r["p_perm_partial"] for r in partial_rows if r["map"] == m], dtype=float)
        goodp = np.isfinite(pr_m) & np.isfinite(pp_m)
        pr_m = pr_m[goodp]
        pp_m = pp_m[goodp]

        # --- Map ~ GS ---
        r2_m = np.array([r["r2_map_gs"] for r in gs_rows if r["map"] == m], dtype=float)
        rgs_m = np.array([r["r_map_gs"] for r in gs_rows if r["map"] == m], dtype=float)
        r2_m = r2_m[np.isfinite(r2_m)]
        rgs_m = rgs_m[np.isfinite(rgs_m)]

        mean_r, p_group = fisher_mean_p(r_m)
        mean_pr, p_group_partial = fisher_mean_p(pr_m)

        sem_r = sem(r_m)
        sem_pr = sem(pr_m)
        mean_r2 = float(np.mean(r2_m)) if r2_m.size else np.nan
        sem_r2 = sem(r2_m)

        med_r = float(np.median(r_m)) if r_m.size else np.nan
        med_pr = float(np.median(pr_m)) if pr_m.size else np.nan
        med_r2 = float(np.median(r2_m)) if r2_m.size else np.nan

        frac_sig = float(np.mean(p_m < 0.05)) if p_m.size else np.nan
        frac_sig_partial = float(np.mean(pp_m < 0.05)) if pp_m.size else np.nan

        mean_rgs = float(np.mean(rgs_m)) if rgs_m.size else np.nan
        med_rgs = float(np.median(rgs_m)) if rgs_m.size else np.nan

        print(f"MAP {m}")
        print(f"  Runs analysed: {r_m.size}")
        print(f"  LC–Map:      mean r = {mean_r:.3f} ± {sem_r:.3f} | median r = {med_r:.3f} | group p = {p_group:.3e}")
        print(f"              % runs p_perm<0.05: {100*frac_sig:.1f}%")
        print(f"  LC–Map|GS:   mean r = {mean_pr:.3f} ± {sem_pr:.3f} | median r = {med_pr:.3f} | group p = {p_group_partial:.3e}")
        print(f"              % runs p_perm_partial<0.05: {100*frac_sig_partial:.1f}%")
        print(f"  Map~GS:      mean R^2 = {mean_r2:.3f} ± {sem_r2:.3f} | median R^2 = {med_r2:.3f}")
        print(f"              signed corr(Map,GS): mean r = {mean_rgs:.3f} | median r = {med_rgs:.3f}  (sign shown; R^2 ignores sign)")

        # Interpretation (same logic as original script)
        if np.isfinite(mean_r) and np.isfinite(mean_pr) and np.isfinite(frac_sig_partial):
            if (mean_r > 0.3) and (mean_pr > 0.3) and (frac_sig_partial >= 0.5):
                interp = "LC coupling survives GS removal (not driven by global signal)."
            elif (mean_r > 0.3) and (mean_pr < 0.15):
                interp = "LC coupling largely disappears after GS removal (likely global/confounded)."
            elif (mean_r < 0.2) and (mean_pr > 0.3):
                interp = "LC coupling becomes clearer after GS removal (GS was masking the relationship)."
            else:
                interp = "Mixed pattern across runs; inspect run-level tables."
        else:
            interp = "Not enough data to interpret."

        print(f"  Interpretation: {interp}\n")

    print("Saved run-level tables:")
    print(f"  - {out_corr}")
    print(f"  - {out_part}")
    print(f"  - {out_gs}")
    print("====================================================\n")


if __name__ == "__main__":
    main()
