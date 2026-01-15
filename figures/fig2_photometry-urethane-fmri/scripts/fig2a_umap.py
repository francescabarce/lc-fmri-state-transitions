#!/usr/bin/env python3
"""
fig2a_umap.py — UMAP embedding of LC burst–triggered brain dynamics
------------------------------------------------------------------

This script collects ROI segments aligned to LC burst peaks,
computes the average trajectory, fits a UMAP embedding (0→70 s),
and saves a single paper-ready UMAP figure.

Fixed settings (paper):
    - TR = 1.2 s
    - window: 0 → +70 s post-peak
    - UMAP: n_neighbors=3, min_dist=0.1, metric='euclidean'

Example
-------
python scripts/fig2a_umap.py \
  --roi_dir /path/to/roi_txt_folder \
  --burst_dir /path/to/LC_burst_outputs_full_dataset \
  --out_dir /path/to/output_folder
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import umap
from matplotlib.collections import LineCollection

# ----------------------------
# === PAPER PARAMS (fixed) ===
# ----------------------------
TR = 1.2
WINDOW_SEC = 70
WINDOW_SAMPLES = int(WINDOW_SEC / TR)  # 70 / 1.2 ≈ 58 vol + 1
SAMPLE_EVERY_SEC = 3
SAMPLE_EVERY = int(SAMPLE_EVERY_SEC / TR)

# Matches the original “reducer_hi” settings
UMAP_PARAMS = dict(
    n_neighbors=3,
    min_dist=0.10,
    metric="euclidean",
    random_state=42,
)

# ----------------------------
# === INPUT / OUTPUT (CLI) ===
# ----------------------------
PEAK_SUFFIX = "_burst_indices.txt"  # suffix for burst index files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fig2a — UMAP embedding of LC burst–triggered brain dynamics (0→70 s)"
    )
    p.add_argument(
        "--roi_dir",
        required=True,
        help="Folder containing ROI time series TXT files named '<base>_EPI_HP_ROI.txt'",
    )
    p.add_argument(
        "--burst_dir",
        required=True,
        help=f"Folder containing burst index files named '<base>{PEAK_SUFFIX}'",
    )
    p.add_argument(
        "--out_dir",
        required=True,
        help="Output directory where the UMAP figure will be saved",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    roi_base_folder = args.roi_dir
    peak_folder = args.burst_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------
    # === LOAD & COLLECT SEGMENTS ===
    # ----------------------------
    print("\n=== Collecting burst-aligned ROI segments ===")
    print("Peak edge filtering based on ROI length")

    all_segments: list[np.ndarray] = []

    peak_files = sorted(
        f for f in os.listdir(peak_folder)
        if f.endswith(PEAK_SUFFIX) and not f.startswith('.')
    )

    for filename in peak_files:
        base_id = filename.replace(PEAK_SUFFIX, "")
        peak_path = os.path.join(peak_folder, filename)

        roi_txt = os.path.join(roi_base_folder, base_id + "_EPI_HP_ROI.txt")
        if not os.path.exists(roi_txt):
            print(f"[WARN] ROI file missing for {base_id}; skipping")
            continue

        roi_data = np.loadtxt(roi_txt)
        if roi_data.ndim != 2:
            print(f"[WARN] Unexpected ROI shape for {base_id}; skipping")
            continue

        # z-score per ROI
        stds = roi_data.std(axis=0)
        stds[stds == 0] = 1
        roi_data = (roi_data - roi_data.mean(axis=0)) / stds

        # Peak edge filtering is done against ROI length
        signal_len = roi_data.shape[0]

        # Load burst indices robustly:
        # - avoids NumPy int-from-float deprecation
        # - handles single-value files (0-d arrays)
        raw_peaks = np.loadtxt(peak_path, dtype=float)
        raw_peaks = np.atleast_1d(raw_peaks).astype(np.int64)

        valid_peaks = [
            int(p)
            for p in raw_peaks
            if 0 <= int(p) and (int(p) + WINDOW_SAMPLES) < signal_len
        ]

        for p in valid_peaks:
            seg = roi_data[p : p + WINDOW_SAMPLES + 1, :]
            all_segments.append(seg)

    print("Total events collected:", len(all_segments))
    if len(all_segments) == 0:
        raise RuntimeError("No valid burst segments found!")

    # stack → (n_events, time, n_rois)
    segments_matrix = np.stack(all_segments, axis=0)

    # --- Robust NaN handling (matches original working pipeline) ---
    # Remove ROIs that contain NaN in ANY event OR ANY timepoint
    roi_nan = np.isnan(segments_matrix).any(axis=(0, 1))
    if roi_nan.any():
        print("Removing NaN ROIs:", np.where(roi_nan)[0])
    segments_matrix = segments_matrix[:, :, ~roi_nan]

    # Drop events that still contain NaN (should be rare after ROI filtering)
    event_nan = np.isnan(segments_matrix).any(axis=(1, 2))
    if event_nan.any():
        print(f"Removing {event_nan.sum()} events containing NaN after ROI filtering")
    segments_matrix = segments_matrix[~event_nan]

    if segments_matrix.shape[0] == 0:
        raise RuntimeError("No valid burst segments remain after NaN filtering.")

    # compute average (time × ROI) after filtering
    X_mean = np.nanmean(segments_matrix, axis=0)

    # ----------------------------
    # === UMAP (paper settings) ===
    # ----------------------------
    print("\n=== Fitting UMAP (paper parameters) ===")

    X_mean_sub = X_mean[::SAMPLE_EVERY, :]

    reducer = umap.UMAP(**UMAP_PARAMS)
    reducer.fit(X_mean_sub)
    embedding = reducer.transform(X_mean_sub)

    # project all event trajectories
    projected: list[np.ndarray] = []
    for e in range(segments_matrix.shape[0]):
        traj = segments_matrix[e, :, :][::SAMPLE_EVERY, :]
        projected.append(reducer.transform(traj))

    # ---------------------------------
    # === SINGLE PAPER-READY FIGURE ===
    # ---------------------------------
    print("\n=== Producing UMAP figure ===")

    # Deterministic jitter to match the original look, but reproducible
    rng = np.random.default_rng(42)
    JITTER_STD = 0.12

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    # Background: all event trajectories (jittered points)
    for traj in projected:
        t = np.linspace(0, WINDOW_SEC, traj.shape[0])
        xj = traj[:, 0] + rng.normal(0.0, JITTER_STD, size=traj.shape[0])
        yj = traj[:, 1] + rng.normal(0.0, JITTER_STD, size=traj.shape[0])
        ax.scatter(
            xj,
            yj,
            c=t,
            cmap="plasma",
            alpha=0.30,
            s=6,
            linewidths=0.0,
            zorder=1,
        )

    # Mean trajectory: colored line + time-colored points (as in the original)
    segs = np.array([[embedding[i], embedding[i + 1]] for i in range(len(embedding) - 1)])
    lc = LineCollection(
        segs,
        cmap="plasma",
        norm=plt.Normalize(vmin=0, vmax=WINDOW_SEC),
        linewidths=4.0,
        zorder=30,
    )
    lc.set_array(np.linspace(0, WINDOW_SEC, len(embedding) - 1))
    ax.add_collection(lc)

    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=np.linspace(0, WINDOW_SEC, len(embedding)),
        cmap="plasma",
        s=28,
        edgecolor="none",
        linewidth=0.0,
        zorder=40,
    )

    # Start/end markers
    ax.scatter(*embedding[0], c="green", s=45, edgecolors="k", linewidths=1.2, zorder=50)
    ax.scatter(*embedding[-1], c="red", s=55, marker="s", edgecolors="k", linewidths=1.2, zorder=50)

    sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=WINDOW_SEC))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Time from burst (s)")

    ax.set_title("UMAP — LC-triggered brain dynamics (0 → +70 s)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    fig.tight_layout()

    outpath = os.path.join(out_dir, "UMAP_Fig2_paper.svg")
    fig.savefig(outpath)
    plt.close(fig)

    print(f"UMAP figure saved: {outpath}")


if __name__ == "__main__":
    main()