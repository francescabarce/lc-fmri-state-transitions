#!/usr/bin/env python3
"""
fig2b_ridgeline_waves.py — Ridgeline plot of LC-burst aligned ROI waveforms
--------------------------------------------------------------------------

Goal (Fig.2b):
- Plot a subset of ROI timecourses (peak-aligned) as stacked "ridgeline" curves.

Inputs
------
1) Structured dataset CSV (ROI x time), columns are time in seconds.
2) ROI list (.txt), one ROI acronym per line (must match CSV index).
3) ABI parcellation table (xlsx) to map ROI_ID -> ROI_NAME (pretty label) and MACROAREA (optional).

Outputs
-------
- A single PNG (default): ridgeline plot, paper-ready.
- SVG output is optional via --format.
  
Notes
-----
- Smoothing uses gaussian_filter1d to improve readability.

Example usage (command line)
---------------------------

python scripts/fig2b_ridgeline_waves.py \
  --input data/fMRI_urethane_timelocked_dataset.csv \
  --roi_list data/Paper_selected_roi.txt \
  --parcellation data/ABI_template_2021_Full_ROIs_corrected.xlsx \
  --tmin -40 \
  --tmax 80 \
  --out outputs/figures_png/Fig2B_ridgeline.png

This produces the Fig.2b ridgeline plot (LC-burst aligned ROI waveforms) exactly as shown in the paper, using a -40 to +80 s window.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter1d

plt.rcParams["svg.fonttype"] = "none"


# ==========================
# ======= CONFIG ===========
# ==========================
# ROI hierarchy used when ORDERING_MODE="hierarchy"
CORTICAL_HIERARCHY = [
    "RSPd","RSPv","RSPagl","ORBl","BLA","CA1","CA2","CA3","ENTl","DG",
    "MOp","SSp-bfd","SSp-ll","SSp-m","SSp-n","SSp-tr","SSp-ul","SSs",
    "TEa","PTLp","VAL","VPL","VPM"
]

# Macro-area color map (only used if FILL_MODE="anatomical")
COLOR_MAP = {
    "Isocortex": "red",
    "Olfactory Areas": "orange",
    "Hippocampal Formation": "purple",
    "Cortical Subplate": "darkred",
    "Striatum": "green",
    "Pallidum": "lime",
    "Thalamus": "pink",
    "Hypothalamus": "cyan",
    "Midbrain": "blue",
    "Pons": "steelblue",
    "Medulla": "navy",
    "Cerebellum": "brown",
}


# ==========================
# ====== HELPERS ===========
# ==========================
def load_roi_list(path: str) -> list[str]:
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_parcellation_table(path: str) -> pd.DataFrame:
    # Your ABI xlsx has no header; keep it robust
    df = pd.read_excel(path, header=None)
    # Expected columns (based on your usage):
    # 0 ROI_ID, 1 ROI_NAME (acronym), 2 FULL, 3 DESCRIPTION, 4 MACROAREA, 5 HEMISPHERE, ...
    df = df.copy()
    df.columns = ["ROI_ID", "ROI_NAME", "ROI_FULL_NAME", "DESCRIPTION", "MACROAREA", "HEMISPHERE"] + \
                 [f"col{i}" for i in range(6, df.shape[1])]
    return df


def get_curve_onset_time(trace: np.ndarray, time_axis: np.ndarray) -> float:
    """
    Onset heuristic:
    - prefer first negative→positive zero-crossing
    - else fall back to time of max absolute derivative
    """
    trace = np.asarray(trace, dtype=float)
    time_axis = np.asarray(time_axis, dtype=float)

    zc = np.where(np.diff(np.sign(trace)) > 0)[0]
    if zc.size > 0:
        return float(time_axis[zc[0]])

    diff = np.gradient(trace)
    return float(time_axis[int(np.argmax(np.abs(diff)))])


def pick_fill_color(i: int, n: int, macroarea: str | None, fill_mode: str, color_map: dict) -> tuple:
    if fill_mode == "position":
        cmap = plt.colormaps["plasma"]
        norm = Normalize(vmin=0, vmax=max(n - 1, 1))
        return cmap(norm(i))
    if fill_mode == "anatomical":
        if macroarea is None:
            return (0.6, 0.6, 0.6, 1.0)
        c = color_map.get(macroarea, "gray")
        return plt.matplotlib.colors.to_rgba(c)
    return (0.8, 0.8, 0.8, 1.0)


# ==========================
# ========= MAIN ===========
# ==========================
def main():
    ap = argparse.ArgumentParser(
        description="Fig.2b ridgeline plot of LC-burst aligned ROI waveforms (stacked curves)."
    )
    ap.add_argument("--input", required=True, help="Structured dataset CSV (ROI x time), columns are time (s).")
    ap.add_argument("--roi_list", required=True, help="Text file with ROI acronyms (one per line), must match CSV index.")
    ap.add_argument("--parcellation", required=True, help="ABI parcellation XLSX used to map ROI_ID -> ROI_NAME / MACROAREA.")
    ap.add_argument("--out", required=True, help="Output figure path (e.g., outputs/Fig2B_ridgeline.png).")
    ap.add_argument("--tmin", type=float, default=-40.0, help="Window start (s). Default: -40")
    ap.add_argument("--tmax", type=float, default=80.0, help="Window end (s). Default: 80")
    ap.add_argument("--smooth_sigma", type=float, default=1.5, help="Gaussian smoothing sigma. Default: 1.5")
    ap.add_argument("--z_offset", type=float, default=0.70, help="Vertical spacing between ridgelines. Default: 0.70")
    ap.add_argument("--label_pad", type=float, default=20.0, help="How far left to place ROI labels (seconds). Default: 20")
    ap.add_argument("--ordering", choices=["hierarchy", "onset", "input"], default="hierarchy",
                    help="ROI ordering mode. Default: hierarchy")
    ap.add_argument("--fill", choices=["position", "anatomical"], default="position",
                    help="Ridgeline fill coloring. Default: position")
    ap.add_argument("--format", choices=["png", "svg"], default=None,
                    help="Force output format. Default inferred from --out extension.")
    args = ap.parse_args()

    time_window = (args.tmin, args.tmax)
    label_x = args.tmin - args.label_pad

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load dataset ----
    df = pd.read_csv(args.input, sep=",", index_col=0)
    df.columns = df.columns.astype(float)

    roi_list = load_roi_list(args.roi_list)

    # Keep only requested ROIs (and average duplicates if any)
    df = df.loc[df.index.isin(roi_list)]
    df = df.groupby(df.index).mean()

    # ---- Load parcellation metadata (pretty names + macroareas) ----
    parc = load_parcellation_table(args.parcellation)
    parc_by_id = parc.set_index("ROI_ID", drop=False)

    def pretty_label(roi_id: str) -> str:
        if roi_id in parc_by_id.index:
            return str(parc_by_id.loc[roi_id, "ROI_NAME"])
        return roi_id

    def roi_macroarea(roi_id: str) -> str | None:
        if roi_id in parc_by_id.index:
            ma = parc_by_id.loc[roi_id, "MACROAREA"]
            if pd.isna(ma):
                return None
            return str(ma)
        return None

    # ---- Time window mask ----
    time_points = df.columns.to_numpy(dtype=float)
    mask = (time_points >= time_window[0]) & (time_points <= time_window[1])
    time_axis = time_points[mask]

    # ---- Build smoothed traces dict ----
    roi_data = {}
    for roi in df.index:
        tr = df.loc[roi].to_numpy(dtype=float).flatten()
        tr = gaussian_filter1d(tr, sigma=args.smooth_sigma)
        roi_data[roi] = tr

    # ---- Ordering ----
    if args.ordering == "hierarchy":
        ordered = [r for r in CORTICAL_HIERARCHY if r in roi_data]
        # in your original you reversed the hierarchy for plotting order
        curve_features = [(r, roi_data[r][mask]) for r in reversed(ordered)]

    elif args.ordering == "onset":
        onset_list = []
        for roi, tr in roi_data.items():
            onset = get_curve_onset_time(tr, time_points)
            onset_list.append((roi, tr[mask], onset))
        onset_list.sort(key=lambda x: x[2])
        curve_features = [(roi, tr_masked) for roi, tr_masked, _ in onset_list]

    else:  # "input"
        curve_features = [(r, roi_data[r][mask]) for r in roi_list if r in roi_data]

    if len(curve_features) == 0:
        raise RuntimeError("No ROI traces to plot (check ROI list vs CSV index).")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7.2, 10.0))
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)

    for i, (roi, trace) in enumerate(curve_features):
        y0 = i * args.z_offset
        macro = roi_macroarea(roi)
        fill = pick_fill_color(i, len(curve_features), macro, args.fill, COLOR_MAP)

        # Edge: slightly more opaque than fill
        edge = (fill[0], fill[1], fill[2], min(fill[3] + 0.25, 1.0))

        ax.plot(time_axis, trace + y0, color=edge, lw=1.4)
        ax.fill_between(time_axis, y0, trace + y0, color=fill, alpha=0.70)

        ax.text(label_x, y0, pretty_label(roi), va="center", ha="right", fontsize=8)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_xlabel("Time from burst (s)")

    fig.tight_layout()
    out_path = args.out
    fmt = args.format
    if fmt is None:
        ext = os.path.splitext(out_path)[1].lower().lstrip(".")
        fmt = ext if ext in ("png", "svg") else "png"

    save_kwargs = {}
    if fmt == "png":
        save_kwargs.update(dict(dpi=300))

    fig.savefig(out_path, format=fmt, **save_kwargs)
    plt.close(fig)

    print(f"[OK] Saved ridgeline figure: {out_path}")


if __name__ == "__main__":
    main()