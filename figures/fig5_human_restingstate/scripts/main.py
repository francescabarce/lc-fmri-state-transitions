"""Fig. 5 (Human resting-state) — LC-peak–aligned similarity analysis

This script loads a concatenated similarity time series (e.g., MapLC1/MapLC2 similarity values)
and a list of LC peak indices, then:
  1) extracts event-aligned windows around each LC peak (per subject)
  2) plots the mean ± SEM similarity traces
  3) runs an event-wise AUC(|z|) comparison against a subject-specific random-window null model

Inputs are provided at runtime via CLI arguments,

Example usage:
    python main.py \
      --similarity_csv /path/to/Awake_HUMAN_RS_MAP1_MAP2_regressionResult.csv \
      --lc_peaks /path/to/LC_peaks.txt \
      --output_dir ./outputs \
      --params_json /path/to/params.json

Minimal required inputs:
  - similarity_csv: CSV/TSV with shape (T_total, n_maps)
  - lc_peaks: text file with integer indices (global indices in the concatenated time series)

"""
# main_fig5_from_similarity.py
import os
import json
import argparse
import numpy as np
import pandas as pd

from peaks import analyze_peaks
from plotting import plot_event_similarity
from eventwise_auc_human import eventwise_auc_test_human


def main():
    # -------------------------
    # Runtime inputs (CLI)
    # -------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="LC-peak–aligned similarity analysis (Human RS): mean±SEM plots + event-wise AUC null test."
    )
    parser.add_argument("--similarity_csv", type=str, required=True,
                        help="Path to CSV/TSV containing similarity time series (T_total x n_maps).")
    parser.add_argument("--lc_peaks", type=str, required=True,
                        help="Path to .txt file with integer LC peak indices (global indices in concatenated series).")
    parser.add_argument("--output_dir", type=str, default=os.path.join(BASE_DIR, "outputs"),
                        help="Output directory (default: ./outputs next to this script).")
    parser.add_argument("--params_json", type=str, default=None,
                        help="Optional params.json to override defaults (e.g., tr, window, baseline, n_random).")

    args = parser.parse_args()

    similarity_csv = args.similarity_csv
    lc_peaks_path = args.lc_peaks
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # If not provided, look for params.json next to the similarity file
    params_json = args.params_json
    if params_json is None:
        params_json = os.path.join(os.path.dirname(os.path.abspath(similarity_csv)), "params.json")

    # -------------------------
    # Default params (your values)
    # -------------------------
    params = dict(
        tr=0.58,
        tr_per_subject=1050,
        window_sec_similarity=30,
        baseline_win=[-40, -15],
        t0=-10,
        t1=15,
        n_random=5000,
    )
    # Parameter precedence: defaults < params.json (if found) < CLI (explicit arguments above)

    if os.path.exists(params_json):
        with open(params_json, "r") as f:
            loaded = json.load(f)
        params.update(loaded)

    tr = float(params["tr"])
    tr_per_subject = int(params["tr_per_subject"])
    window_sec_similarity = float(params["window_sec_similarity"])
    baseline_win = tuple(params["baseline_win"])
    t0 = float(params["t0"])
    t1 = float(params["t1"])
    n_random = int(params["n_random"])

    # -------------------------
    # Load inputs
    # -------------------------
    if not os.path.exists(similarity_csv):
        raise FileNotFoundError(f"Missing: {similarity_csv}")
    if not os.path.exists(lc_peaks_path):
        raise FileNotFoundError(f"Missing: {lc_peaks_path}")

    df = pd.read_csv(similarity_csv, sep=None, engine="python")  # handles tab or comma
    sims_concat = df.values.astype(float)
    map_names = list(df.columns)

    peaks = np.loadtxt(lc_peaks_path, dtype=int)
    if peaks.ndim == 0:
        peaks = np.array([int(peaks)])

    print("=== FIG5 — from similarity (LC only) ===")
    print(f"Similarity: {similarity_csv}")
    print(f"Shape: {sims_concat.shape} (T_total, n_maps)")
    print(f"Maps: {map_names}")
    print(f"LC peaks: {lc_peaks_path} (n={peaks.size})")
    print(f"tr={tr}, tr_per_subject={tr_per_subject}, window_sec={window_sec_similarity}")
    print("=======================================")

    # -------------------------
    # Analyze peaks (LC only)
    # -------------------------
    label = "LC"
    time_axis, mean_all, sem_all, segments_all, subject_ids_all = analyze_peaks(
        sims_concat=sims_concat,
        peaks=peaks,
        tr=tr,
        tr_per_subject=tr_per_subject,
        window_sec=window_sec_similarity,
        label=label,
        output_folder=OUTPUT_DIR,
    )

    # Save plot inputs for reviewers
    np.save(os.path.join(OUTPUT_DIR, "time_axis_LC.npy"), time_axis)
    np.save(os.path.join(OUTPUT_DIR, "mean_LC.npy"), mean_all)
    np.save(os.path.join(OUTPUT_DIR, "sem_LC.npy"), sem_all)

    # -------------------------
    # Plot
    # -------------------------
    plot_event_similarity(
        time_axis=time_axis,
        mean_trace=mean_all,
        sem_trace=sem_all,
        labels=map_names,
        title="Similarity aligned to LC peaks",
        outfile=os.path.join(OUTPUT_DIR, "similarity_LC.png"),
    )
    print("✓ Saved: outputs/similarity_LC.png")

    # -------------------------
    # Event-wise AUC (LC only)
    # -------------------------
    eventwise_auc_test_human(
        segments_all=segments_all,
        time_axis=time_axis,
        sims_concat=sims_concat,
        tr=tr,
        tr_per_subject=tr_per_subject,
        peak_label=label,
        output_folder=OUTPUT_DIR,
        baseline_win=baseline_win,
        t0=t0,
        t1=t1,
        n_random=n_random,
    )

    print("✓ Done.")


if __name__ == "__main__":
    main()