# main_fig5_from_similarity.py
import os
import json
import numpy as np
import pandas as pd

from peaks import analyze_peaks
from plotting import plot_event_similarity
from eventwise_auc_human import eventwise_auc_test_human


def main():
    # -------------------------
    # Paths (repo-relative)
    # -------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #BASE_DIR = "/Users/barcelin/Documents/Codes/paper_figures/fig5/OUTPUT_eventwise_auc_from_similarity"
    #INPUT_DIR = os.path.join(BASE_DIR, "inputs")
    INPUT_DIR = '/Users/barcelin/Documents/Codes/paper_figures/data/Awake_HUMAN_RS/FunctionalConnectivity'
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    similarity_csv = os.path.join(INPUT_DIR, "/Users/barcelin/Documents/Codes/paper_figures/data/Awake_HUMAN_RS/Awake_HUMAN_RS_MAP1_MAP2_regressionResult.csv")
    lc_peaks_path = os.path.join(INPUT_DIR, "/Users/barcelin/Documents/Codes/paper_figures/data/Awake_HUMAN_RS/LC_peaks.txt")

    # Optional but recommended: a tiny JSON with params (so reviewers don't edit code)
    params_json = os.path.join(INPUT_DIR, "params.json")

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