# main_fig5_fc_states_only.py
import os
import json
import numpy as np
import sys
from .fc_states_only import run_fc_states_only_pipeline


def main():
    # -------------------------
    # Paths (repo-relative)
    # -------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #INPUT_DIR = os.path.join(BASE_DIR, "inputs")
    INPUT_DIR = '/Users/barcelin/Documents/Codes/paper_figures/data/Awake_HUMAN_RS/FunctionalConnectivity'
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fc_strength_path = os.path.join(INPUT_DIR, "FC_strength_fisherz.npy")
    states_npz_path = os.path.join(INPUT_DIR, "template_states.npz")
    fc_mats_path = os.path.join(INPUT_DIR, "FC_matrices_fisherz.npy")  # optional
    params_json = os.path.join(INPUT_DIR, "params_fc_states_only.json")  # optional

    # -------------------------
    # Default params
    # -------------------------
    params = dict(
        tr_sec=0.58,
        strength_method="subspace_rms",   # "subspace_rms" or "subspace_energy"
        n_boot=5000,
        seed=0,
        panelA_subjects=None,  # e.g. [0, 10] or leave None for auto (min/max strength)
        out_prefix="Fig5_FC_states_only",
    )

    if os.path.exists(params_json):
        with open(params_json, "r") as f:
            loaded = json.load(f)
        params.update(loaded)

    # -------------------------
    # Load inputs
    # -------------------------
    if not os.path.exists(fc_strength_path):
        raise FileNotFoundError(f"Missing: {fc_strength_path}")
    if not os.path.exists(states_npz_path):
        raise FileNotFoundError(f"Missing: {states_npz_path}")

    fc_strength = np.load(fc_strength_path).astype(float).ravel()

    st = np.load(states_npz_path)
    if "t1_state" not in st or "t2_state" not in st:
        raise KeyError("template_states.npz must contain keys: 't1_state' and 't2_state'")

    t1_state = np.asarray(st["t1_state"], dtype=float)
    t2_state = np.asarray(st["t2_state"], dtype=float)

    # optional FC matrices
    fc_mats = None
    if os.path.exists(fc_mats_path):
        fc_mats = np.load(fc_mats_path).astype(float)

    print("=== FIG5 — FC states-only pipeline ===")
    print(f"FC strength: {fc_strength_path} | shape={fc_strength.shape}")
    print(f"States NPZ:  {states_npz_path} | t1={t1_state.shape}, t2={t2_state.shape}")
    if fc_mats is not None:
        print(f"FC mats:    {fc_mats_path} | shape={fc_mats.shape} (optional)")
    print("======================================")

    results = run_fc_states_only_pipeline(
        t1_state=t1_state,
        t2_state=t2_state,
        fc_strength=fc_strength,
        output_folder=OUTPUT_DIR,
        tr_sec=float(params["tr_sec"]),
        strength_method=str(params["strength_method"]),
        n_boot=int(params["n_boot"]),
        seed=int(params["seed"]),
        panelA_subjects=params["panelA_subjects"],
        out_prefix=str(params["out_prefix"]),
    )

    print("\n✓ Done. Key results:")
    for k in ("r", "p", "strength_method", "fname_tag"):
        if k in results:
            print(f"  {k}: {results[k]}")


if __name__ == "__main__":
    main()