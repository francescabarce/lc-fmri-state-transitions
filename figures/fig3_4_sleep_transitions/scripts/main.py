import argparse
import numpy as np

from .config import WIN_SECONDS, TRANSITION_PATTERNS, RESULTS_DIR
from .plots.group_plots import plot_all_group_level
from .plots.subject_heatmaps import plot_group_subject_heatmaps
from .stats.auc_eventwise import auc_eventwise
from .io.from_combined import build_epochs_from_combined_npz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", type=str, required=True,
                        help="Path to similarity_trace_combined.npz")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for all results (default: config RESULTS_DIR)")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--t0", type=float, default=-10)
    parser.add_argument("--t1", type=float, default=30)
    parser.add_argument("--n-rand", type=int, default=5000)
    args = parser.parse_args()

    # Read TR from the combined NPZ so this script works for both mouse (e.g., TR=2s)
    # and human (e.g., TR=3s) without editing config.
    f = np.load(args.combined, allow_pickle=True)
    tr_in_file = float(f.get("tr", np.nan))
    if not np.isfinite(tr_in_file) or tr_in_file <= 0:
        raise ValueError(f"Invalid or missing 'tr' in combined NPZ: {args.combined}")

    # Window is defined in seconds in config; convert to TRs based on the file's TR.
    win_trs = int(np.round(float(WIN_SECONDS) / tr_in_file))
    if win_trs <= 0:
        raise ValueError(f"Computed win_trs={win_trs} from WIN_SECONDS={WIN_SECONDS} and TR={tr_in_file}")

    subjects, all_subject_epochs, _ = build_epochs_from_combined_npz(
        args.combined, win_trs=win_trs
    )

    print(f"[main] Using TR from combined NPZ: {tr_in_file} s | WIN_SECONDS={WIN_SECONDS} → win_trs={win_trs}")

    output_dir = args.output_dir if args.output_dir is not None else RESULTS_DIR

    auc_eventwise(
        all_subject_epochs,
        combined_npz_path=args.combined,
        output_dir=output_dir,
        t0=args.t0,
        t1=args.t1,
        n_random=args.n_rand
    )
    print("\n✅ AUC eventwise completed.\n")

    if not args.skip_plots:
        plot_all_group_level(
            all_subject_epochs,
            output_dir=output_dir,
            patterns=TRANSITION_PATTERNS,
            title_prefix="MOUSE",
            per_event_zscore=True,
            tr_s=tr_in_file,
        )

        plot_group_subject_heatmaps(
            subjects,
            all_subject_epochs,
            output_dir=output_dir,
            win_trs=win_trs,
            tr=tr_in_file,
        )

    print("\n✅ Pipeline from combined NPZ completed.\n")


if __name__ == "__main__":
    main()