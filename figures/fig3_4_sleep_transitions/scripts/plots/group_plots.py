import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..config import RESULTS_DIR, TRANSITION_PATTERNS


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    """Z-score a 1D array safely (ignores NaNs)."""
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (x - mu) / sd


def plot_all_group_level(
    all_subject_epochs: dict,
    output_dir: str = RESULTS_DIR,
    patterns: Optional[Sequence[Tuple[int, int]]] = None,
    map_names: Optional[Sequence[str]] = None,
    title_prefix: str = "MOUSE",
    per_event_zscore: bool = True,
    tr_s: Optional[float] = None,
) -> None:
    """Paper-ready group-level plot of LC-template similarity around transitions.

    Assumes `all_subject_epochs` format:
        dict[(a,b)] -> list of subject arrays
    where each subject array has shape:
        (n_events, T, n_maps)
    """

    os.makedirs(output_dir, exist_ok=True)

    if patterns is None:
        patterns = TRANSITION_PATTERNS

    if tr_s is None:
        raise ValueError(
            "plot_all_group_level requires tr_s (seconds). Pass the TR read from the combined NPZ."
        )

    # Ensure SVG text stays editable
    plt.rcParams["svg.fonttype"] = "none"

    # Infer n_maps from first available data
    n_maps: Optional[int] = None
    for pat in patterns:
        subj_list = all_subject_epochs.get(pat, [])
        if not subj_list:
            continue
        example = subj_list[0]
        if isinstance(example, np.ndarray) and example.ndim == 3 and example.shape[-1] > 0:
            n_maps = int(example.shape[-1])
            break

    if n_maps is None:
        print("[plot_all_group_level] No data found in all_subject_epochs.")
        return

    if map_names is None:
        map_names = [f"Map{i+1}" for i in range(n_maps)]

    cmap = plt.cm.get_cmap("tab10", n_maps)

    for pat in patterns:
        subjects_list = all_subject_epochs.get(pat, [])
        if not subjects_list:
            print(f"[plot_all_group_level] No subjects for transition {pat[0]}→{pat[1]}.")
            continue

        # Concatenate all subjects’ epochs -> (n_events_total, T, n_maps)
        concat_epochs = np.concatenate(subjects_list, axis=0)
        if concat_epochs.ndim != 3 or concat_epochs.shape[-1] != n_maps:
            print(f"[plot_all_group_level] Unexpected epoch shape for {pat}: {concat_epochs.shape}")
            continue

        n_events, T_local, n_maps_local = concat_epochs.shape

        # Time axis based on actual epoch length
        half = T_local // 2
        time_axis = np.arange(-half, half + 1) * float(tr_s)

        # Handle edge case where T_local is even (shouldn't happen, but keep robust)
        if time_axis.shape[0] != T_local:
            time_axis = np.linspace(-half * float(tr_s), half * float(tr_s), T_local)

        # Optionally z-score per-event, per-map (recommended)
        if per_event_zscore:
            epochs_for_stats = np.empty_like(concat_epochs, dtype=float)
            for e in range(n_events):
                for m in range(n_maps_local):
                    epochs_for_stats[e, :, m] = _zscore_1d(concat_epochs[e, :, m])
        else:
            epochs_for_stats = concat_epochs.astype(float)

        mean_ts = np.nanmean(epochs_for_stats, axis=0)  # (T, n_maps)
        sem_ts = np.nanstd(epochs_for_stats, axis=0, ddof=1) / np.sqrt(n_events)

        fig, ax = plt.subplots(figsize=(10, 5))

        for m in range(n_maps_local):
            label = map_names[m] if m < len(map_names) else f"Map{m+1}"
            ax.plot(time_axis, mean_ts[:, m], lw=2.5, color=cmap(m), label=label)
            ax.fill_between(
                time_axis,
                mean_ts[:, m] - sem_ts[:, m],
                mean_ts[:, m] + sem_ts[:, m],
                alpha=0.25,
                color=cmap(m),
                linewidth=0.0,
            )

        ax.axvline(0, color="black", linestyle="--", linewidth=1)

        transition_label = f"{pat[0]}→{pat[1]}"
        ax.set_title(f"{title_prefix} – {transition_label}  (n={n_events} epochs)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(
            "Similarity (z-scored per event, per map)" if per_event_zscore else "Similarity"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=11)
        ax.legend(frameon=False)

        fig.tight_layout()

        fname = f"{title_prefix}_{pat[0]}to{pat[1]}_combinedmaps.svg"
        outpath = os.path.join(output_dir, fname)
        fig.savefig(outpath)
        plt.close(fig)

        print(f"✓ saved {outpath}")