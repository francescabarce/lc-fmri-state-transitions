import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (x - mu) / sd


def plot_group_subject_heatmaps(
    subjects,
    all_subject_epochs,
    output_dir,
    win_trs,
    tr,
    patterns=None,
    vlim=1.3,
):
    """
    Heatmap per transition e per map:
      rows = subjects
      cols = time
      value = mean across events of per-event z-scored similarity (per subject)
    """
    if patterns is None:
        patterns = list(all_subject_epochs.keys())

    outdir = os.path.join(output_dir, "subject_heatmaps_group")
    os.makedirs(outdir, exist_ok=True)

    tvec = np.arange(-win_trs, win_trs + 1) * tr

    # infer n_maps
    n_maps = None
    for pat in patterns:
        subj_list = all_subject_epochs.get(pat, [])
        if len(subj_list) == 0:
            continue
        n_maps = subj_list[0].shape[-1]
        break
    if n_maps is None:
        print("[heatmaps] No data.")
        return

    # Build per-subject matrices
    # For each transition: mat shape (n_subjects, n_time, n_maps)
    for pat in patterns:
        subj_epochs_list = all_subject_epochs.get(pat, [])
        if len(subj_epochs_list) == 0:
            continue

        # Align to subjects order: assumes build_epochs_from_combined_npz returned subjects
        # and all_subject_epochs[pat] is a list of subject arrays in the same order.
        # If your builder skips subjects with 0 events, we handle that below.
        per_subject = []
        subj_names_used = []

        for subj_name, arr in zip(subjects, subj_epochs_list):
            if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[0] == 0:
                continue

            # arr: (n_events, T, n_maps)
            n_events, T, n_maps_local = arr.shape
            subj_mean = np.zeros((T, n_maps_local), dtype=float)

            for m in range(n_maps_local):
                z_events = np.stack([_zscore_1d(arr[e, :, m]) for e in range(n_events)], axis=0)
                subj_mean[:, m] = np.nanmean(z_events, axis=0)

            per_subject.append(subj_mean)  # (T, n_maps)
            subj_names_used.append(subj_name)

        if len(per_subject) == 0:
            continue

        per_subject = np.stack(per_subject, axis=0)  # (n_subj_used, T, n_maps)

        for m in range(n_maps):
            mat = per_subject[:, :, m]  # (subjects, time)

            plt.figure(figsize=(12, 4))
            plt.imshow(
                mat,
                aspect="auto",
                cmap="RdBu_r",
                extent=[tvec[0], tvec[-1], 0, mat.shape[0]],
                vmin=-vlim,
                vmax=vlim,
            )
            plt.colorbar(label="Z-scored similarity")
            plt.axvline(0, color="k", linestyle="--")
            plt.yticks(np.arange(len(subj_names_used)) + 0.5, subj_names_used, fontsize=6)
            plt.ylabel("Subjects")
            plt.xlabel("Time (s)")
            plt.title(f"{pat[0]}to{pat[1]} — Map {m+1}")

            outpath = os.path.join(outdir, f"{pat[0]}to{pat[1]}_MAP{m+1}_group_heatmap.svg")
            plt.tight_layout()
            plt.savefig(outpath)
            plt.close()
            print(f"✔ Saved {outpath}")