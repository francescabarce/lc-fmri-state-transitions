import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu
import pandas as pd
from typing import Optional
import seaborn as sns

plt.rcParams["svg.fonttype"] = "none"

from ..config import TRANSITION_PATTERNS


def cliff_delta(x, y):
    """Cliff's delta effect size."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return np.nan
    total = 0
    for xv in x:
        total += np.sum(xv > y)
        total -= np.sum(xv < y)
    return total / (x.size * y.size)


def compute_event_auc(segment, tvec, baseline_win=(-20, -10), t0=0, t1=30):
    """Compute AUC(|z|) in [t0, t1] after baseline subtraction and within-window z-score."""
    segment = np.asarray(segment, dtype=float)
    tvec = np.asarray(tvec, dtype=float)

    baseline_mask = (tvec >= baseline_win[0]) & (tvec < baseline_win[1])
    post_mask = (tvec >= t0) & (tvec <= t1)

    if not baseline_mask.any() or not post_mask.any():
        return np.nan

    baseline = np.nanmean(segment[baseline_mask])
    seg_corr = segment - baseline

    mu = np.nanmean(seg_corr)
    sigma = np.nanstd(seg_corr, ddof=1)
    if not np.isfinite(sigma) or sigma == 0:
        sigma = 1.0

    seg_z = (seg_corr - mu) / sigma

    mask = post_mask & np.isfinite(seg_z) & np.isfinite(tvec)
    if not mask.any():
        return np.nan

    return float(np.trapz(np.abs(seg_z[mask]), tvec[mask]))


def _sample_random_windows_from_combined(sims_all, window_len, n_random, rng):
    """Sample random contiguous windows from combined sims (padded with NaNs)."""
    sims_all = np.asarray(sims_all, dtype=float)
    n_sub, T, n_maps = sims_all.shape

    windows = []
    n_per_sub = max(1, int(np.ceil(n_random / max(n_sub, 1))))

    for si in range(n_sub):
        sims = sims_all[si]  # (T, n_maps)
        valid = np.all(np.isfinite(sims), axis=1)
        if not np.any(valid):
            continue

        last_valid = int(np.where(valid)[0].max())
        n_vol = last_valid + 1

        max_start = n_vol - window_len
        if max_start <= 0:
            continue

        starts = rng.integers(0, max_start, size=n_per_sub)
        for s in starts:
            block = sims[s : s + window_len, :]
            if block.shape[0] != window_len:
                continue
            if not np.all(np.isfinite(block)):
                continue
            windows.append(block)
            if len(windows) >= n_random:
                break

        if len(windows) >= n_random:
            break

    if len(windows) == 0:
        return np.empty((0, window_len, n_maps), dtype=float)

    return np.stack(windows[:n_random], axis=0)


def auc_eventwise(
    all_subject_epochs,
    *,
    combined_npz_path: str,
    output_dir: Optional[str] = None,
    tr_s: Optional[float] = None,
    t0=-10,
    t1=30,
    n_random=2000,
    baseline_win=(-20, -10),
    random_seed: int = 0,
):
    """Event-wise AUC test (NEW pipeline only).

    Random/NULL windows are sampled from `combined_npz_path` (similarity_trace_combined.npz).

    Saves:
      - <output_dir>/auc_eventwise/AUC_EVENTWISE_<a>to<b>.svg
      - <output_dir>/auc_eventwise/auc_eventwise_stats.csv

    If `output_dir` is None, defaults to the directory containing `combined_npz_path`.

    TR handling
    -----------
    The TR (seconds) used for the AUC time axis is taken from the combined NPZ key `tr`.
    You can override it by passing `tr_s`.
    """

    if combined_npz_path is None:
        raise ValueError("combined_npz_path is required (NEW pipeline only).")

    print("\n================ EVENT-WISE AUC TEST ================\n")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(combined_npz_path))

    out_dir = os.path.join(output_dir, "auc_eventwise")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(random_seed)

    # Load combined sims ONCE
    f = np.load(combined_npz_path, allow_pickle=True)
    sims_all = f["sims"]  # (n_sub, T, n_maps) padded with NaNs

    if tr_s is None:
        if "tr" not in f.files:
            raise ValueError(
                "Combined NPZ must contain key 'tr' (TR in seconds), or pass tr_s explicitly."
            )
        tr_s = float(f["tr"])
    else:
        tr_s = float(tr_s)

    results = []

    for pat in TRANSITION_PATTERNS:
        subj_list = all_subject_epochs.get(pat, [])
        if len(subj_list) == 0:
            continue

        print(f"\n--- Transition {pat} ---")

        real_concat = np.concatenate(subj_list, axis=0)  # (n_events, T, n_maps)
        n_events, T, n_maps = real_concat.shape

        half = T // 2
        tvec_full = np.arange(-half, half + 1) * tr_s

        # REAL AUCs
        real_aucs = {m: [] for m in range(n_maps)}
        for m in range(n_maps):
            for ev in range(n_events):
                auc_val = compute_event_auc(
                    real_concat[ev, :, m],
                    tvec_full,
                    baseline_win=baseline_win,
                    t0=t0,
                    t1=t1,
                )
                real_aucs[m].append(auc_val)

        # NULL AUCs from random windows
        rand_windows = _sample_random_windows_from_combined(
            sims_all=sims_all,
            window_len=T,
            n_random=n_random,
            rng=rng,
        )

        rand_aucs = {m: [] for m in range(n_maps)}
        for w in range(rand_windows.shape[0]):
            block = rand_windows[w]  # (T, n_maps)
            for m in range(n_maps):
                auc_val = compute_event_auc(
                    block[:, m],
                    tvec_full,
                    baseline_win=baseline_win,
                    t0=t0,
                    t1=t1,
                )
                rand_aucs[m].append(auc_val)

        # 1×n_maps paper-ready plot (match legacy styling)
        sns.set_style("white")

        fig, axes = plt.subplots(1, n_maps, figsize=(12, 4))

        # If n_maps == 1, axes is a single Axes; make it indexable
        if n_maps == 1:
            axes = [axes]

        for m in range(n_maps):
            r = np.array(real_aucs[m], dtype=float)
            u = np.array(rand_aucs[m], dtype=float)

            r = r[~np.isnan(r)]
            u = u[~np.isnan(u)]

            print(
                f"[DEBUG] {pat} Map{m+1}: real NaNs removed = {np.sum(np.isnan(real_aucs[m]))}, retained={len(r)}"
            )
            print(
                f"[DEBUG] {pat} Map{m+1}: random NaNs removed = {np.sum(np.isnan(rand_aucs[m]))}, retained={len(u)}"
            )

            if len(r) == 0 or len(u) == 0:
                continue

            ks_p = ks_2samp(r, u).pvalue
            mw_p = mannwhitneyu(r, u, alternative="two-sided").pvalue
            d = cliff_delta(r, u)

            results.append(
                {
                    "transition": f"{pat[0]}to{pat[1]}",
                    "map": int(m + 1),
                    "n_real": int(len(r)),
                    "n_null": int(len(u)),
                    "ks_p": float(ks_p),
                    "mw_p": float(mw_p),
                    "cliffs_delta": float(d),
                    "real_mean": float(np.mean(r)),
                    "null_mean": float(np.mean(u)),
                }
            )

            ax = axes[m]

            # Paper colors (match legacy)
            real_color = "#d95f02"  # warm red/orange
            rand_color = "#7570b3"  # muted purple
            edge = "black"

            combined = np.concatenate([r, u])
            bins = np.linspace(combined.min(), combined.max(), 25)

            # --- Histograms ---
            ax.hist(
                u,
                bins=bins,
                density=True,
                alpha=0.55,
                color=rand_color,
                edgecolor=edge,
                linewidth=0.5,
                label="Null",
            )
            ax.hist(
                r,
                bins=bins,
                density=True,
                alpha=0.55,
                color=real_color,
                edgecolor=edge,
                linewidth=0.5,
                label="Real",
            )

            # --- KDE Curves ---
            sns.kdeplot(u, color=rand_color, linewidth=2, ax=ax)
            sns.kdeplot(r, color=real_color, linewidth=2, ax=ax)

            # --- Title + annotation ---
            ax.set_title(
                f"{pat} – Map {m+1}\n" f"KS={ks_p:.1e} | MW={mw_p:.1e} | Δ={d:.3f}",
                fontsize=11,
                weight="bold",
            )

            # --- Axis styling ---
            ax.set_xlabel("AUC(|z|)", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.tick_params(labelsize=9)

            # Remove top/right frame (Nature style)
            sns.despine(ax=ax)

            # Legend
            ax.legend(frameon=False, fontsize=9)

        plt.tight_layout()
        out_fig = os.path.join(out_dir, f"AUC_EVENTWISE_{pat[0]}to{pat[1]}.svg")
        plt.savefig(out_fig, dpi=300)  # high res
        plt.close()
        print("📈 Saved (paper-style):", out_fig)

    df = pd.DataFrame(results)
    df_path = os.path.join(out_dir, "auc_eventwise_stats.csv")
    df.to_csv(df_path, index=False)
    print("\n💾 Saved AUC eventwise stats:", df_path)

    return df
