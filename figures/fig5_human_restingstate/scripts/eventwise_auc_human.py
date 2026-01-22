import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ks_2samp

# ============================================================
# Event-wise AUC: baseline correction + event-specific Z-score
# ============================================================
def compute_event_auc(segment, time_axis,
                      baseline_win=(-20, -10), t0=0, t1=15):
    
    baseline_mask = (time_axis >= baseline_win[0]) & (time_axis < baseline_win[1])
    post_mask = (time_axis >= t0) & (time_axis <= t1)

    if not baseline_mask.any() or not post_mask.any():
        return np.nan

    baseline = segment[baseline_mask].mean()
    seg_corr = segment - baseline

    mu = seg_corr.mean()
    sd = seg_corr.std(ddof=1)
    if sd == 0:
        sd = 1
    seg_z = (seg_corr - mu) / sd

    auc_abs = np.trapz(np.abs(seg_z[post_mask]), time_axis[post_mask])
    return auc_abs

# ============================================================
# FULL REAL vs RANDOM EVENTWISE AUC TEST
# ============================================================
def eventwise_auc_test_human(
        segments_all,           # output from analyze_peaks → (TEMPLATE, EVENTS, TIME)
        time_axis,
        sims_concat,
        tr, tr_per_subject,
        peak_label,
        output_folder,
        baseline_win=(-30, -20),
        t0=-5, t1=10,
        n_random=2000):

    print(f"\n================ EVENT-WISE AUC TEST ({peak_label}) ================\n")

    out_dir = os.path.join(output_folder, "eventwise_auc")
    os.makedirs(out_dir, exist_ok=True)

    n_templates = segments_all.shape[0]
    window_len = len(time_axis)

    # -----------------------------------------
    # Generate RANDOM windows (same length)
    # -----------------------------------------
    rng = np.random.default_rng()
    total_tr = sims_concat.shape[0]
    n_subjects = total_tr // tr_per_subject

    random_segments = []

    for _ in range(n_random):

        subj = rng.integers(0, n_subjects)
        start_min = 0
        start_max = tr_per_subject - window_len - 1

        if start_max <= 0:
            continue

        start_local = rng.integers(start_min, start_max)
        start_global = subj*tr_per_subject + start_local
        block = sims_concat[start_global:start_global+window_len, :]

        random_segments.append(block)

    random_segments = np.array(random_segments)     # (rand, T, templates)

    # ============================================================
    # Per-template statistics
    # ============================================================
    for t in range(n_templates):

        seg_real = segments_all[t]      # (events, time)
        n_real = seg_real.shape[0]

        print(f"\nTemplate {t+1} — Real events = {n_real}")

        # REAL AUC
        real_auc = np.array([
            compute_event_auc(seg_real[i], time_axis,
                              baseline_win=baseline_win,
                              t0=t0, t1=t1)
            for i in range(n_real)
        ])
        real_auc = real_auc[~np.isnan(real_auc)]


        # RANDOM AUC
        rand_auc = np.array([
            compute_event_auc(random_segments[i,:,t], time_axis,
                              baseline_win=baseline_win,
                              t0=t0, t1=t1)
            for i in range(len(random_segments))
        ])
        rand_auc = rand_auc[~np.isnan(rand_auc)]


        if len(real_auc)==0 or len(rand_auc)==0:
            print("Skipping: insufficient data")
            continue


        # -----------------------------------------
        # Plot histogram REAL vs RANDOM
        # -----------------------------------------


        # Violin plot AUC-only
        plt.figure(figsize=(6,5))
        parts = plt.violinplot([real_auc, rand_auc], showmeans=True, showextrema=False, widths=0.7)
        colors = ["#2CA02C", "#17BECF"]
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor('black')
            pc.set_alpha(0.35)
        if 'cmeans' in parts:
            parts['cmeans'].set_color("black")
            parts['cmeans'].set_linewidth(2)

        # Stats: Mann–Whitney U (non-parametric location shift) + KS (distribution difference)
        mw_res = mannwhitneyu(real_auc, rand_auc, alternative="two-sided")
        mw_u_auc = mw_res.statistic
        mw_p_auc = mw_res.pvalue

        ks_res = ks_2samp(real_auc, rand_auc, alternative="two-sided", mode="auto")
        ks_d_auc = ks_res.statistic
        ks_p_auc = ks_res.pvalue

        # Print test results + basic descriptives
        print(
            f"  AUC stats | real n={len(real_auc)}, rand n={len(rand_auc)} | "
            f"median(real)={np.median(real_auc):.4g}, median(rand)={np.median(rand_auc):.4g} | "
            f"MW-U={mw_u_auc:.4g}, p={mw_p_auc:.4g} | KS-D={ks_d_auc:.4g}, p={ks_p_auc:.4g}"
        )
        print(f"Template {t+1} — AUC Mann-Whitney U p-value: {mw_p_auc:.4f}")
        
        
        # Significance stars based on MW p-value (kept for continuity with previous figures)
        if mw_p_auc < 0.001:
            sig_label_auc = "***"
        elif mw_p_auc < 0.01:
            sig_label_auc = "**"
        elif mw_p_auc < 0.05:
            sig_label_auc = "*"
        else:
            sig_label_auc = "n.s."
        y_max_auc = max(np.max(real_auc), np.max(rand_auc))
        y_min_auc = min(np.min(real_auc), np.min(rand_auc))
        y_pos_auc = y_max_auc + 0.1 * (y_max_auc - y_min_auc)
        plt.text(1.5, y_pos_auc, sig_label_auc, ha="center", va="bottom",
                 fontsize=20, fontweight="bold")
        plt.text(1.5, y_pos_auc, f"MW p={mw_p_auc:.2g} | KS p={ks_p_auc:.2g}", ha="center", va="top",
                 fontsize=10)

        plt.xticks([1,2], ["Real AUC","Random AUC"], fontsize=12)
        plt.ylabel("AUC(|z|)", fontsize=13)
        plt.title(f"{peak_label} – Template {t+1} — AUC", fontsize=14, fontweight="bold")
        vout_auc = os.path.join(out_dir, f"Violin_AUC_{peak_label}_T{t+1}.svg")
        plt.tight_layout()
        plt.savefig(vout_auc, dpi=160)
        plt.close()