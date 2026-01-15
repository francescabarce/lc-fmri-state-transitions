# fc_states_only.py
import os
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

plt.rcParams["svg.fonttype"] = "none"

# Paper palette
TEMPLATE1_COLOR = "#2b2c7b"
TEMPLATE2_COLOR = "#f79441"
SUBSPACE_COLOR = "#2F2F2F"


def compute_subspace_strength(
    t1_state: np.ndarray,
    t2_state: np.ndarray,
    method: str = "rms_mag",
) -> np.ndarray:
    """Compute a rotation-invariant (Template1, Template2) subspace strength per subject.

    Parameters
    ----------
    t1_state, t2_state : (n_tr, n_sub)
        Template-expression time series.
    method : {"rms_mag", "energy"}
        - "rms_mag": sqrt(mean(t1^2 + t2^2)) per subject.
        - "energy": mean(t1^2 + t2^2) per subject.

    Returns
    -------
    strength : (n_sub,)
    """
    s1 = np.asarray(t1_state, dtype=float)
    s2 = np.asarray(t2_state, dtype=float)

    if s1.ndim != 2 or s2.ndim != 2 or s1.shape != s2.shape:
        raise ValueError(
            f"t1_state and t2_state must be 2D and the same shape. Got {s1.shape} vs {s2.shape}."
        )

    mag2 = s1**2 + s2**2
    method = str(method).lower()

    if method in ("energy", "mean_mag2"):
        return np.nanmean(mag2, axis=0)
    if method in ("rms_mag", "rms"):
        return np.sqrt(np.nanmean(mag2, axis=0))

    raise ValueError("Unknown method. Use 'rms_mag' or 'energy'.")


def _pick_example_subjects(strength: np.ndarray) -> list[int]:
    """Pick example subjects by extreme strength (low/high)."""
    strength = np.asarray(strength, dtype=float).ravel()
    if strength.size < 1:
        raise ValueError("No subjects available.")
    order = np.argsort(strength)
    return [int(order[0]), int(order[-1])]


def _smooth_ts(x: np.ndarray, win: int = 7) -> np.ndarray:
    """Simple moving-average smoothing for visualization only."""
    x = np.asarray(x, dtype=float)
    if win <= 1:
        return x
    k = np.ones(int(win), dtype=float) / float(win)
    return np.convolve(x, k, mode="same")


def plot_panelA_states_timeseries(
    t1_state: np.ndarray,
    t2_state: np.ndarray,
    strength: np.ndarray,
    outpath: str,
    tr_sec: float = 0.58,
    subject_indices: Optional[Sequence[int]] = None,
) -> None:
    """Panel A: example Template1/Template2 state(t) and subspace magnitude."""
    t1_state = np.asarray(t1_state, dtype=float)
    t2_state = np.asarray(t2_state, dtype=float)

    if t1_state.shape != t2_state.shape or t1_state.ndim != 2:
        raise ValueError(f"Invalid states shapes: {t1_state.shape} vs {t2_state.shape}.")

    n_tr, n_sub = t1_state.shape
    strength = np.asarray(strength, dtype=float).ravel()
    if strength.size != n_sub:
        raise ValueError(f"strength length ({strength.size}) must match n_sub ({n_sub}).")

    if subject_indices is None:
        subject_indices = _pick_example_subjects(strength)
    subject_indices = [int(i) for i in subject_indices]
    if len(subject_indices) not in (1, 2):
        raise ValueError("Panel A supports 1 or 2 example subjects.")
    for s in subject_indices:
        if s < 0 or s >= n_sub:
            raise ValueError(f"Invalid subject index {s}. n_sub={n_sub}.")

    t = np.arange(n_tr) * float(tr_sec)

    traces = []
    for s in subject_indices:
        s1 = t1_state[:, s]
        s2 = t2_state[:, s]
        mag = np.sqrt(s1**2 + s2**2)
        traces.append((s, _smooth_ts(s1), _smooth_ts(s2), _smooth_ts(mag)))

    all_vals = np.concatenate([np.concatenate([a, b, c]) for _, a, b, c in traces])
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    pad = 0.07 * (hi - lo) if hi > lo else 1.0
    ylo, yhi = lo - pad, hi + pad

    fig_h = 3.2 * len(traces)
    fig, axs = plt.subplots(len(traces), 1, figsize=(12, fig_h), sharex=True)
    if len(traces) == 1:
        axs = [axs]

    for ax, (s, s1s, s2s, mags) in zip(axs, traces):
        ax.plot(t, mags, color=SUBSPACE_COLOR, linewidth=2.8, label="Subspace magnitude")
        ax.plot(t, s1s, color=TEMPLATE1_COLOR, linewidth=1.6, alpha=0.9, label="Template 1")
        ax.plot(t, s2s, color=TEMPLATE2_COLOR, linewidth=1.6, alpha=0.9, label="Template 2")

        ax.set_ylim(ylo, yhi)
        ax.set_ylabel("Expression (a.u.)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.text(
            0.01,
            0.92,
            f"RMS = {float(strength[s]):.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="black",
                alpha=0.98,
            ),
        )
        ax.text(
            0.01,
            0.10,
            f"Subject {s}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#444444",
        )

    axs[-1].set_xlabel("Time (s)")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outpath)
    plt.close(fig)


def plot_panelB_magnitude_distribution_states_only(
    t1_state: np.ndarray,
    t2_state: np.ndarray,
    strength: np.ndarray,
    outpath: str,
    subject_indices: Optional[Sequence[int]] = None,
    bins: int = 40,
) -> None:
    """Panel B: magnitude distribution for low/high strength example subjects."""
    t1_state = np.asarray(t1_state, dtype=float)
    t2_state = np.asarray(t2_state, dtype=float)

    if t1_state.shape != t2_state.shape or t1_state.ndim != 2:
        raise ValueError(f"Invalid states shapes: {t1_state.shape} vs {t2_state.shape}.")

    n_tr, n_sub = t1_state.shape
    strength = np.asarray(strength, dtype=float).ravel()
    if strength.size != n_sub:
        raise ValueError(f"strength length ({strength.size}) must match n_sub ({n_sub}).")

    if subject_indices is None:
        subject_indices = _pick_example_subjects(strength)
    subject_indices = [int(i) for i in subject_indices]
    if len(subject_indices) != 2:
        raise ValueError("Panel B requires exactly 2 subjects (low/high).")

    s_low, s_high = subject_indices
    for s in (s_low, s_high):
        if s < 0 or s >= n_sub:
            raise ValueError(f"Invalid subject index {s}. n_sub={n_sub}.")

    mag_low = np.sqrt(t1_state[:, s_low] ** 2 + t2_state[:, s_low] ** 2)
    mag_high = np.sqrt(t1_state[:, s_high] ** 2 + t2_state[:, s_high] ** 2)

    rms_low = float(np.sqrt(np.mean(mag_low**2)))
    rms_high = float(np.sqrt(np.mean(mag_high**2)))

    mmin = float(min(np.min(mag_low), np.min(mag_high)))
    mmax = float(max(np.max(mag_low), np.max(mag_high)))
    if not np.isfinite(mmin) or not np.isfinite(mmax) or mmax <= mmin:
        raise ValueError("Invalid magnitude range for histogram.")

    edges = np.linspace(mmin, mmax, int(bins) + 1)

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(111)

    ax.hist(
        mag_low,
        bins=edges,
        density=True,
        alpha=0.35,
        color=TEMPLATE1_COLOR,
        label=f"Low strength | subj {s_low} | RMS={rms_low:.3g}",
    )
    ax.hist(
        mag_high,
        bins=edges,
        density=True,
        alpha=0.35,
        color=TEMPLATE2_COLOR,
        label=f"High strength | subj {s_high} | RMS={rms_high:.3g}",
    )

    ax.axvline(rms_low, linewidth=2.6, alpha=0.95, color=TEMPLATE1_COLOR)
    ax.axvline(rms_high, linewidth=2.6, alpha=0.95, color=TEMPLATE2_COLOR)

    ax.set_xlabel("Subspace magnitude  sqrt(T1(t)^2 + T2(t)^2)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    ax.text(
        0.03,
        0.97,
        "RMS(magnitude) summarizes typical expression strength",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="black",
            alpha=0.95,
        ),
    )

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_panelC_scatter_with_bootstrap_ci(
    strength: np.ndarray,
    fc: np.ndarray,
    outpath: str,
    xlabel: str,
    title: str,
    n_boot: int = 5000,
    seed: int = 0,
) -> dict:
    """Panel C: scatter + OLS line + bootstrap 95% CI band."""
    strength = np.asarray(strength, dtype=float).ravel()
    fc = np.asarray(fc, dtype=float).ravel()

    good = np.isfinite(strength) & np.isfinite(fc)
    x = strength[good]
    y = fc[good]

    if x.size < 3:
        raise ValueError("Need at least 3 subjects for bootstrap CI plot.")

    lr = linregress(x, y)
    xg = np.linspace(np.min(x), np.max(x), 250)

    rng = np.random.default_rng(seed)
    slopes = np.zeros(int(n_boot), dtype=float)
    intercepts = np.zeros(int(n_boot), dtype=float)

    n = x.size
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        lrb = linregress(x[idx], y[idx])
        slopes[b] = lrb.slope
        intercepts[b] = lrb.intercept

    yg_boot = intercepts[:, None] + slopes[:, None] * xg[None, :]
    lo = np.percentile(yg_boot, 2.5, axis=0)
    hi = np.percentile(yg_boot, 97.5, axis=0)

    r, p = pearsonr(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    ax.scatter(x, y, s=90, edgecolors="black", linewidth=0.7, alpha=0.9)
    ax.plot(xg, lr.intercept + lr.slope * xg, linewidth=3.0, alpha=0.95)
    ax.fill_between(xg, lo, hi, alpha=0.25)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Global FC strength (mean Fisher-z)")
    ax.set_title(title)

    ax.text(
        0.03,
        0.97,
        f"r = {float(r):.2f}\np = {float(p):.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="black",
            alpha=0.95,
        ),
    )

    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

    return {"slope": float(lr.slope), "intercept": float(lr.intercept), "r": float(r), "p": float(p)}


def run_fc_states_only_pipeline(
    t1_state: np.ndarray,
    t2_state: np.ndarray,
    fc_strength: np.ndarray,
    output_folder: str,
    tr_sec: float = 0.58,
    strength_method: str = "subspace_rms",
    n_boot: int = 5000,
    seed: int = 0,
    panelA_subjects: Optional[Sequence[int]] = None,
    out_prefix: str = "Fig5_FC_states_only",
) -> dict:
    """Reviewer entrypoint: build Fig5 FC panels using only saved *states* and FC strength.

    Outputs (ONLY):
      - {out_prefix}_Subspace_RMSmag_vs_FC_scatter_bootstrapCI.svg
      - {out_prefix}_PanelA_states_timeseries.svg
      - {out_prefix}_PanelB_magnitude_distribution.svg
    """

    os.makedirs(output_folder, exist_ok=True)
    plot_folder = os.path.join(output_folder, "FC_states_only_plots")
    os.makedirs(plot_folder, exist_ok=True)

    t1_state = np.asarray(t1_state, dtype=float)
    t2_state = np.asarray(t2_state, dtype=float)
    fc_strength = np.asarray(fc_strength, dtype=float).ravel()

    if t1_state.shape != t2_state.shape or t1_state.ndim != 2:
        raise ValueError(f"t1_state and t2_state must match and be 2D. Got {t1_state.shape} vs {t2_state.shape}.")

    n_tr, n_sub = t1_state.shape
    if fc_strength.size != n_sub:
        raise ValueError(f"fc_strength length ({fc_strength.size}) must match n_sub ({n_sub}).")

    sm = str(strength_method).lower()
    if sm in ("subspace_rms", "subspace_rms_mag"):
        strength = compute_subspace_strength(t1_state, t2_state, method="rms_mag")
        xlab = "Template(1,2) subspace strength (RMS of magnitude)"
        fname_tag = "Subspace_RMSmag"
    elif sm in ("subspace_energy", "subspace"):
        strength = compute_subspace_strength(t1_state, t2_state, method="energy")
        xlab = "Template(1,2) subspace strength (mean magnitude^2)"
        fname_tag = "Subspace_Energy"
    else:
        raise ValueError("strength_method must be: 'subspace_rms' or 'subspace_energy'.")

    # Choose subjects
    if panelA_subjects is None:
        panelA_subjects = _pick_example_subjects(strength)
    else:
        panelA_subjects = [int(x) for x in panelA_subjects]

    # Panel A
    panelA_path = os.path.join(plot_folder, f"{out_prefix}_PanelA_states_timeseries.svg")
    plot_panelA_states_timeseries(
        t1_state=t1_state,
        t2_state=t2_state,
        strength=strength,
        outpath=panelA_path,
        tr_sec=tr_sec,
        subject_indices=panelA_subjects,
    )

    # Panel B
    panelB_path = os.path.join(plot_folder, f"{out_prefix}_PanelB_magnitude_distribution.svg")
    plot_panelB_magnitude_distribution_states_only(
        t1_state=t1_state,
        t2_state=t2_state,
        strength=strength,
        outpath=panelB_path,
        subject_indices=panelA_subjects,
    )

    # Panel C (bootstrap CI scatter) — ONLY scatter we keep
    panelC_path = os.path.join(
        plot_folder,
        f"{out_prefix}_{fname_tag}_vs_FC_scatter_bootstrapCI.svg",
    )
    stats = plot_panelC_scatter_with_bootstrap_ci(
        strength=strength,
        fc=fc_strength,
        outpath=panelC_path,
        xlabel=xlab,
        title=f"{fname_tag} vs Global FC (bootstrap 95% CI)",
        n_boot=n_boot,
        seed=seed,
    )

    return {
        "strength": strength,
        "fc_strength": fc_strength,
        "strength_method": strength_method,
        "fname_tag": fname_tag,
        "r": stats["r"],
        "p": stats["p"],
        "plot_folder": plot_folder,
    }