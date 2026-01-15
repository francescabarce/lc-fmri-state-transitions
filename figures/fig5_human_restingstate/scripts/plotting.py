import matplotlib.pyplot as plt
import numpy as np
import os


def plot_event_similarity(time_axis, mean_trace, sem_trace, labels, title, outfile, zscore_mean=True):
    # Optional: z-score the mean trace across time (and adjust SEM accordingly)
    if zscore_mean:
        # Compute baseline over the whole trace
        baseline_mean = np.mean(mean_trace, axis=0)
        baseline_std = np.std(mean_trace, axis=0)

        # Avoid division by zero
        baseline_std[baseline_std == 0] = 1e-8

        mean_trace = (mean_trace - baseline_mean) / baseline_std
        sem_trace = sem_trace / baseline_std
        
    plt.figure(figsize=(8,5))

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=11)

    # Nature-style mouse pipeline colors
    colors = ["#0D0887", "#F89441"]
    alpha_fill = 0.25

    for i, lab in enumerate(labels):
        plt.plot(
            time_axis,
            mean_trace[:, i],
            lw=2.5,
            color=colors[i % len(colors)],
            label=lab
        )
        plt.fill_between(
            time_axis,
            mean_trace[:, i] - sem_trace[:, i],
            mean_trace[:, i] + sem_trace[:, i],
            color=colors[i % len(colors)],
            alpha=alpha_fill,
            edgecolor="black",
            linewidth=0.5
        )

    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Similarity (z-scored)", fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(frameon=False, fontsize=11)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()