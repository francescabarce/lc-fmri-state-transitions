# peaks.py
import numpy as np
from io_utils import zscore


def extract_segments(sim, peaks, tr_per_subject, window):
    all_segments = []
    subject_ids = []
    total_tr = sim.shape[0]
    seg_len = 2 * window

    for p in peaks:
        subj = p // tr_per_subject
        tr_loc = p % tr_per_subject

        start_local = max(0, tr_loc - window)
        end_local   = min(tr_per_subject, tr_loc + window)

        start_global = subj * tr_per_subject + start_local
        end_global   = subj * tr_per_subject + end_local

        # invalid global window
        if start_global < 0 or end_global > total_tr:
            continue

        seg = np.full(seg_len, np.nan)
        seg_start = window - (tr_loc - start_local)
        seg_end   = seg_start + (end_local - start_local)

        if seg_start < 0 or seg_end > seg_len:
            continue

        seg[seg_start:seg_end] = sim[start_global:end_global]
        seg = zscore(seg)
        all_segments.append(seg)
        subject_ids.append(subj)

    return np.array(all_segments), np.array(subject_ids)


def analyze_peaks(sims_concat, peaks, tr, tr_per_subject, window_sec, label, output_folder):

    window = int(window_sec / tr)
    T = 2 * window
    n_templates = sims_concat.shape[1]

    time_axis = np.arange(-window, window) * tr

    mean_all = []
    sem_all = []
    segments_all = []
    subject_ids_all = []

    for t in range(n_templates):
        segs, subj_ids = extract_segments(
            sims_concat[:, t],
            peaks,
            tr_per_subject,
            window
        )

        segments_all.append(segs)
        subject_ids_all.append(subj_ids)

        mean_t = np.nanmean(segs, axis=0)
        sem_t = np.nanstd(segs, axis=0) / np.sqrt(segs.shape[0])

        mean_all.append(mean_t)
        sem_all.append(sem_t)

    mean_all = np.vstack(mean_all).T
    sem_all  = np.vstack(sem_all).T
    segments_all = np.array(segments_all)  # shape = (templates, N_events, T)
    subject_ids_all = np.array(subject_ids_all, dtype=object)

    return time_axis, mean_all, sem_all, segments_all, subject_ids_all