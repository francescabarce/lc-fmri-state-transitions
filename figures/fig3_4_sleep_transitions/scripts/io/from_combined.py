# sleep_transition_similarity/io/from_combined.py
import numpy as np

def build_epochs_from_combined_npz(combined_npz_path: str, win_trs: int):
    f = np.load(combined_npz_path, allow_pickle=True)

    sims_all = np.asarray(f["sims"], dtype=float)          # (n_sub, T, n_maps)
    subjects = [str(s) for s in f["subjects"]]
    tr = float(f["tr"])

    trans_1to3 = f["transitions_1to3"]  # object array length n_sub
    trans_3to1 = f["transitions_3to1"]

    n_sub, T, n_maps = sims_all.shape
    W = int(win_trs)

    all_subject_epochs = {(1, 3): [], (3, 1): []}

    def _extract_subject_epochs(sims_sub: np.ndarray, event_idx) -> np.ndarray:
        epochs = []
        for idx in np.asarray(event_idx, dtype=int).ravel():
            start = idx - W
            end = idx + W
            if start < 0 or end >= sims_sub.shape[0]:
                continue
            ep = sims_sub[start:end + 1, :]  # (2W+1, n_maps)
            if np.any(~np.isfinite(ep)):
                continue
            epochs.append(ep)
        if len(epochs) == 0:
            return np.zeros((0, 2 * W + 1, n_maps), dtype=float)
        return np.stack(epochs, axis=0)

    for i in range(n_sub):
        sims_sub = sims_all[i, :, :]

        # trim NaN padding (if any)
        valid = np.all(np.isfinite(sims_sub), axis=1)
        if np.any(valid):
            last = np.where(valid)[0][-1] + 1
            sims_sub = sims_sub[:last, :]

        ep13 = _extract_subject_epochs(sims_sub, trans_1to3[i])
        ep31 = _extract_subject_epochs(sims_sub, trans_3to1[i])

        if ep13.shape[0] > 0:
            all_subject_epochs[(1, 3)].append(ep13)
        if ep31.shape[0] > 0:
            all_subject_epochs[(3, 1)].append(ep31)

    return subjects, all_subject_epochs, tr