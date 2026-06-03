"""
Feature extraction for the AMBER ERP classification pipeline.

Replaces the original MATLAB scripts (extract_features_stat_params.m and
extract_features_across_time.m) with a unified Python implementation using
MNE-Python.

Epoch structure
~~~~~~~~~~~~~~~
In the .set EEGLAB files, recordings are pre-epoched for each event labeled
'1' (rare) or '2' (frequent). Each epoch has exactly ONE event centered at
t=0, with a time window spanning [-0.2, 1.0] seconds. There are 360 epochs
per recording: 36 rare and 324 frequent. Events appear sequentially every
250 ms (ISI) in the original continuous recording, but each epoch captures
only the response to its triggering event. Only the first (and only) event
in each epoch is considered for labelling and feature extraction.

This script:
1. Loads cleaned .set files from preprocessing/clean/<method>/
2. Labels epochs based on their single event: event '1' = rare, event '2' = frequent
3. Validates expected epoch counts (36 rare, 324 frequent per recording)
4. Rejects trials with amplitudes exceeding a threshold
5. Extracts features (statistical ERP params and/or temporal features)
6. Saves per-method .npz files to data_for_classification/

Usage:
    python extract_features.py [METHOD] [--feature-type {stat,temporal,both}]
                                [--tasks X1 X2 ...]

    METHOD is one of: RAW, ASR, ICLabel, MARA, GEDAI  (default: all)
    --feature-type controls what features to extract:
        stat     – statistical ERP parameters (PA, MA, PL, FL) per component
        temporal – amplitude + ERSP time-frequency features per cluster
        both     – both feature types (default)
    --tasks restricts which RSVP tasks to include (default: all standard + artifact RSVP tasks)

Examples:
    python extract_features.py RAW
    python extract_features.py ASR --feature-type stat
    python extract_features.py --tasks X1 X2
    python extract_features.py all              # run for every method
"""

import argparse
import os
import re
import sys
import warnings

import mne
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import stft

# Local config
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    ARTIFACT_RSVP,
    CLEAN_ROOT,
    DATA_OUT_ROOT,
    DENOISING_METHODS,
    ELEC_CLUSTS,
    ERP_COMPONENTS,
    ERP_PARAMS,
    N_EPOCHS_EXPECTED,
    N_FREQ_EXPECTED,
    N_JOBS,
    N_RARE_EXPECTED,
    N_TRIALS_MAX,
    N_TRIALS_THRESHOLD,
    NOISE_THRESHOLD_UV,
    SAMPLING_FREQ,
    STANDARD_RSVP,
    get_data_version,
    get_task_id,
    resolve_recordings,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ──────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────


def discover_set_files(clean_dir: str, method: str) -> list[dict]:
    """Find all .set files in *clean_dir* and return metadata dicts.

    Each dict has keys: 'path', 'subject', 'session', 'task', 'method'.
    Only RSVP task files (X1, X2, X4, X6, X8) are returned.
    """
    pattern = re.compile(r"P(\d+)-Ss(\d+)-(X\d+)-eeg_" + re.escape(method) + r"\.set$")
    files = []
    for fname in sorted(os.listdir(clean_dir)):
        m = pattern.match(fname)
        if m:
            files.append(
                {
                    "path": os.path.join(clean_dir, fname),
                    "subject": int(m.group(1)),
                    "session": int(m.group(2)),
                    "task": m.group(3),
                    "method": method,
                }
            )
    return files


# ──────────────────────────────────────────────────────────────────────
# Epoch loading & trial selection
# ──────────────────────────────────────────────────────────────────────


def load_and_epoch(file_info: dict):
    """Load a pre-epoched .set file and assign labels based on event type.

    In the AMBER dataset, each .set file contains pre-epoched data with
    exactly ONE event per epoch, centered at t=0. Events are labeled '1'
    (rare) or '2' (frequent). There are 360 epochs per recording: 36 rare
    and 324 frequent. Only the first (and only) event in each epoch is
    considered for labelling.

    Returns
    -------
    epochs : mne.Epochs | None
        Epochs object, or None if the file cannot be loaded or has
        insufficient event types.
    labels : np.ndarray | None
        1-D array of labels (0=frequent, 1=rare) for each epoch.
    task : str
        Task identifier (e.g. 'X1').
    """
    # Load pre-epoched data from EEGLAB
    epochs = mne.io.read_epochs_eeglab(file_info["path"])

    # Build event mapping: '1' -> rare, '2' -> frequent.
    # The AMBER dataset uses event labels '1' (rare) and '2' (frequent).
    # Each epoch has exactly one event (centered at t=0), so we assign
    # the label based on that single event.
    stim_ids = {}
    for k, v in epochs.event_id.items():
        k_str = str(k).strip()
        k_str = k_str[0]
        # Exact match first (most common case: event names are '1' or '2')
        if k_str == "1":
            stim_ids["rare"] = v
        elif k_str == "2":
            stim_ids["frequent"] = v
        # Substring fallback for names like 'R  1', 'stimulus_1', etc.
        elif "1" in k_str and "2" not in k_str:
            stim_ids["rare"] = v
        elif "2" in k_str and "1" not in k_str:
            stim_ids["frequent"] = v

    if len(stim_ids) < 2:
        print(
            f"  [SKIP] Could not identify both stimulus types in "
            f"{os.path.basename(file_info['path'])}. "
            f"Events: {list(epochs.event_id.keys())}"
        )
        return None, None, file_info["task"]

    # Assign labels: 1 = rare (event '1'), 0 = frequent (event '2')
    labels = np.array(
        [
            1 if epochs.events[i, 2] == stim_ids.get("rare", -1) else 0
            for i in range(len(epochs.events))
        ]
    )

    n_total = len(epochs)
    n_rare = int(np.sum(labels == 1))
    n_freq = int(np.sum(labels == 0))
    print(
        f"    Loaded {n_total} epochs: {n_rare} rare (event '1'), "
        f"{n_freq} frequent (event '2')"
    )

    # Validate expected epoch counts
    if n_total != N_EPOCHS_EXPECTED:
        print(f"    [WARN] Expected {N_EPOCHS_EXPECTED} total epochs, got {n_total}")
    if n_rare != N_RARE_EXPECTED:
        print(f"    [WARN] Expected {N_RARE_EXPECTED} rare epochs, got {n_rare}")
    if n_freq != N_FREQ_EXPECTED:
        print(f"    [WARN] Expected {N_FREQ_EXPECTED} frequent epochs, got {n_freq}")

    return epochs, labels, file_info["task"]


def reject_trials(
    epochs: mne.Epochs,
    labels: np.ndarray,
    cluster_channels: dict,
    threshold_uv: float,
    min_trials: int = N_TRIALS_THRESHOLD,
    max_trials: int = N_TRIALS_MAX,
):
    """Reject trials with amplitudes exceeding *threshold_uv* at relevant clusters.

    Returns filtered (data, labels) or (None, None) if too few trials remain.
    """
    data = epochs.get_data()  # (n_trials, n_channels, n_times)
    times = epochs.times

    # Build channel indices for rejection
    ch_names = epochs.info["ch_names"]
    reject_idx = []
    for cluster_chs in cluster_channels.values():
        for ch in cluster_chs:
            if ch in ch_names:
                reject_idx.append(ch_names.index(ch))

    if not reject_idx:
        # Fall back to all channels if cluster names don't match
        reject_idx = list(range(data.shape[1]))

    reject_idx = np.array(reject_idx)

    # Find bad trials
    data_check = data[:, reject_idx, :]
    bad_trials = np.any(np.abs(data_check) > threshold_uv, axis=(1, 2))

    # Apply rejection
    good_idx = ~bad_trials
    data_clean = data[good_idx]
    labels_clean = labels[good_idx]

    n_good = len(labels_clean)
    n_rare = int(np.sum(labels_clean == 1))
    n_freq = int(np.sum(labels_clean == 0))

    if n_good < min_trials:
        print(
            f"  [SKIP] Only {n_good} good trials (min={min_trials}): "
            f"{n_rare} rare, {n_freq} frequent"
        )
        return None, None

    print(f"    After rejection: {n_good} trials ({n_rare} rare, {n_freq} frequent)")

    # Cap the number of trials per subject to max_trials
    if n_good > max_trials:
        # Balance by keeping equal-ish numbers of rare and frequent
        rare_idx = np.where(labels_clean == 1)[0]
        freq_idx = np.where(labels_clean == 0)[0]
        n_each = max_trials // 2
        if len(rare_idx) > n_each:
            rng = np.random.default_rng(0)
            rare_idx = rng.choice(rare_idx, n_each, replace=False)
        if len(freq_idx) > max_trials - len(rare_idx):
            n_freq_keep = max_trials - len(rare_idx)
            rng = np.random.default_rng(0)
            freq_idx = rng.choice(freq_idx, n_freq_keep, replace=False)
        keep_idx = np.sort(np.concatenate([rare_idx, freq_idx]))
        data_clean = data_clean[keep_idx]
        labels_clean = labels_clean[keep_idx]

    return data_clean, labels_clean


# ──────────────────────────────────────────────────────────────────────
# Feature extraction: Statistical ERP parameters
# ──────────────────────────────────────────────────────────────────────


def extract_stat_params(
    data: np.ndarray, times: np.ndarray, sfreq: float, cluster_channels: dict
):
    """Extract statistical ERP parameters (PA, MA, PL, FL) for each component.

    Parameters
    ----------
    data : np.ndarray, shape (n_trials, n_channels, n_times)
    times : np.ndarray, shape (n_times,)
    sfreq : float, sampling frequency
    cluster_channels : dict mapping cluster name to list of channel names

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_features)
        Feature matrix. Columns are PA/MA/PL/FL for each component in order.
    feature_names : list of str
    """
    n_trials = data.shape[0]
    feature_names = []
    features = []

    for comp_name, comp_info in ERP_COMPONENTS.items():
        window_ms = comp_info["window_ms"]
        polarity = comp_info["polarity"]
        cluster = comp_info["cluster"]

        # Find time indices for this window
        t_start = window_ms[0] / 1000.0  # convert ms to seconds
        t_end = window_ms[1] / 1000.0
        time_mask = (times >= t_start) & (times <= t_end)
        time_idx = np.where(time_mask)[0]

        if len(time_idx) == 0:
            # Window doesn't overlap with data – fill with NaN
            for param in ERP_PARAMS:
                feature_names.append(f"{comp_name}_{param}")
                features.append(np.full(n_trials, np.nan))
            continue

        # Get channel indices for this cluster
        ch_names_in_data = None  # Will use channel indices directly
        # Average across cluster channels
        ch_idx = []
        # Use indices relative to the 32-channel layout
        # cluster_channels maps names -> indices (1-based)
        cluster_indices = cluster_channels.get(cluster, [])
        # Convert to 0-based
        cluster_indices_0 = [i - 1 for i in cluster_indices if i - 1 < data.shape[1]]
        if not cluster_indices_0:
            # Fallback: use all channels
            cluster_indices_0 = list(range(data.shape[1]))

        # Average ERP across cluster
        erp = np.mean(data[:, cluster_indices_0, :], axis=1)  # (n_trials, n_times)
        erp_window = erp[:, time_idx]  # (n_trials, n_timepoints_in_window)

        # Peak amplitude (PA)
        if polarity == "positive":
            pa = np.max(erp_window, axis=1)
            pa_idx = np.argmax(erp_window, axis=1)
        else:
            pa = np.min(erp_window, axis=1)
            pa_idx = np.argmin(erp_window, axis=1)

        # Mean amplitude (MA)
        ma = np.mean(erp_window, axis=1)

        # Peak latency (PL) in ms
        pl = times[time_idx[pa_idx]] * 1000  # convert s -> ms

        # Fractional latency (FL) in ms
        fl = np.zeros(n_trials)
        for i in range(n_trials):
            peak_val = pa[i]
            half_val = peak_val / 2.0
            # Search from onset to peak
            if polarity == "positive":
                cross_idx = np.where(erp[i, : time_idx[pa_idx[i]] + 1] >= half_val)[0]
            else:
                cross_idx = np.where(erp[i, : time_idx[pa_idx[i]] + 1] <= half_val)[0]
            if len(cross_idx) > 0:
                fl[i] = times[cross_idx[-1]] * 1000  # ms
            else:
                fl[i] = np.nan

        for param_name, param_vals in zip(ERP_PARAMS, [pa, ma, pl, fl]):
            feature_names.append(f"{comp_name}_{param_name}")
            features.append(param_vals)

    X = np.column_stack(features)
    return X, feature_names


# ──────────────────────────────────────────────────────────────────────
# Feature extraction: Temporal features (amplitude + ERSP)
# ──────────────────────────────────────────────────────────────────────


def extract_temporal_features(
    data: np.ndarray, times: np.ndarray, sfreq: float, cluster_channels: dict
):
    """Extract temporal features: amplitude and ERSP for each cluster.

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_features, n_times)
        Temporal feature tensor.
    feature_names : list of str
    """
    freqs = np.array([4, 8, 12, 16, 20, 24, 28, 32, 36])
    n_freqs = len(freqs)
    n_trials, n_channels, n_times = data.shape

    feature_names_list = []
    all_features = []

    for cluster_name, cluster_indices in cluster_channels.items():
        # Convert to 0-based
        idx_0 = [i - 1 for i in cluster_indices if i - 1 < n_channels]
        if not idx_0:
            idx_0 = list(range(n_channels))

        # Amplitude
        amp = np.mean(data[:, idx_0, :], axis=1)  # (n_trials, n_times)
        feature_names_list.append(f"X_amp_{cluster_name[:3].capitalize()}")
        all_features.append(amp[:, np.newaxis, :])  # (n_trials, 1, n_times)

        # ERSP (time-frequency via STFT)
        ersp = np.zeros((n_trials, n_times, n_freqs))
        for trial in range(n_trials):
            for ci, ch_idx in enumerate(idx_0):
                f, t, Zxx = stft(
                    data[trial, ch_idx, :], fs=sfreq, nperseg=64, noverlap=48
                )
                # Interpolate to original times
                power = np.abs(Zxx) ** 2
                # Pick closest frequencies
                for fi, target_f in enumerate(freqs):
                    freq_idx = np.argmin(np.abs(f - target_f))
                    ersp[trial, :, fi] += np.interp(times, t, power[freq_idx, :])
            ersp[trial] /= len(idx_0)  # average over channels

        ersp_db = 10 * np.log10(ersp + 1e-12)  # convert to dB
        for fi, target_f in enumerate(freqs):
            feature_names_list.append(
                f"X_tf_{cluster_name[:3].capitalize()}{int(target_f)}"
            )
            # Select frequency band fi: ersp_db[:,:,fi] has shape (n_trials, n_times)
            all_features.append(
                ersp_db[:, :, fi][:, np.newaxis, :]
            )  # (n_trials, 1, n_times)

    # Stack: (n_trials, n_features, n_times)
    X = np.concatenate(all_features, axis=1)
    return X, feature_names_list


# ──────────────────────────────────────────────────────────────────────
# Per-file processing (standalone for parallel dispatch)
# ──────────────────────────────────────────────────────────────────────


def _process_single_file(file_info, feature_type, cluster_channels, noise_threshold_uv):
    """Process a single .set file: load, reject trials, extract features.

    Standalone function designed for use with joblib parallel processing.
    Returns a dict with extracted data, or None if the file could not be
    processed.
    """
    try:
        epochs, labels, task = load_and_epoch(file_info)
    except Exception as e:
        print(f"  [SKIP] Error loading {file_info['path']}: {e}")
        return None

    if epochs is None:
        return None

    # Get channel names from epochs info
    ch_names = epochs.info["ch_names"]
    # Build cluster_channels with actual channel name resolution
    cluster_ch_by_name = {}
    for cluster_name, indices in cluster_channels.items():
        chs = []
        for idx in indices:
            if idx - 1 < len(ch_names):
                chs.append(ch_names[idx - 1])
        cluster_ch_by_name[cluster_name] = chs

    # Reject bad trials
    data_clean, labels_clean = reject_trials(
        epochs, labels, cluster_ch_by_name, noise_threshold_uv
    )
    if data_clean is None:
        return None

    n_trials = data_clean.shape[0]
    n_rare = int(np.sum(labels_clean == 1))
    n_freq = int(np.sum(labels_clean == 0))
    print(
        f"    Subject {file_info['subject']}, Session {file_info['session']}, "
        f"Task {task}: {n_trials} good trials ({n_rare} rare, {n_freq} frequent)"
    )

    times = epochs.times
    sfreq = epochs.info["sfreq"]

    # Determine task type label (0=standard RSVP, 1=artifact RSVP)
    task_label = 1 if task in ARTIFACT_RSVP else 0
    sub_id = file_info["subject"]

    result = {
        "IDs": [[sub_id, task_label]] * n_trials,
        "y_stim": labels_clean.tolist(),
        "y_task": [task_label] * n_trials,
        "subject_ids": [sub_id] * n_trials,
        "sub_id": sub_id,
        "task": task,
        "task_label": task_label,
        "n_trials": n_trials,
        "times": times,
        "sfreq": sfreq,
    }

    if feature_type in ("stat", "both"):
        X_stat, feat_names_stat = extract_stat_params(
            data_clean, times, sfreq, cluster_channels
        )
        result["X_stat"] = X_stat
        result["feat_names_stat"] = feat_names_stat

    if feature_type in ("temporal", "both"):
        X_temp, feat_names_temp = extract_temporal_features(
            data_clean, times, sfreq, cluster_channels
        )
        result["X_temp"] = X_temp
        result["feat_names_temp"] = feat_names_temp

    return result


# ──────────────────────────────────────────────────────────────────────
# Main extraction loop
# ──────────────────────────────────────────────────────────────────────


def extract_features_for_method(
    method: str,
    feature_type: str = "both",
    tasks: list[str] | None = None,
    n_jobs: int = N_JOBS,
):
    """Run feature extraction for a single denoising method.

    Parameters
    ----------
    method : str
        One of the keys in DENOISING_METHODS.
    feature_type : str
        "stat", "temporal", or "both".
    tasks : list[str] or None
        Which RSVP tasks to include. None = all RSVP tasks.
    n_jobs : int
        Number of parallel jobs. -1 = all CPUs, 1 = sequential.
    """
    if tasks is None:
        tasks = STANDARD_RSVP + ARTIFACT_RSVP

    clean_dir = os.path.join(CLEAN_ROOT, DENOISING_METHODS[method])
    if not os.path.isdir(clean_dir):
        print(f"[ERROR] Clean directory not found: {clean_dir}")
        return

    file_list = discover_set_files(clean_dir, method)
    # Filter to requested tasks
    file_list = [f for f in file_list if f["task"] in tasks]

    if not file_list:
        print(f"[WARN] No .set files found for method={method}, tasks={tasks}")
        return

    print(f"=== Extracting features for method={method}, tasks={tasks} ===")
    print(f"  Found {len(file_list)} .set files")

    # Build channel cluster mapping from config (1-based indices)
    cluster_channels = ELEC_CLUSTS  # dict: name -> list of 1-based indices

    # Process files (parallel or sequential)
    effective_n_jobs = n_jobs if n_jobs != 1 and len(file_list) > 1 else 1
    if effective_n_jobs != 1:
        print(
            f"  Processing {len(file_list)} files in parallel (n_jobs={effective_n_jobs})"
        )
        results = Parallel(n_jobs=effective_n_jobs, verbose=10)(
            delayed(_process_single_file)(
                fi, feature_type, cluster_channels, NOISE_THRESHOLD_UV
            )
            for fi in file_list
        )
    else:
        results = [
            _process_single_file(fi, feature_type, cluster_channels, NOISE_THRESHOLD_UV)
            for fi in file_list
        ]

    # Filter out None results
    results = [r for r in results if r is not None]
    sub_counter = len(results)

    if sub_counter == 0:
        print(f"[WARN] No subjects were successfully processed for method={method}")
        return

    print(f"  Successfully processed {sub_counter} files")

    # Collect data across all subjects from results
    all_data = {
        "stat": {"X": [], "IDs": [], "y_stim": [], "y_task": [], "subject_ids": []},
        "temporal": {"X": [], "IDs": [], "y_stim": [], "y_task": [], "subject_ids": []},
    }

    for result in results:
        for ftype in ["stat", "temporal"]:
            X_key = f"X_{ftype}"
            if X_key in result:
                all_data[ftype]["X"].append(result[X_key])
                all_data[ftype]["IDs"].extend(result["IDs"])
                all_data[ftype]["y_stim"].extend(result["y_stim"])
                all_data[ftype]["y_task"].extend(result["y_task"])
                all_data[ftype]["subject_ids"].extend(result["subject_ids"])

    # Get feature names and metadata from last successful result
    feat_names_stat = results[-1].get("feat_names_stat")
    feat_names_temp = results[-1].get("feat_names_temp")
    times = results[-1].get("times")
    sfreq = results[-1].get("sfreq")

    # Save outputs
    os.makedirs(DATA_OUT_ROOT, exist_ok=True)

    for ftype in ["stat", "temporal"] if feature_type == "both" else [feature_type]:
        if not all_data[ftype]["X"]:
            continue

        X_concat = np.concatenate(all_data[ftype]["X"], axis=0)
        IDs = np.array(all_data[ftype]["IDs"])
        y_stim = np.array(all_data[ftype]["y_stim"])
        y_task = np.array(all_data[ftype]["y_task"])
        subject_ids = np.array(all_data[ftype]["subject_ids"])

        # Create label strings like in VEP code: "0-frequent", "1-rare"
        y_cats = np.where(y_stim == 1, "rare", "frequent")
        yi = y_stim.copy()  # 0 = frequent, 1 = rare

        task_id = get_task_id(tasks)
        time_suffix = "_notime" if ftype == "stat" else ""
        if ftype == "stat":
            feat_names = feat_names_stat
        else:
            feat_names = feat_names_temp

        data_version = get_data_version(method, task_id)
        filename = f"data_{data_version}.npz"
        filepath = os.path.join(DATA_OUT_ROOT, filename)

        np.savez(
            filepath,
            X=X_concat,
            IDs=IDs,
            y=y_cats,
            yi=yi,
            y_task=y_task,
            X_features=feat_names,
            time_info_orig=times if ftype == "temporal" else np.array([0]),
            subject_ids=subject_ids,
            sfreq=sfreq,
            task_set=task_id,
        )
        print(f"  Saved {filepath}")
        print(f"    X shape: {X_concat.shape}")
        print(f"    Task set: {task_id} (tasks: {tasks})")
        print(f"    Labels: {dict(zip(*np.unique(y_cats, return_counts=True)))}")


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Extract ERP features from AMBER cleaned EEG data"
    )
    parser.add_argument(
        "method",
        nargs="?",
        default="all",
        help="Denoising method (RAW, ASR, ICLabel, MARA, GEDAI) or 'all'",
    )
    parser.add_argument(
        "--feature-type",
        choices=["stat", "temporal", "both"],
        default="both",
        help="Type of features to extract (default: both)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None, help="RSVP tasks to include (e.g. X1 X2 X4)"
    )
    parser.add_argument(
        "--recordings",
        nargs="+",
        default=None,
        help="Recording groups to include: 'standard' (X1,X2), "
        "'artifact' (X4,X6,X8), or 'all'. "
        "Can also use individual task codes (e.g. X1 X4). "
        "Overrides --tasks if specified.",
    )
    parser.add_argument(
        "--artifact-conditions",
        nargs="+",
        default=None,
        help="When 'artifact' is in --recordings, which conditions to include: "
        "X4, X6, X8, or any subset. Default: all (X4 X6 X8).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help=(
            "Number of parallel jobs for feature extraction "
            "(-1 = all CPUs, 1 = sequential). Default: from config."
        ),
    )
    args = parser.parse_args()

    # Resolve recordings to task list
    if args.recordings is not None:
        tasks = resolve_recordings(args.recordings, args.artifact_conditions)
        print(f"Resolved recordings to tasks: {tasks}")
    elif args.tasks is not None:
        tasks = [t.upper() for t in args.tasks]
    else:
        tasks = None  # will use RSVP_TASKS default

    if args.method == "all":
        methods = list(DENOISING_METHODS.keys())
    else:
        if args.method not in DENOISING_METHODS:
            print(
                f"Unknown method '{args.method}'. Choose from: {list(DENOISING_METHODS.keys())}"
            )
            return
        methods = [args.method]

    for method in methods:
        extract_features_for_method(
            method, feature_type=args.feature_type, tasks=tasks, n_jobs=args.n_jobs
        )

    print("\n=== Feature extraction complete ===")


if __name__ == "__main__":
    main()
