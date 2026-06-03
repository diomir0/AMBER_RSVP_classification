"""
Plot mean ERPs for each denoising algorithm and each RSVP condition.

The script operates in two modes:

Grand-average mode** (default)
  For every combination of denoising method (RAW, ASR, ICLabel, MARA,
  GEDAI) and RSVP condition (X1, X2, X4, X6, X8), epochs are averaged
  per recording first, then re-averaged per subject, and finally
  grand-averaged across subjects with equal weight.  Mean pre-stimulus
  baseline (−200 ms to 0 ms) is subtracted from each epoch before
  averaging.  The resulting ERP figures show individual trials (light)
  and their mean (bold).

**Single-recording mode** (``--single``)
  Each recording file (e.g. P01-Ss1-X1) is plotted separately so you can
  inspect subject-level variability.  Optionally narrow the selection with
  ``--subject`` and/or ``--session``.

For every figure the script uses MNE dedicated methods:

* ``mne.io.read_epochs_eeglab`` — load the pre-epoched ``.set`` files.
* ``mne.concatenate_epochs`` — pool trials across subjects/sessions (grand-average).
* ``mne.Epochs.average`` — compute the mean ERP (``mne.Evoked``).
* ``mne.Evoked.plot`` — render the mean ERP with channel traces.
* ``mne.Epochs.plot_image`` — render the per-trial GFP image.

Outputs are written to ``erp/figures/``::

    figures/
    ├── RAW/
    │   ├── X1_rare_occipital.png          # grand-average (trials + mean)
    │   ├── X1_rare_occipital_mean.png      # MNE Evoked.plot()
    │   ├── X1_rare_occipital_trials.png    # MNE plot_image (GFP)
    │   ├── ...
    │   ├── summary_occipital.png           # per-method summary grid
    │   └── single/                         # per-recording figures
    │       ├── P01-Ss1-X1_rare_occipital.png
    │       ├── P01-Ss1-X1_rare_occipital_mean.png
    │       └── ...
    └── ...

Usage
-----
    # Default: grand-average for every method and RSVP condition
    python plot_erps.py

    # Restrict to a single denoising method
    python plot_erps.py --method RAW

    # Restrict to a subset of RSVP tasks
    python plot_erps.py --tasks X1 X2

    # Use a different electrode cluster for the channel view
    python plot_erps.py --cluster occipital

    # Pick a specific channel (e.g. POz) instead of averaging a cluster
    python plot_erps.py --picks POz

    # ── Grouped-task mode ──────────────────────────────────────────
    # Combine X1+X2 (standard RSVP) into a single ERP figure
    python plot_erps.py --group-tasks --tasks X1 X2

    # Combine X4+X6+X8 (artifact RSVP) into a single ERP figure
    python plot_erps.py --group-tasks --tasks X4 X6 X8

    # Both groups at once (standard + artifact)
    python plot_erps.py --group-tasks

    # ── Single-recording mode ──────────────────────────────────────
    # Plot every recording individually
    python plot_erps.py --single

    # Only subject P01
    python plot_erps.py --single --subject 1

    # Only P01-Ss2
    python plot_erps.py --single --subject 1 --session 2

    # Single recording + grouped tasks
    python plot_erps.py --single --subject 1 --session 2 --group-tasks --tasks X1 X2
"""

import argparse
import os
import re
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import stats

# Local config (DENOISING_METHODS, RSVP_TASKS, ELEC_CLUSTS, …)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "classification"))
from config import (  # noqa: E402
    DENOISING_COLORS,
    DENOISING_LABELS,
    DENOISING_METHODS,
    ELEC_CLUSTS,
    RSVP_TASKS,
    get_clean_dir,
    get_task_id,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")


# ──────────────────────────────────────────────────────────────────────
# Event decoding
# ──────────────────────────────────────────────────────────────────────

EVENT_LABEL_RARE = "rare"
EVENT_LABEL_FREQ = "frequent"


def _decode_event_mapping(epochs: mne.BaseEpochs) -> dict:
    """Map EEGLAB event names to 'rare' / 'frequent' for the AMBER dataset.

    The pre-epoched ``.set`` files contain exactly one event per epoch,
    labelled '1' (rare) or '2' (frequent). The original label strings can
    be slightly different across pipelines (e.g. ``'R  1'``,
    ``'stimulus_1'``), so we apply a robust matching strategy.
    """
    stim_ids: dict = {}
    for key, val in epochs.event_id.items():
        k_str = str(key).strip()
        first_char = k_str[0] if k_str else ""
        if first_char == "1":
            stim_ids[EVENT_LABEL_RARE] = val
        elif first_char == "2":
            stim_ids[EVENT_LABEL_FREQ] = val
    if EVENT_LABEL_RARE not in stim_ids or EVENT_LABEL_FREQ not in stim_ids:
        raise RuntimeError(
            f"Could not identify rare/frequent events. Found event_id={epochs.event_id}"
        )
    return stim_ids


# ──────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────


def discover_set_files(clean_dir: str, method: str) -> list[dict]:
    """Return metadata dicts for every RSVP ``.set`` file in *clean_dir*."""
    pattern = re.compile(r"P(\d+)-Ss(\d+)-(X\d+)-eeg_" + re.escape(method) + r"\.set$")
    files: list[dict] = []
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
# Loading helpers
# ──────────────────────────────────────────────────────────────────────


def _load_and_relabel_epochs(
    path: str,
    event_id_counter: dict[str, int] | None = None,
) -> mne.Epochs | None:
    """Load a single ``.set`` file, split into rare/frequent, and relabel.

    Parameters
    ----------
    path : str
        Path to an EEGLAB ``.set`` file.
    event_id_counter : dict or None
        Mapping ``{'rare': int, 'frequent': int}`` used to re-label events
        so that epochs from different files share a common event-id space
        (required for :func:`mne.concatenate_epochs`). When *None* the
        default ``{'rare': 1, 'frequent': 2}`` is used.

    Returns
    -------
    mne.Epochs or None
        Epochs object with event_id ``{'rare': ..., 'frequent': ...}``,
        or ``None`` if the file could not be loaded or lacked rare/frequent
        events.
    """
    if event_id_counter is None:
        event_id_counter = {EVENT_LABEL_RARE: 1, EVENT_LABEL_FREQ: 2}

    try:
        ep = mne.io.read_epochs_eeglab(path, verbose="ERROR")
    except Exception as exc:  # pragma: no cover – defensive
        print(f"  [SKIP] {path}: {exc}")
        return None

    try:
        stim_ids = _decode_event_mapping(ep)
    except RuntimeError as exc:
        print(f"  [SKIP] {path}: {exc}")
        return None

    epochs_parts: list[mne.Epochs] = []
    for label in (EVENT_LABEL_RARE, EVENT_LABEL_FREQ):
        try:
            ep_sub = ep[stim_ids[label]].copy()
        except Exception as exc:  # pragma: no cover – defensive
            print(f"  [SKIP] {path} ({label}): {exc}")
            continue
        new_events = np.column_stack(
            [
                np.zeros(len(ep_sub), dtype=int),
                np.zeros(len(ep_sub), dtype=int),
                np.full(len(ep_sub), event_id_counter[label], dtype=int),
            ]
        )
        ep_sub.event_id = {label: event_id_counter[label]}
        ep_sub.events = new_events
        epochs_parts.append(ep_sub)

    if not epochs_parts:
        return None

    if len(epochs_parts) == 1:
        result = epochs_parts[0]
    else:
        result = mne.concatenate_epochs(epochs_parts, verbose="ERROR")

    # Subtract mean pre-stimulus baseline (−200 ms to 0 ms) from each epoch
    result.apply_baseline((None, 0))
    return result


def load_single_recording(
    method: str, subject: int, session: int, task: str
) -> mne.Epochs | None:
    """Load epochs for a single recording (one subject/session/task).

    Returns
    -------
    mne.Epochs or None
        Epochs with event_id ``{'rare': 1, 'frequent': 2}``, or ``None``.
    """
    clean_dir = get_clean_dir(method)
    # Filenames use the method key (e.g. 'GEDAI'), not the folder name
    # (e.g. 'GEDAIv1.7').  This matches discover_set_files() logic.
    pattern = re.compile(
        rf"P{subject:02d}-Ss{session}-{re.escape(task)}-eeg_"
        rf"{re.escape(method)}\.set$"
    )
    matching = [f for f in os.listdir(clean_dir) if pattern.match(f)]
    if not matching:
        print(f"  [WARN] No file for P{subject:02d}-Ss{session}-{task} ({method})")
        return None
    filepath = os.path.join(clean_dir, matching[0])
    print(f"  Loading {os.path.basename(filepath)} …")
    epochs = _load_and_relabel_epochs(filepath)
    if epochs is not None:
        n_rare = sum(epochs.events[:, 2] == epochs.event_id.get(EVENT_LABEL_RARE, -1))
        n_freq = sum(epochs.events[:, 2] == epochs.event_id.get(EVENT_LABEL_FREQ, -1))
        print(f"    {len(epochs)} epochs (rare={n_rare}, frequent={n_freq})")
    return epochs


def load_epochs_for_condition(
    method: str, task: str, subject_filter: list[int] | None = None
) -> dict | None:
    """Load Epochs for *method* / *task*, organised per subject.

    For each subject, per-recording Epochs are kept as a list so that
    :func:`_compute_grand_average` can average by recording first, then
    by subject, giving equal weight to each recording regardless of
    trial count.

    Parameters
    ----------
    subject_filter : list of int or None
        If provided, only include these subject IDs.  ``None`` loads all.

    Returns
    -------
    dict or None
        ``{'subjects': {subject_id: [mne.Epochs, ...], ...},
           'combined': mne.Epochs,
           'n_subjects': int}``
        or ``None`` if no files could be loaded.
    """
    clean_dir = get_clean_dir(method)
    files = [f for f in discover_set_files(clean_dir, method) if f["task"] == task]
    if subject_filter is not None:
        files = [f for f in files if f["subject"] in subject_filter]
    if not files:
        print(f"  [WARN] No .set files for method={method}, task={task}")
        return None

    event_id_counter = {EVENT_LABEL_RARE: 1, EVENT_LABEL_FREQ: 2}
    subject_epochs: dict[int, list[mne.Epochs]] = {}
    n_skipped = 0

    for fi in files:
        ep = _load_and_relabel_epochs(fi["path"], event_id_counter)
        if ep is None:
            n_skipped += 1
            continue
        subject_epochs.setdefault(fi["subject"], []).append(ep)

    if not subject_epochs:
        print(f"  [WARN] No epochs could be loaded for method={method}, task={task}")
        return None

    # Keep per-recording Epochs per subject for proper averaging
    # (average by recording first, then by subject)
    subjects: dict[int, list[mne.Epochs]] = {}
    all_parts: list[mne.Epochs] = []
    for sub_id, ep_list in sorted(subject_epochs.items()):
        subjects[sub_id] = ep_list
        all_parts.extend(ep_list)

    # Combined epochs for trial-level visualisation
    combined = (
        all_parts[0]
        if len(all_parts) == 1
        else mne.concatenate_epochs(all_parts, verbose="ERROR")
    )
    n_rare = sum(combined.events[:, 2] == event_id_counter[EVENT_LABEL_RARE])
    n_freq = sum(combined.events[:, 2] == event_id_counter[EVENT_LABEL_FREQ])
    print(
        f"  Loaded {len(combined)} epochs "
        f"(rare={n_rare}, frequent={n_freq}) from "
        f"{len(subjects)} subjects ({n_skipped} files skipped)"
    )
    return {"subjects": subjects, "combined": combined, "n_subjects": len(subjects)}


def load_epochs_for_task_group(
    method: str, tasks: list[str], subject_filter: list[int] | None = None
) -> dict | None:
    """Load Epochs across multiple RSVP tasks, organised per subject.

    Same structure as :func:`load_epochs_for_condition` but files from
    all tasks in *tasks* are included.

    Parameters
    ----------
    subject_filter : list of int or None
        If provided, only include these subject IDs.  ``None`` loads all.

    Returns
    -------
    dict or None
        ``{'subjects': {subject_id: [mne.Epochs, ...], ...},
           'combined': mne.Epochs,
           'n_subjects': int}``
    """
    clean_dir = get_clean_dir(method)
    files = [f for f in discover_set_files(clean_dir, method) if f["task"] in tasks]
    if subject_filter is not None:
        files = [f for f in files if f["subject"] in subject_filter]
    if not files:
        print(f"  [WARN] No .set files for method={method}, tasks={tasks}")
        return None

    event_id_counter = {EVENT_LABEL_RARE: 1, EVENT_LABEL_FREQ: 2}
    subject_epochs: dict[int, list[mne.Epochs]] = {}
    n_skipped = 0

    for fi in files:
        ep = _load_and_relabel_epochs(fi["path"], event_id_counter)
        if ep is None:
            n_skipped += 1
            continue
        subject_epochs.setdefault(fi["subject"], []).append(ep)

    if not subject_epochs:
        print(f"  [WARN] No epochs could be loaded for method={method}, tasks={tasks}")
        return None

    # Keep per-recording Epochs per subject for proper averaging
    # (average by recording first, then by subject)
    subjects: dict[int, list[mne.Epochs]] = {}
    all_parts: list[mne.Epochs] = []
    for sub_id, ep_list in sorted(subject_epochs.items()):
        subjects[sub_id] = ep_list
        all_parts.extend(ep_list)

    combined = (
        all_parts[0]
        if len(all_parts) == 1
        else mne.concatenate_epochs(all_parts, verbose="ERROR")
    )
    n_rare = sum(combined.events[:, 2] == event_id_counter[EVENT_LABEL_RARE])
    n_freq = sum(combined.events[:, 2] == event_id_counter[EVENT_LABEL_FREQ])
    task_label = get_task_id(tasks)
    print(
        f"  Loaded {len(combined)} epochs "
        f"(rare={n_rare}, frequent={n_freq}) from "
        f"{len(subjects)} subjects ({n_skipped} files skipped) "
        f"[task group: {task_label}]"
    )
    return {"subjects": subjects, "combined": combined, "n_subjects": len(subjects)}


# ──────────────────────────────────────────────────────────────────────
# Plotting helpers (MNE dedicated methods)
# ──────────────────────────────────────────────────────────────────────


def _resolve_picks(
    epochs: mne.BaseEpochs, cluster: str, picks: str | None
) -> list[str]:
    """Resolve ``picks`` to a list of channel names.

    * ``picks`` is a channel name (e.g. ``"POz"``): use it directly.
    * ``cluster`` is the name of a region in :data:`config.ELEC_CLUSTS`:
      average the matching channels (1-based indices, see config).
    """
    ch_names = epochs.info["ch_names"]
    if picks is not None:
        if picks not in ch_names:
            raise ValueError(
                f"Channel '{picks}' not found. Available: {ch_names[:5]} ..."
            )
        return [picks]

    indices_1b = ELEC_CLUSTS[cluster]  # 1-based indices
    picks_names = [ch_names[i - 1] for i in indices_1b if i - 1 < len(ch_names)]
    if not picks_names:
        raise ValueError(
            f"No channels matched cluster '{cluster}'. Available: {ch_names}"
        )
    return picks_names


def _pick_label(cluster: str, picks: str | None) -> str:
    """Return the label to use in filenames: channel name or cluster name."""
    if picks is not None:
        return picks
    return cluster


def _compute_grand_average(
    subject_recordings: dict[int, list[mne.Epochs]],
    event_label: str,
) -> mne.Evoked | None:
    """Compute a proper grand-average ERP across subjects.

    Averaging hierarchy (ensures each recording contributes equally):

    1. For each recording (one ``.set`` file), average its epochs into
       one Evoked per recording.
    2. For each subject, average the recording-level Evokeds with equal
       weight via ``mne.combine_evoked(weights='equal')``, so each
       recording contributes equally regardless of trial count.
    3. Call ``mne.grand_average()`` to average across subjects with
       equal weight.

    All epochs are baseline-corrected (mean pre-stimulus subtracted)
    before averaging, as applied in :func:`_load_and_relabel_epochs`.

    Parameters
    ----------
    subject_recordings : dict
        ``{subject_id: [mne.Epochs, ...], ...}`` — one list of
        per-recording Epochs objects per subject (each recording = one
        session file).
    event_label : str
        Event type to average (``'rare'`` or ``'frequent'``).

    Returns
    -------
    mne.Evoked or None
        Grand-average Evoked, or None if no subject had this event.
    """
    sub_evokeds: list[mne.Evoked] = []
    for sub_id, rec_list in sorted(subject_recordings.items()):
        rec_evokeds: list[mne.Evoked] = []
        for ep in rec_list:
            if event_label not in ep.event_id:
                continue
            rec_ev = ep[event_label].average(picks="all")
            rec_evokeds.append(rec_ev)
        if not rec_evokeds:
            continue
        # Average recording-level Evokeds per subject with equal weight
        if len(rec_evokeds) == 1:
            sub_ev = rec_evokeds[0]
        else:
            sub_ev = mne.combine_evoked(rec_evokeds, weights="equal")
        sub_ev.comment = f"Sub {sub_id:02d} {event_label}"
        sub_evokeds.append(sub_ev)

    if not sub_evokeds:
        return None
    if len(sub_evokeds) == 1:
        return sub_evokeds[0]
    return mne.grand_average(sub_evokeds)


def _adaptive_clim(data: np.ndarray, pct_lo: float = 1.0, pct_hi: float = 99.0):
    """Return (vmin, vmax) in volts that adapt to the data distribution.

    Uses percentiles instead of the absolute peak so that a few outlier
    trials do not compress the colour scale.  The limits are rounded
    outward to the nearest µV to keep the colourbar ticks clean.

    Parameters
    ----------
    data : ndarray
        Trial data in volts (any shape — will be flattened).
    pct_lo, pct_hi : float
        Lower and upper percentiles (0–100) for clipping outliers.

    Returns
    -------
    (vmin, vmax) in volts.
    """
    if not data.size:
        return -1e-6, 1e-6
    flat = data.ravel()
    lo = float(np.nanpercentile(flat, pct_lo))
    hi = float(np.nanpercentile(flat, pct_hi))
    if lo == hi:
        lo, hi = lo - 1e-6, hi + 1e-6
    # Round outward to the nearest µV for clean colourbar ticks
    lo_uv = np.floor(lo * 1e6)
    hi_uv = np.ceil(hi * 1e6)
    return lo_uv, hi_uv


def _p300_ttest(
    subject_recordings: dict[int, list[mne.Epochs]],
    picks: list[str],
    alpha: float = 0.05,
    p300_window: tuple[float, float] = (200.0, 400.0),
) -> dict | None:
    """Paired t-test on mean P300 amplitude (rare vs frequent).

    For each subject, the per-recording evokeds are averaged (equal weight)
    into a subject-level ERP for both conditions.  The mean amplitude in
    the P300 window (default 200–400 ms) is then computed per subject, and
    a single paired t-test is run across subjects.  No multiple-comparisons
    correction is needed because only one hypothesis is tested.

    Parameters
    ----------
    subject_recordings : dict
        ``{subject_id: [mne.Epochs, ...], ...}`` — one list of
        per-recording Epochs objects per subject.
    picks : list[str]
        Channel names to include in the test.
    alpha : float
        Significance level (default 0.05).
    p300_window : tuple of float
        Start and end of the P300 time window in ms (default 200–400 ms).

    Returns
    -------
    dict or None
        ``{'t': float, 'p': float, 'significant': bool,
           'mean_rare_uv': float, 'mean_freq_uv': float,
           'mean_diff_uv': float, 'n_subjects': int,
           'window_ms': tuple}``
        or ``None`` if fewer than 2 subjects have both conditions.
    """
    # Build per-subject evokeds for rare and frequent
    sub_rare: list[mne.Evoked] = []
    sub_freq: list[mne.Evoked] = []

    for sub_id, rec_list in sorted(subject_recordings.items()):
        rec_rare: list[mne.Evoked] = []
        rec_freq: list[mne.Evoked] = []
        for ep in rec_list:
            if EVENT_LABEL_RARE in ep.event_id:
                rec_rare.append(ep[EVENT_LABEL_RARE].average(picks="all"))
            if EVENT_LABEL_FREQ in ep.event_id:
                rec_freq.append(ep[EVENT_LABEL_FREQ].average(picks="all"))
        if not rec_rare or not rec_freq:
            continue  # subject must have both conditions
        # Average recordings with equal weight
        sub_rare_ev = (
            rec_rare[0]
            if len(rec_rare) == 1
            else mne.combine_evoked(rec_rare, weights="equal")
        )
        sub_freq_ev = (
            rec_freq[0]
            if len(rec_freq) == 1
            else mne.combine_evoked(rec_freq, weights="equal")
        )
        sub_rare.append(sub_rare_ev)
        sub_freq.append(sub_freq_ev)

    n_subs = len(sub_rare)
    if n_subs < 2:
        return None  # need at least 2 subjects for a t-test

    win_lo, win_hi = float(p300_window[0]), float(p300_window[1])

    # Compute mean amplitude in P300 window for each subject
    amp_rare = []
    amp_freq = []
    for ev_r, ev_f in zip(sub_rare, sub_freq):
        data_r = ev_r.get_data(picks=picks)  # (n_ch, n_times)
        data_f = ev_f.get_data(picks=picks)
        times_ms = ev_r.times * 1000.0
        win_mask = (times_ms >= win_lo) & (times_ms <= win_hi)
        # Average across both channels and time within the window
        amp_rare.append(float(data_r[:, win_mask].mean()))
        amp_freq.append(float(data_f[:, win_mask].mean()))

    amp_rare_arr = np.array(amp_rare)  # (n_subs,) in V
    amp_freq_arr = np.array(amp_freq)
    diff_arr = amp_rare_arr - amp_freq_arr

    # Paired t-test
    t_val, p_val = stats.ttest_rel(amp_rare_arr, amp_freq_arr)
    significant = bool(p_val < alpha)

    return {
        "t": float(t_val),
        "p": float(p_val),
        "significant": significant,
        "mean_rare_uv": float(amp_rare_arr.mean()) * 1e6,
        "mean_freq_uv": float(amp_freq_arr.mean()) * 1e6,
        "mean_diff_uv": float(diff_arr.mean()) * 1e6,
        "n_subjects": n_subs,
        "window_ms": (win_lo, win_hi),
    }


def _print_ttest_results(result: dict, context_label: str) -> None:
    """Print a readable summary of the P300 paired t-test to stdout."""
    n_subs = result["n_subjects"]
    win_lo, win_hi = result["window_ms"]
    direction = "rare > freq" if result["t"] > 0 else "freq > rare"
    sig_marker = "*" if result["significant"] else ""

    print(
        f"\n  ── Paired t-test (P300 {win_lo:.0f}-{win_hi:.0f} ms): {context_label} ──"
    )
    print(f"     Subjects:           {n_subs}")
    print(f"     Mean rare amp:      {result['mean_rare_uv']:+.2f} \u00b5V")
    print(f"     Mean freq amp:      {result['mean_freq_uv']:+.2f} \u00b5V")
    print(f"     Mean diff (r-f):    {result['mean_diff_uv']:+.2f} \u00b5V")
    print(
        f"     t({n_subs - 1}) = {result['t']:+.2f},  p = {result['p']:.4f}  {sig_marker}"
    )
    if result["significant"]:
        print(f"     Significant (p < 0.05): {direction}")
    else:
        print(f"     Not significant (p >= 0.05)")


def _shade_p300_window(
    ax,
    ttest_result: dict,
    color: str = "#FFD700",
    shade_alpha: float = 0.25,
):
    """Shade the P300 window on an axes if the t-test is significant."""
    if not ttest_result.get("significant", False):
        return
    win_lo, win_hi = ttest_result["window_ms"]
    ax.axvspan(
        win_lo,
        win_hi,
        color=color,
        alpha=shade_alpha,
        label=f"p = {ttest_result['p']:.3f} (P300 sig.)",
        zorder=0,
    )


def _mean_ylim(mean_uv: np.ndarray, margin_factor: float = 0.35) -> tuple[float, float]:
    """Compute y-axis limits (µV) that tightly frame the *mean* ERP.

    Parameters
    ----------
    mean_uv : 1-D array
        Mean ERP amplitudes in µV.
    margin_factor : float
        Fraction of the mean range to add as padding above/below.

    Returns
    -------
    (ymin, ymax) in µV.
    """
    lo = float(np.nanmin(mean_uv))
    hi = float(np.nanmax(mean_uv))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    span = hi - lo
    return lo - margin_factor * span, hi + margin_factor * span


def _resize_figs(figs, width: float, height: float) -> None:
    """Resize one figure or a list of figures (handles MNE return types)."""
    if figs is None:
        return
    if isinstance(figs, (list, tuple)):
        for f in figs:
            _resize_figs(f, width, height)
        return
    try:
        figs.set_size_inches(width, height)
    except Exception:
        pass


def _save_first_fig(figs, path: str) -> None:
    """Save a single figure or the first figure of a list/tuple."""
    if figs is None:
        return
    if isinstance(figs, (list, tuple)):
        if not figs:
            return
        fig = figs[0]
    else:
        fig = figs
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _plot_trials_with_mean(
    epochs: mne.Epochs,
    event_label: str,
    picks: list[str],
    fig_path: str,
    title: str,
    grand_evoked: mne.Evoked | None = None,
    ttest_result: dict | None = None,
):
    """Plot individual trials and the mean ERP using MNE's helpers.

    Parameters
    ----------
    epochs : mne.Epochs
        All trials (concatenated across subjects) — used for the
        trial-level background traces and ``plot_image``.
    event_label : str
        ``'rare'`` or ``'frequent'``.
    picks : list[str]
        Channel names to plot.
    fig_path : str
        Output path for the combined figure.
    title : str
        Figure title.
    grand_evoked : mne.Evoked or None
        Pre-computed grand-average Evoked (from
        :func:`_compute_grand_average`).  If None, a naive average across
        all trials is used as fallback.
    ttest_result : dict or None
        Result dict from :func:`_p300_ttest`.  If provided,
        the P300 window is shaded in gold when the t-test is
        significant.
    """
    # Use the proper grand average if available; otherwise fall back
    # to the naive trial-level average.
    if grand_evoked is not None:
        evoked = grand_evoked
    else:
        evoked = epochs[event_label].average(picks="all")

    # ── Mean ERP via MNE's Evoked.plot() ─────────────────────────────
    fig_mean = evoked.plot(
        picks=picks,
        titles=dict(title=f"Mean ERP — {title}"),
        show=False,
        time_unit="ms",
        spatial_colors=True,
        selectable=True,
    )
    fig_mean_path = fig_path.replace(".png", "_mean.png")
    try:
        fig_mean.savefig(fig_mean_path, dpi=150, bbox_inches="tight")
    except Exception:
        pass  # Some matplotlib backends do not support savefig on MNE figs
    plt.close("all")

    # ── Individual trials via plot_image() ─────────────────────────
    trial_data = epochs[event_label].get_data(picks=picks)  # (n_trials, n_ch, n_times)
    # `combine='gfp'` requires more than one channel; fall back to the
    # default (mean across channels) for single-channel picks.
    combine = "gfp" if len(picks) > 1 else None

    # Derive colour limits from the *actual displayed values* (GFP or
    # single-channel), not from the raw multi-channel data, because GFP
    # is always non-negative and lives in a completely different range.
    if combine == "gfp":
        displayed = np.std(trial_data, axis=1)  # (n_trials, n_times)
    else:
        displayed = trial_data[:, 0, :]  # (n_trials, n_times)
    clim_lo, clim_hi = _adaptive_clim(displayed)

    fig_trials = epochs[event_label].plot_image(
        picks=picks,
        combine=combine,
        show=False,
        vmin=clim_lo,
        vmax=clim_hi,
        title=f"Individual trials — {title}",
    )
    # `plot_image` returns a list of figures (one per picked channel in
    # some MNE versions, a single Figure in others). Resize them all
    # uniformly.
    _resize_figs(fig_trials, width=8.0, height=2.5 * max(1, len(picks)))
    fig_trials_path = fig_path.replace(".png", "_trials.png")
    try:
        _save_first_fig(fig_trials, fig_trials_path)
    except Exception:
        pass
    plt.close("all")

    # ── Combined summary figure (matplotlib) ─────────────────────────
    # Individual trials from concatenated epochs (background), and the
    # grand-average ERP from _compute_grand_average (bold line).
    data = epochs[event_label].get_data(picks=picks)  # (n_trials, n_ch, n_times)
    times_ms = epochs[event_label].times * 1000.0
    n_ch = len(picks)

    fig, axes = plt.subplots(
        n_ch,
        1,
        figsize=(8, 2.0 * n_ch),
        sharex=True,
        constrained_layout=True,
    )
    if n_ch == 1:
        axes = [axes]

    # Use grand-average Evoked data for the bold mean line
    if grand_evoked is not None:
        evoked_data = grand_evoked.get_data(picks=picks)  # (n_ch, n_times)
    else:
        evoked_data = np.mean(data, axis=0)  # fallback: naive mean across trials

    for ch_idx, (ax, ch_name) in enumerate(zip(axes, picks)):
        # Individual trials (light)
        for trial in data[:, ch_idx, :]:
            ax.plot(times_ms, trial * 1e6, color="lightgray", linewidth=0.4, alpha=0.6)
        # Mean ERP (bold)
        ax.plot(
            times_ms,
            evoked_data[ch_idx] * 1e6,
            color=DENOISING_COLORS.get("RAW", "#d62728"),
            linewidth=2.0,
            label="Mean",
        )
        ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
        ax.axvline(0.0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel(f"{ch_name}\n(µV)")
        ax.set_title(f"{ch_name}", loc="left", fontsize=9)
        # Scale y-axis to the mean ERP range so the bold trace is
        # clearly visible; individual trials may clip outside.
        ylim = _mean_ylim(evoked_data[ch_idx] * 1e6)
        ax.set_ylim(ylim)
        # Shade P300 window if t-test is significant
        if ttest_result is not None:
            _shade_p300_window(ax, ttest_result)
        if ch_idx == 0:
            ax.legend(loc="upper right", fontsize=8)
        if ch_idx == n_ch - 1:
            ax.set_xlabel("Time (ms)")

    fig.suptitle(f"{title} — {event_label}", fontsize=12)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Single-recording plotting
# ──────────────────────────────────────────────────────────────────────


def plot_single_recording(
    method: str,
    subject: int,
    session: int,
    task: str,
    output_dir: str,
    cluster: str,
    picks: str | None,
):
    """Plot ERP figures for a single subject/session/task recording."""
    rec_label = f"P{subject:02d}-Ss{session}-{task}"
    print(f"\n=== {DENOISING_LABELS.get(method, method)} — {rec_label} (single) ===")

    epochs = load_single_recording(method, subject, session, task)
    if epochs is None:
        return False

    try:
        picks_names = _resolve_picks(epochs, cluster, picks)
    except ValueError as exc:
        print(f"  [SKIP] {exc}")
        return False

    single_dir = os.path.join(output_dir, method, "single")
    os.makedirs(single_dir, exist_ok=True)

    for event_label in (EVENT_LABEL_RARE, EVENT_LABEL_FREQ):
        if event_label not in epochs.event_id:
            print(f"  [WARN] No {event_label} trials for {rec_label} — skipping")
            continue
        n_trials = sum(epochs.events[:, 2] == epochs.event_id[event_label])
        if n_trials == 0:
            print(f"  [WARN] No {event_label} trials for {rec_label} — skipping")
            continue
        title = (
            f"{DENOISING_LABELS.get(method, method)} — {rec_label} — "
            f"{event_label.capitalize()} (n={n_trials})"
        )
        pl = _pick_label(cluster, picks)
        fig_path = os.path.join(single_dir, f"{rec_label}_{event_label}_{pl}.png")
        print(
            f"  Plotting {n_trials} {event_label} trials "
            f"on {picks_names} → {os.path.relpath(fig_path)}"
        )
        _plot_trials_with_mean(epochs, event_label, picks_names, fig_path, title)

    return True


def plot_single_summary(
    method: str,
    subject: int,
    session: int,
    output_dir: str,
    cluster: str,
    picks: str | None,
    tasks: list[str],
):
    """Build a combined summary figure for one subject/session across tasks."""
    sub_label = f"P{subject:02d}-Ss{session}"
    single_dir = os.path.join(output_dir, method, "single")
    os.makedirs(single_dir, exist_ok=True)
    summary_path = os.path.join(
        single_dir, f"{sub_label}_summary_{_pick_label(cluster, picks)}.png"
    )

    n_tasks = len(tasks)
    fig, axes = plt.subplots(
        n_tasks,
        2,
        figsize=(10, 2.2 * n_tasks),
        sharex=True,
        constrained_layout=True,
    )
    if n_tasks == 1:
        axes = np.array([axes])

    any_plotted = False
    for row, task in enumerate(tasks):
        epochs = load_single_recording(method, subject, session, task)
        if epochs is None:
            for col in range(2):
                axes[row, col].set_title(f"{task} — (no data)", fontsize=9)
                axes[row, col].axis("off")
            continue
        try:
            picks_names = _resolve_picks(epochs, cluster, picks)
        except ValueError:
            for col in range(2):
                axes[row, col].axis("off")
            continue
        ch_name = picks_names[0]
        for col, event_label in enumerate((EVENT_LABEL_RARE, EVENT_LABEL_FREQ)):
            ax = axes[row, col]
            if event_label not in epochs.event_id:
                ax.axis("off")
                continue
            data = epochs[event_label].get_data(picks=[ch_name])[:, 0, :]
            times_ms = epochs[event_label].times * 1000.0
            for trial in data:
                ax.plot(
                    times_ms, trial * 1e6, color="lightgray", linewidth=0.4, alpha=0.6
                )
            mean = data.mean(axis=0) * 1e6
            color = "#d62728" if event_label == EVENT_LABEL_RARE else "#1f77b4"
            ax.plot(
                times_ms,
                mean,
                color=color,
                linewidth=2.0,
                label=f"Mean (n={data.shape[0]})",
            )
            ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
            ax.axvline(0.0, color="black", linewidth=0.5, linestyle="--")
            ax.set_title(f"{task} — {event_label} ({ch_name})", fontsize=10)
            if col == 0:
                ax.set_ylabel("Amplitude (µV)")
            if row == n_tasks - 1:
                ax.set_xlabel("Time (ms)")
            # Scale y-axis to the mean ERP range so the bold trace
            # is clearly visible; individual trials may clip outside.
            ax.set_ylim(_mean_ylim(mean))
            ax.legend(loc="upper right", fontsize=7)
            any_plotted = True

    fig.suptitle(
        f"{DENOISING_LABELS.get(method, method)} — {sub_label} (single recording)",
        fontsize=13,
    )
    if any_plotted:
        fig.savefig(summary_path, dpi=150, bbox_inches="tight")
        print(f"  Single-recording summary saved to {os.path.relpath(summary_path)}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Grouped-task plotting (multi-task ERP)
# ──────────────────────────────────────────────────────────────────────


def plot_grouped_tasks(
    method: str,
    tasks: list[str],
    output_dir: str,
    cluster: str,
    picks: str | None,
    subject_filter: list[int] | None = None,
) -> dict | bool | None:
    """Plot ERPs for all tasks combined into a single grand-average.

    This concatenates epochs across the requested RSVP tasks (e.g.
    X1+X2 for "standard", or X4+X6+X8 for "artifact") and produces
    one set of figures per event type (rare/frequent).
    """
    task_label = get_task_id(tasks)
    task_str = "+".join(tasks)
    print(
        f"\n=== {DENOISING_LABELS.get(method, method)} — Grouped tasks [{task_str}] ==="
    )
    epochs = load_epochs_for_task_group(method, tasks, subject_filter=subject_filter)
    if epochs is None:
        return False

    combined = epochs["combined"]
    subjects = epochs["subjects"]

    try:
        picks_names = _resolve_picks(combined, cluster, picks)
    except ValueError as exc:
        print(f"  [SKIP] {exc}")
        return False

    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)

    # P300 paired t-test (rare vs frequent) -- computed once
    # before the plotting loop so the result can be passed to figures
    ttest_result = _p300_ttest(subjects, picks_names)
    if ttest_result is not None:
        _print_ttest_results(
            ttest_result,
            f"{DENOISING_LABELS.get(method, method)} -- {task_str} ({picks_names})",
        )
    else:
        print(f"  [SKIP] t-test not possible (need ≥2 subjects with both conditions)")

    for event_label in (EVENT_LABEL_RARE, EVENT_LABEL_FREQ):
        n_trials = sum(combined.events[:, 2] == combined.event_id[event_label])
        if n_trials == 0:
            print(f"  [WARN] No {event_label} trials — skipping")
            continue
        grand = _compute_grand_average(subjects, event_label)
        n_subs = epochs["n_subjects"]
        title = (
            f"{DENOISING_LABELS.get(method, method)} — {task_str} — "
            f"{event_label.capitalize()} (n_trials={n_trials}, n_subs={n_subs})"
        )
        fig_path = os.path.join(
            method_dir, f"{task_label}_{event_label}_{_pick_label(cluster, picks)}.png"
        )
        print(
            f"  Plotting {n_trials} {event_label} trials (grouped) "
            f"on {picks_names} → {os.path.relpath(fig_path)}"
        )
        _plot_trials_with_mean(
            combined,
            event_label,
            picks_names,
            fig_path,
            title,
            grand_evoked=grand,
            ttest_result=ttest_result,
        )

    return ttest_result  # dict or None (truthy on success)


def plot_grouped_summary(
    method: str,
    task_groups: dict[str, list[str]],
    output_dir: str,
    cluster: str,
    picks: str | None,
    subject_filter: list[int] | None = None,
    ttest_results: dict | None = None,
):
    """Build a combined rare/frequent summary across task groups.

    *task_groups* maps a human-readable label to a list of task codes,
    e.g. {'standard': ['X1', 'X2'], 'artifact': ['X4', 'X6', 'X8']}.
    Each group becomes a row in the summary grid.

    Parameters
    ----------
    ttest_results : dict or None
        Pre-computed t-test results keyed by group label,
        as returned by :func:`_p300_ttest`.  Avoids re-running the t-test.
    """
    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)
    group_labels = list(task_groups.keys())
    summary_path = os.path.join(
        method_dir, f"grouped_summary_{_pick_label(cluster, picks)}.png"
    )

    n_groups = len(task_groups)
    fig, axes = plt.subplots(
        n_groups,
        2,
        figsize=(10, 2.2 * n_groups),
        sharex=True,
        constrained_layout=True,
    )
    if n_groups == 1:
        axes = np.array([axes])

    any_plotted = False
    for row, label in enumerate(group_labels):
        tasks = task_groups[label]
        result = load_epochs_for_task_group(
            method, tasks, subject_filter=subject_filter
        )
        if result is None:
            for col in range(2):
                axes[row, col].set_title(f"{label} — (no data)", fontsize=9)
                axes[row, col].axis("off")
            continue
        combined = result["combined"]
        subjects = result["subjects"]
        try:
            picks_names = _resolve_picks(combined, cluster, picks)
        except ValueError:
            for col in range(2):
                axes[row, col].axis("off")
            continue
        ch_name = picks_names[0]

        # Look up pre-computed t-test result for this group
        ttest_result = ttest_results.get(label) if ttest_results is not None else None

        for col, event_label in enumerate((EVENT_LABEL_RARE, EVENT_LABEL_FREQ)):
            ax = axes[row, col]
            data = combined[event_label].get_data(picks=[ch_name])[:, 0, :]
            times_ms = combined[event_label].times * 1000.0
            for trial in data:
                ax.plot(
                    times_ms, trial * 1e6, color="lightgray", linewidth=0.4, alpha=0.6
                )
            # Grand-average ERP (proper per-subject averaging)
            grand = _compute_grand_average(subjects, event_label)
            if grand is not None:
                mean = grand.get_data(picks=[ch_name])[0] * 1e6
            else:
                mean = data.mean(axis=0) * 1e6
            color = "#d62728" if event_label == EVENT_LABEL_RARE else "#1f77b4"
            ax.plot(
                times_ms,
                mean,
                color=color,
                linewidth=2.0,
                label=f"Mean (n={data.shape[0]})",
            )
            ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
            ax.axvline(0.0, color="black", linewidth=0.5, linestyle="--")
            ax.set_title(f"{label} — {event_label} ({ch_name})", fontsize=10)
            if col == 0:
                ax.set_ylabel("Amplitude (µV)")
            if row == n_groups - 1:
                ax.set_xlabel("Time (ms)")
            ylim = _mean_ylim(mean)
            ax.set_ylim(ylim)
            # Shade P300 window if t-test is significant
            if ttest_result is not None:
                _shade_p300_window(ax, ttest_result)
            ax.legend(loc="upper right", fontsize=7)
            any_plotted = True

    fig.suptitle(
        f"{DENOISING_LABELS.get(method, method)} — Grouped ERPs",
        fontsize=13,
    )
    if any_plotted:
        fig.savefig(summary_path, dpi=150, bbox_inches="tight")
        print(f"  Grouped summary saved to {os.path.relpath(summary_path)}")
    plt.close(fig)


def plot_single_grouped(
    method: str,
    subject: int,
    session: int,
    tasks: list[str],
    output_dir: str,
    cluster: str,
    picks: str | None,
):
    """Plot ERPs for a single recording, grouping across tasks."""
    task_label = get_task_id(tasks)
    task_str = "+".join(tasks)
    sub_label = f"P{subject:02d}-Ss{session}"
    print(
        f"\n=== {DENOISING_LABELS.get(method, method)} — {sub_label} "
        f"[{task_str}] (single grouped) ==="
    )

    # Load and concatenate epochs across tasks for this subject/session
    epochs_list: list[mne.Epochs] = []
    event_id_counter = {EVENT_LABEL_RARE: 1, EVENT_LABEL_FREQ: 2}
    n_loaded = 0
    for task in tasks:
        ep = load_single_recording(method, subject, session, task)
        if ep is not None:
            epochs_list.append(ep)
            n_loaded += 1

    if not epochs_list:
        print(f"  [WARN] No epochs for {sub_label} [{task_str}]")
        return False

    if len(epochs_list) == 1:
        combined = epochs_list[0]
    else:
        combined = mne.concatenate_epochs(epochs_list, verbose="ERROR")

    try:
        picks_names = _resolve_picks(combined, cluster, picks)
    except ValueError as exc:
        print(f"  [SKIP] {exc}")
        return False

    single_dir = os.path.join(output_dir, method, "single")
    os.makedirs(single_dir, exist_ok=True)

    for event_label in (EVENT_LABEL_RARE, EVENT_LABEL_FREQ):
        if event_label not in combined.event_id:
            continue
        n_trials = sum(combined.events[:, 2] == combined.event_id[event_label])
        if n_trials == 0:
            continue
        title = (
            f"{DENOISING_LABELS.get(method, method)} — {sub_label} [{task_str}] — "
            f"{event_label.capitalize()} (n={n_trials})"
        )
        fig_path = os.path.join(
            single_dir,
            f"{sub_label}_{task_label}_{event_label}_{_pick_label(cluster, picks)}.png",
        )
        print(
            f"  Plotting {n_trials} {event_label} trials (grouped) "
            f"on {picks_names} → {os.path.relpath(fig_path)}"
        )
        _plot_trials_with_mean(combined, event_label, picks_names, fig_path, title)

    return True


# ──────────────────────────────────────────────────────────────────────
# Grand-average driver
# ──────────────────────────────────────────────────────────────────────


def plot_method_task(
    method: str,
    task: str,
    output_dir: str,
    cluster: str,
    picks: str | None,
    subject_filter: list[int] | None = None,
) -> dict | bool | None:
    """Generate the rare/frequent ERP figures for one method and task.

    Returns the t-test result dict on success, or False on failure.
    """
    print(f"\n=== {DENOISING_LABELS.get(method, method)} — Task {task} ===")
    result = load_epochs_for_condition(method, task, subject_filter=subject_filter)
    if result is None:
        return False

    combined = result["combined"]
    subjects = result["subjects"]

    try:
        picks_names = _resolve_picks(combined, cluster, picks)
    except ValueError as exc:
        print(f"  [SKIP] {exc}")
        return False

    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)

    # P300 paired t-test (rare vs frequent) -- computed once
    # before the plotting loop so the result can be passed to figures
    ttest_result = _p300_ttest(subjects, picks_names)
    if ttest_result is not None:
        _print_ttest_results(
            ttest_result,
            f"{DENOISING_LABELS.get(method, method)} -- {task} ({picks_names})",
        )
    else:
        print(f"  [SKIP] t-test not possible (need ≥2 subjects with both conditions)")

    for event_label in (EVENT_LABEL_RARE, EVENT_LABEL_FREQ):
        n_trials = sum(combined.events[:, 2] == combined.event_id[event_label])
        if n_trials == 0:
            print(f"  [WARN] No {event_label} trials — skipping")
            continue
        grand = _compute_grand_average(subjects, event_label)
        n_subs = result["n_subjects"]
        title = (
            f"{DENOISING_LABELS.get(method, method)} — {task} — "
            f"{event_label.capitalize()} (n_trials={n_trials}, n_subs={n_subs})"
        )
        fig_path = os.path.join(
            method_dir, f"{task}_{event_label}_{_pick_label(cluster, picks)}.png"
        )
        print(
            f"  Plotting {n_trials} {event_label} trials "
            f"on {picks_names} → {os.path.relpath(fig_path)}"
        )
        _plot_trials_with_mean(
            combined,
            event_label,
            picks_names,
            fig_path,
            title,
            grand_evoked=grand,
            ttest_result=ttest_result,
        )

    return ttest_result  # dict or None (truthy on success)


def plot_method_summary(
    method: str,
    output_dir: str,
    cluster: str,
    picks: str | None,
    tasks: list[str],
    subject_filter: list[int] | None = None,
    ttest_results: dict | None = None,
):
    """Build a combined rare/frequent ERPs figure for *method* over all tasks.

    Parameters
    ----------
    ttest_results : dict or None
        Pre-computed t-test results keyed by task code
        (e.g. ``{'X1': {...}, 'X2': {...}}``), as returned by
        :func:`_p300_ttest`.  Avoids re-running the t-test.
    """
    method_dir = os.path.join(output_dir, method)
    os.makedirs(method_dir, exist_ok=True)

    summary_path = os.path.join(
        method_dir, f"summary_{_pick_label(cluster, picks)}.png"
    )
    n_tasks = len(tasks)
    fig, axes = plt.subplots(
        n_tasks,
        2,
        figsize=(10, 2.2 * n_tasks),
        sharex=True,
        constrained_layout=True,
    )
    if n_tasks == 1:
        axes = np.array([axes])

    any_plotted = False
    for row, task in enumerate(tasks):
        result = load_epochs_for_condition(method, task, subject_filter=subject_filter)
        if result is None:
            for col in range(2):
                axes[row, col].set_title(f"{task} — (no data)", fontsize=9)
                axes[row, col].axis("off")
            continue
        combined = result["combined"]
        subjects = result["subjects"]
        try:
            picks_names = _resolve_picks(combined, cluster, picks)
        except ValueError:
            for col in range(2):
                axes[row, col].axis("off")
            continue

        # Use the first channel of the cluster for the summary view
        ch_name = picks_names[0]

        # Look up pre-computed t-test result for this task
        ttest_result = ttest_results.get(task) if ttest_results is not None else None

        for col, event_label in enumerate((EVENT_LABEL_RARE, EVENT_LABEL_FREQ)):
            ax = axes[row, col]
            data = combined[event_label].get_data(picks=[ch_name])[
                :, 0, :
            ]  # (n_trials, n_times)
            times_ms = combined[event_label].times * 1000.0
            for trial in data:
                ax.plot(
                    times_ms, trial * 1e6, color="lightgray", linewidth=0.4, alpha=0.6
                )
            # Grand-average ERP (proper per-subject averaging)
            grand = _compute_grand_average(subjects, event_label)
            if grand is not None:
                mean = grand.get_data(picks=[ch_name])[0] * 1e6
            else:
                mean = data.mean(axis=0) * 1e6
            color = "#d62728" if event_label == EVENT_LABEL_RARE else "#1f77b4"
            ax.plot(
                times_ms,
                mean,
                color=color,
                linewidth=2.0,
                label=f"Mean (n={data.shape[0]})",
            )
            ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--")
            ax.axvline(0.0, color="black", linewidth=0.5, linestyle="--")
            ax.set_title(f"{task} — {event_label} ({ch_name})", fontsize=10)
            if col == 0:
                ax.set_ylabel("Amplitude (µV)")
            if row == n_tasks - 1:
                ax.set_xlabel("Time (ms)")
            # Scale y-axis to the mean ERP range so the bold trace
            # is clearly visible; individual trials may clip outside.
            ylim = _mean_ylim(mean)
            ax.set_ylim(ylim)
            # Shade P300 window if t-test is significant
            if ttest_result is not None:
                _shade_p300_window(ax, ttest_result)
            ax.legend(loc="upper right", fontsize=7)
            any_plotted = True

    fig.suptitle(
        f"{DENOISING_LABELS.get(method, method)} — ERPs per RSVP condition",
        fontsize=13,
    )
    if any_plotted:
        fig.savefig(summary_path, dpi=150, bbox_inches="tight")
        print(f"  Summary saved to {os.path.relpath(summary_path)}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean ERPs (and individual trials) for each denoising method "
            "and each RSVP condition using MNE dedicated methods."
        )
    )
    parser.add_argument(
        "--method",
        choices=list(DENOISING_METHODS) + ["all"],
        default="all",
        help="Denoising method to process (default: all).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=RSVP_TASKS,
        help=(
            f"RSVP tasks to plot. Choose from {', '.join(RSVP_TASKS)} (default: all)."
        ),
    )
    parser.add_argument(
        "--cluster",
        default="occipital",
        choices=list(ELEC_CLUSTS),
        help=(
            "Electrode cluster to average for the channel view (default: occipital)."
        ),
    )
    parser.add_argument(
        "--picks",
        default=None,
        help=("Specific channel name to plot (overrides --cluster). E.g. POz, Pz, Cz."),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "figures"),
        help="Directory to write the PNG figures to.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the per-method combined summary figure.",
    )
    # ── Single-recording mode ──────────────────────────────────────────
    parser.add_argument(
        "--single",
        action="store_true",
        help=(
            "Plot each recording individually instead of grand-averaging "
            "across subjects. Produces per-recording ERP figures under "
            "<output>/<method>/single/."
        ),
    )
    parser.add_argument(
        "--subject",
        type=int,
        default=None,
        help=(
            "Restrict to one subject (1–10). Works in all modes "
            "(grand-average, grouped-task, and single-recording)."
        ),
    )
    parser.add_argument(
        "--session",
        type=int,
        default=None,
        help=(
            "Restrict single-recording mode to one session (1–4). "
            "Only used with --single."
        ),
    )
    # ── Grouped-task mode ───────────────────────────────────────────────
    parser.add_argument(
        "--group-tasks",
        action="store_true",
        help=(
            "Combine all tasks specified by --tasks into a single ERP "
            "figure, instead of plotting each task separately. For example, "
            "--group-tasks --tasks X1 X2 produces one combined ERP for "
            "standard RSVP, and --group-tasks --tasks X4 X6 X8 produces "
            "one combined ERP for artifact RSVP."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    methods = list(DENOISING_METHODS) if args.method == "all" else [args.method]
    # Import subject/session lists from config
    from config import SESSIONS, SUBJECTS  # noqa: E402

    # Build a subject filter that applies to ALL modes
    subject_filter = [args.subject] if args.subject is not None else None

    # ── Single-recording mode ──────────────────────────────────────────
    if args.single:
        subjects = [args.subject] if args.subject is not None else SUBJECTS
        sessions = [args.session] if args.session is not None else SESSIONS

        for method in methods:
            print(
                f"\n=== Single-recording mode: "
                f"{DENOISING_LABELS.get(method, method)} "
                f"(subjects={subjects}, sessions={sessions}) ==="
            )
            for sub in subjects:
                for ses in sessions:
                    if args.group_tasks:
                        # Group all tasks into one ERP per subject/session
                        plot_single_grouped(
                            method,
                            sub,
                            ses,
                            args.tasks,
                            args.output_dir,
                            args.cluster,
                            args.picks,
                        )
                    else:
                        for task in args.tasks:
                            plot_single_recording(
                                method,
                                sub,
                                ses,
                                task,
                                args.output_dir,
                                args.cluster,
                                args.picks,
                            )
                    if not args.no_summary:
                        plot_single_summary(
                            method,
                            sub,
                            ses,
                            args.output_dir,
                            args.cluster,
                            args.picks,
                            args.tasks,
                        )
        print(f"\nAll single-recording ERP figures written to: {args.output_dir}")
        return

    # ── Grouped-task mode (grand-average) ──────────────────────────────────
    if args.group_tasks:
        from config import ARTIFACT_RSVP, STANDARD_RSVP  # noqa: E402

        # Build task groups: standard vs artifact (or a single group
        # if only one category is present in args.tasks).
        std_in_tasks = [t for t in STANDARD_RSVP if t in args.tasks]
        art_in_tasks = [t for t in ARTIFACT_RSVP if t in args.tasks]
        task_groups: dict[str, list[str]] = {}
        if std_in_tasks:
            task_groups["Standard RSVP"] = std_in_tasks
        if art_in_tasks:
            task_groups["Artifact RSVP"] = art_in_tasks

        for method in methods:
            print(
                f"\n--- Processing {DENOISING_LABELS.get(method, method)} "
                f"(grouped, cluster={args.cluster}) ---"
            )
            # Collect t-test results keyed by group label
            ttest_by_group: dict[str, dict] = {}
            # One combined figure per task group
            for label, group_tasks in task_groups.items():
                result = plot_grouped_tasks(
                    method,
                    group_tasks,
                    args.output_dir,
                    args.cluster,
                    args.picks,
                    subject_filter=subject_filter,
                )
                if isinstance(result, dict):
                    ttest_by_group[label] = result
            # Cross-group summary
            if not args.no_summary and len(task_groups) > 0:
                plot_grouped_summary(
                    method,
                    task_groups,
                    args.output_dir,
                    args.cluster,
                    args.picks,
                    subject_filter=subject_filter,
                    ttest_results=ttest_by_group if ttest_by_group else None,
                )
        print(f"\nAll grouped-task ERP figures written to: {args.output_dir}")
        return

    # ── Grand-average mode (default) ────────────────────────────────────
    for method in methods:
        print(
            f"\n--- Processing {DENOISING_LABELS.get(method, method)} "
            f"(cluster={args.cluster}, picks={args.picks}) ---"
        )
        # Collect t-test results keyed by task to avoid duplicate computation
        ttest_by_task: dict[str, dict] = {}
        for task in args.tasks:
            result = plot_method_task(
                method,
                task,
                args.output_dir,
                args.cluster,
                args.picks,
                subject_filter=subject_filter,
            )
            if isinstance(result, dict):
                ttest_by_task[task] = result
        if not args.no_summary:
            plot_method_summary(
                method,
                args.output_dir,
                args.cluster,
                args.picks,
                args.tasks,
                subject_filter=subject_filter,
                ttest_results=ttest_by_task if ttest_by_task else None,
            )

    print(f"\nAll ERP figures written to: {args.output_dir}")


if __name__ == "__main__":
    main()
