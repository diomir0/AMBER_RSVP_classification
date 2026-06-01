"""
Central configuration for the AMBER ERP classification pipeline.

This module defines all configurable parameters used across the pipeline,
including denoising methods, classifiers, feature extraction settings,
classification tasks, and path conventions.
"""

import os

# ──────────────────────────────────────────────────────────────────────
# Project paths
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREPROCESSING_ROOT = os.path.join(PROJECT_ROOT, "preprocessing")
CLEAN_ROOT = os.path.join(PREPROCESSING_ROOT, "clean")

# Path where extracted feature .mat / .npz files are saved
DATA_OUT_ROOT = os.path.join(os.path.dirname(__file__), "data_for_classification")
# Path where classification results are saved
RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")

# ──────────────────────────────────────────────────────────────────────
# Denoising methods
# ──────────────────────────────────────────────────────────────────────
# Maps the method key (used in code & filenames) to the sub-folder name
# inside preprocessing/clean/
DENOISING_METHODS = {
    "RAW": "Raw",
    "ASR": "ASR",
    "ICLabel": "ICLabel",
    "MARA": "MARA",
    "GEDAI": "GEDAIv1.7",
}

# Friendly labels & colours for plots
DENOISING_LABELS = {
    "RAW": "Raw (no denoising)",
    "ASR": "ASR",
    "ICLabel": "IClabel",
    "MARA": "MARA",
    "GEDAI": "GEDAI v1.7",
}
DENOISING_COLORS = {
    "RAW": "#808080",
    "ASR": "#1f77b4",
    "ICLabel": "#ff7f0e",
    "MARA": "#2ca02c",
    "GEDAI": "#d62728",
}

# ──────────────────────────────────────────────────────────────────────
# RSVP tasks (contain frequent / rare stimulus events)
# ──────────────────────────────────────────────────────────────────────
# X1, X2: Standard RSVP (no artifact production)
# X4, X6, X8: RSVP + artifact production
RSVP_TASKS = ["X1", "X2", "X4", "X6", "X8"]
STANDARD_RSVP = ["X1", "X2"]
ARTIFACT_RSVP = ["X4", "X6", "X8"]

# Valid artifact conditions (subsets of ARTIFACT_RSVP)
ARTIFACT_CONDITIONS = ["X4", "X6", "X8"]

# ──────────────────────────────────────────────────────────────────────
# Subject / session info
# ──────────────────────────────────────────────────────────────────────
SUBJECTS = list(range(1, 11))  # P1 – P10
SESSIONS = list(range(1, 5))  # Ss1 – Ss4

# ──────────────────────────────────────────────────────────────────────
# Feature extraction settings
# ──────────────────────────────────────────────────────────────────────
SAMPLING_FREQ = 256  # Hz (after preprocessing, the .set files use this)

# Electrode cluster definitions (1-based indices matching EEGLAB channel order)
# These match the 32-channel layout used in the AMBER dataset.
# Adjust if your montage differs.
ELEC_CLUSTS = {
    "occipital": list(range(28, 33)),  # channels 28-32
    "parietal": list(range(22, 28)),  # channels 22-27
    "central": list(range(13, 19)),  # channels 13-18
    "frontal": list(range(4, 10)),  # channels 4-9
}

# ERP component windows (ms) for statistical feature extraction
ERP_COMPONENTS = {
    "P1": {"window_ms": (50, 150), "polarity": "positive", "cluster": "occipital"},
    "N1": {"window_ms": (100, 200), "polarity": "negative", "cluster": "occipital"},
    "P2": {"window_ms": (200, 325), "polarity": "positive", "cluster": "occipital"},
    "P3": {"window_ms": (250, 500), "polarity": "positive", "cluster": "central"},
}

# ERP statistical parameters to extract per component
ERP_PARAMS = [
    "PA",
    "MA",
    "PL",
    "FL",
]  # Peak Amplitude, Mean Amplitude, Peak Latency, Fractional Latency

# Epoch structure
# ──────────────────────────────────────────────────────────────────────
# In the .set EEGLAB files, recordings are pre-epoched for each event
# labeled '1' (rare) or '2' (frequent). Each epoch has exactly ONE event
# centered at t=0. There are 360 epochs per recording: 36 rare (event '1')
# and 324 frequent (event '2'). Events appear sequentially every 250 ms
# (ISI) in the original continuous recording, but each epoch captures only
# the response to its triggering event.

# Inter-stimulus interval in the RSVP paradigm (ms)
ISI_MS = 250

# Expected epoch counts per .set file
N_RARE_EXPECTED = 36
N_FREQ_EXPECTED = 324
N_EPOCHS_EXPECTED = N_RARE_EXPECTED + N_FREQ_EXPECTED  # 360

# Epoch time window (seconds relative to stimulus onset)
TMIN = -0.2
TMAX = 1.0

# Trial rejection: reject epochs exceeding this amplitude (µV) at relevant clusters
NOISE_THRESHOLD_UV = 100.0
# Minimum number of good trials per subject to include them
N_TRIALS_THRESHOLD = 50
# Maximum number of trials to keep per subject (balance rare/freq)
N_TRIALS_MAX = 200

# ──────────────────────────────────────────────────────────────────────
# Classification settings
# ──────────────────────────────────────────────────────────────────────
TIME_SWITCH = (
    "_notime"  # "" for temporal features, "_notime" for statistical ERP params
)
BALANCING = "smote"  # "smote", "ros", or "none"
MA_WIN = 0  # Moving-average window (0 = disabled)
N_FOLDS = 10  # Stratified K-fold CV splits
RANDOM_STATE = 0

# Classification tasks
#   "bysub"        – classify rare vs frequent per subject
#   "allsubs"      – classify rare vs frequent across all subjects
#   "bytask"       – classify standard RSVP vs RSVP+artifact across subjects
CLASSIFICATION_TYPES = ["bysub", "allsubs", "bytask"]

# Classifiers to evaluate
CLASSIFIER_NAMES = [
    "LDA",
    "KNN",
    "LR",
    "Tree",
    "AdaBoost",
    "XGB",
    "RF",
    "SVC_lin",
    "SVC_rbf",
]

METRIC_NAMES = [
    "Accuracy",
    "Accuracy_Std",
    "Balanced_Accuracy",
    "Balanced_Accuracy_Std",
    "AUROC",
    "AUROC_Std",
    "Precision",
    "Recall",
    "F1",
]

IMPORTANCE_METRIC_NAMES = [
    "Importance_Acc",
    "Importance_Acc_Std",
    "Importance_Balanced_Acc",
    "Importance_Balanced_Acc_Std",
    "Importance_AUROC",
    "Importance_AUROC_Std",
    "Importance_MI",
]

# ──────────────────────────────────────────────────────────────────────
# Parallel processing
# ──────────────────────────────────────────────────────────────────────
N_JOBS = -1  # Number of parallel jobs (-1 = all CPUs, 1 = sequential)

# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────


def resolve_recordings(recordings=None, artifact_conditions=None):
    """Resolve recording specification to a list of RSVP tasks.

    Parameters
    ----------
    recordings : list[str] or None
        Each element can be:
        - 'standard': include STANDARD_RSVP tasks (X1, X2)
        - 'artifact': include ARTIFACT_RSVP tasks (X4, X6, X8),
          optionally filtered by *artifact_conditions*
        - 'all': include all RSVP tasks (default)
        - Individual task codes (e.g. 'X1', 'X4')
    artifact_conditions : list[str] or None
        When 'artifact' is in *recordings*, which conditions to include.
        Any subset of ['X4', 'X6', 'X8'].
        Defaults to all artifact conditions if None.

    Returns
    -------
    list[str]
        Sorted list of RSVP task codes to include.
    """
    if recordings is None:
        return RSVP_TASKS.copy()

    tasks = []
    for rec in recordings:
        rec_lower = rec.lower()
        if rec_lower == "standard":
            tasks.extend(STANDARD_RSVP)
        elif rec_lower == "artifact":
            if artifact_conditions:
                for cond in artifact_conditions:
                    cond_upper = cond.upper()
                    if cond_upper in ARTIFACT_RSVP:
                        tasks.append(cond_upper)
            else:
                tasks.extend(ARTIFACT_RSVP)
        elif rec_lower == "all":
            tasks.extend(RSVP_TASKS)
        else:
            rec_upper = rec.upper()
            if rec_upper in RSVP_TASKS:
                tasks.append(rec_upper)
            else:
                raise ValueError(
                    f"Unknown recording specification: '{rec}'. "
                    f"Use 'standard', 'artifact', 'all', or individual task codes "
                    f"({', '.join(RSVP_TASKS)})."
                )

    # Deduplicate while preserving order, then sort by task number
    seen = set()
    unique_tasks = []
    for t in tasks:
        if t not in seen:
            seen.add(t)
            unique_tasks.append(t)
    return sorted(unique_tasks, key=lambda x: int(x[1:]))


def get_task_id(tasks):
    """Create a short identifier string from a task list for use in filenames.

    Returns
    -------
    str
        Identifier like 'all', 'standard', 'artifact', 'artifact_X4X6', etc.
        For 'all' (the default), the empty string is returned so that
        existing filenames remain unchanged.
    """
    task_set = set(tasks)

    if task_set == set(RSVP_TASKS):
        return "all"
    elif task_set == set(STANDARD_RSVP):
        return "standard"
    elif task_set == set(ARTIFACT_RSVP):
        return "artifact"
    else:
        # Build a descriptive ID from component parts
        std_included = sorted(task_set & set(STANDARD_RSVP), key=lambda x: int(x[1:]))
        art_included = sorted(task_set & set(ARTIFACT_RSVP), key=lambda x: int(x[1:]))
        parts = []
        if std_included == STANDARD_RSVP:
            parts.append("std")
        elif std_included:
            parts.append("".join(std_included))
        if art_included == ARTIFACT_RSVP:
            parts.append("art")
        elif art_included:
            parts.append("art" + "".join(art_included))
        return "_".join(parts) if parts else "none"


def get_clean_dir(method: str) -> str:
    """Return the path to the cleaned data directory for *method*."""
    sub = DENOISING_METHODS[method]
    return os.path.join(CLEAN_ROOT, sub)


def get_data_version(method: str, task_set: str = "all") -> str:
    """Return the data-version string used in filenames.

    Parameters
    ----------
    method : str
        Denoising method key.
    task_set : str
        Task-set identifier from get_task_id().  The default 'all' is
        omitted from the filename for backward compatibility.
    """
    base = method.lower()
    if task_set != "all":
        base += "_" + task_set
    if TIME_SWITCH == "_notime":
        return base + TIME_SWITCH
    return base


def get_exp_version(method: str, task_set: str = "all") -> str:
    """Return the experiment-version string used in filenames and dirs.

    Parameters
    ----------
    method : str
        Denoising method key.
    task_set : str
        Task-set identifier from get_task_id().  The default 'all' is
        omitted for backward compatibility.
    """
    base = method.lower()
    if task_set != "all":
        base += "_" + task_set
    return base


def get_results_dir(
    method: str, classification_type: str, task_set: str = "all"
) -> str:
    """Return the results directory for a given method + classification_type.

    Parameters
    ----------
    method : str
        Denoising method key.
    classification_type : str
        Classification type (e.g. 'bysub_notime').
    task_set : str
        Task-set identifier from get_task_id().
    """
    d = os.path.join(
        RESULTS_ROOT,
        classification_type,
        get_exp_version(method, task_set),
    )
    os.makedirs(d, exist_ok=True)
    return d
