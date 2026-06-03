#!/usr/bin/env python3
"""
Merge per-classifier results into a single DataFrame for analysis.

Adapted from VEP_classification_comp/merge_results.py for the AMBER dataset.

Usage:
    python merge_results.py RAW
    python merge_results.py ASR --classification bysub_notime
"""

import argparse
import os
import pickle
import sys
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BALANCING,
    CLASSIFIER_NAMES,
    DATA_OUT_ROOT,
    DENOISING_METHODS,
    IMPORTANCE_METRIC_NAMES,
    MA_WIN,
    METRIC_NAMES,
    RESULTS_ROOT,
    TIME_SWITCH,
    get_data_version,
    get_exp_version,
    get_results_dir,
)


def merge_results(
    method: str,
    classification_type: Optional[str] = None,
    balancing: str = BALANCING,
    ma_win: int = MA_WIN,
    time_switch: str = TIME_SWITCH,
    task_set: str = "all",
) -> None:
    """Merge individual classifier results into DataFrames."""
    if classification_type is None:
        classification_type = "bysub" + time_switch

    exp_version = get_exp_version(method, task_set)
    data_version = get_data_version(method, task_set)

    if classification_type.startswith("bysub") or classification_type.startswith(
        "bytask"
    ):
        target = "stim"
    else:
        target = "stim"

    path_results_in = get_results_dir(method, classification_type, task_set)
    print(f"=== Merging results for method={method} ===")

    # Load general info
    info_filename = (
        f"classificaton_results_{classification_type}"
        f"_dat{data_version}_exp{exp_version}"
        f"_{balancing}_{target}_ma{ma_win}_general_info.pkl"
    )
    info_path = os.path.join(path_results_in, info_filename)

    if not os.path.exists(info_path):
        print(f"  [ERROR] General info file not found: {info_path}")
        print("  Run classify_main.py first.")
        return

    with open(info_path, "rb") as f:
        general_info = pickle.load(f)

    feature_names_raw = general_info["feature_names"]
    if isinstance(feature_names_raw, np.ndarray):
        feature_names = [str(f).strip() for f in feature_names_raw.flat]
    else:
        feature_names = list(feature_names_raw)

    label_names = general_info["label_names"]
    times = general_info["times"] if time_switch == "" else np.array([0])
    subID = general_info["subID"]
    # Precompute unique subject IDs and counts
    unique_subIDs = np.unique(subID)
    nSubs_unique = len(unique_subIDs)
    nSamples = len(times) if time_switch != "_notime" else 1
    nFeatures = len(feature_names)
    nClassifiers = len(CLASSIFIER_NAMES)

    bysub_columns = ["sub"] if "bysub" in classification_type else []

    # Set up empty DataFrames
    df_probs = pd.DataFrame(
        columns=["time"] + bysub_columns + ["model", "class", "Probabilities"]
    )
    df_metrics = pd.DataFrame(
        columns=["time"] + bysub_columns + ["model"] + METRIC_NAMES
    )
    df_importances = pd.DataFrame(
        columns=["time"]
        + bysub_columns
        + ["feature", "model"]
        + IMPORTANCE_METRIC_NAMES
    )
    df_duration = pd.DataFrame(columns=["model", "Duration"])

    df_duration["model"] = CLASSIFIER_NAMES

    # Populate DataFrames with index structure
    n_rows_base = nSamples * nClassifiers
    if "bysub" in classification_type:
        nSubs_unique = len(np.unique(subID))
        total_rows = n_rows_base * nSubs_unique
        df_probs["sub"] = np.repeat(np.unique(subID), nClassifiers * nSamples)
        df_metrics["sub"] = np.repeat(np.unique(subID), nClassifiers * nSamples)
        df_importances["sub"] = np.repeat(
            np.unique(subID), nClassifiers * nFeatures * nSamples
        )
        df_probs["age"] = 0  # placeholder (AMBER has no age groups)
        df_metrics["age"] = 0
        df_importances["age"] = 0

    # Determine how many times to repeat blocks across subjects
    rep_sub = nSubs_unique if "bysub" in classification_type else 1

    # For probs/metrics: within each subject block we want classifier blocks where
    # each classifier has nSamples rows (time points) in order. So for the full
    # table we tile that pattern across subjects.
    df_probs["model"] = np.tile(np.repeat(CLASSIFIER_NAMES, nSamples), rep_sub)
    df_probs["time"] = np.tile(np.tile(times.flatten(), nClassifiers), rep_sub)

    df_metrics["model"] = np.tile(np.repeat(CLASSIFIER_NAMES, nSamples), rep_sub)
    df_metrics["time"] = np.tile(np.tile(times.flatten(), nClassifiers), rep_sub)

    # For importances: within each subject+classifier block we want time blocks
    # where features vary fastest (feature, then time, then subject).
    df_importances["model"] = np.tile(
        np.repeat(CLASSIFIER_NAMES, nFeatures * nSamples), rep_sub
    )
    df_importances["feature"] = np.tile(
        feature_names, nClassifiers * nSamples * rep_sub
    )
    df_importances["time"] = np.tile(
        np.repeat(times.flatten(), nFeatures), nClassifiers * rep_sub
    )

    # Load and merge results from each classifier
    for clfIdx, clfName in enumerate(CLASSIFIER_NAMES):
        filename = (
            f"classificaton_results_{classification_type}"
            f"_dat{data_version}_exp{exp_version}"
            f"_{balancing}_{target}_ma{ma_win}_{clfName}.pkl"
        )
        filepath = os.path.join(path_results_in, filename)

        if not os.path.exists(filepath):
            print(f"  [SKIP] Results not found for {clfName}")
            continue

        with open(filepath, "rb") as f:
            loaded = pickle.load(f)
            results = loaded[0]
            duration = loaded[4] if len(loaded) > 4 else 0

        print(f"  Loading results: {clfName}")

        # Duration
        df_duration.loc[df_duration["model"] == clfName, "Duration"] = duration

        # Metrics
        for mi, metric in enumerate(METRIC_NAMES):
            metric_key = (
                metric.replace("Accuracy_Std", "acc_std_total")
                .replace("Balanced_Accuracy_Std", "accbal_std_total")
                .replace("AUROC_Std", "auc_std_total")
            )
            # Map metric names to result keys
            metric_map = {
                "Accuracy": "acc_total",
                "Accuracy_Std": "acc_std_total",
                "Balanced_Accuracy": "accbal_total",
                "Balanced_Accuracy_Std": "accbal_std_total",
                "AUROC": "auc_total",
                "AUROC_Std": "auc_std_total",
                "Precision": "precision_total",
                "Recall": "recall_total",
                "F1": "f1_total",
            }
            key = metric_map.get(metric, metric)
            if key in results:
                df_metrics.loc[df_metrics["model"] == clfName, metric] = results[
                    key
                ].flatten("F")

        # Importances
        importance_map = {
            "Importance_Acc": "importance_acc_total",
            "Importance_Acc_Std": "importance_acc_std_total",
            "Importance_Balanced_Acc": "importance_accbal_total",
            "Importance_Balanced_Acc_Std": "importance_accbal_std_total",
            "Importance_AUROC": "importance_auc_total",
            "Importance_AUROC_Std": "importance_auc_std_total",
            "Importance_MI": "importance_mi_total",
        }
        for imp_name, imp_key in importance_map.items():
            if imp_key in results:
                df_importances.loc[df_importances["model"] == clfName, imp_name] = (
                    results[imp_key].flatten("F")
                )

    # Save merged results
    merged_filename = (
        f"merged_classificaton_results_dat{data_version}_exp{exp_version}"
        f"_{balancing}_{target}_ma{ma_win}.pkl"
    )
    with open(os.path.join(path_results_in, merged_filename), "wb") as f:
        pickle.dump(
            [general_info, df_duration, df_probs, df_metrics, df_importances],
            f,
            protocol=-1,
        )

    print(f"=== Finished merging results for method={method} ===")
    print(f"  Saved: {os.path.join(path_results_in, merged_filename)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge classification results")
    parser.add_argument("method", choices=list(DENOISING_METHODS.keys()))
    parser.add_argument("--classification", default=None)
    parser.add_argument(
        "--task-set",
        default="all",
        help=(
            "Task-set identifier (e.g. 'all', 'standard', 'artifact', 'artifact_X4X6'). "
            "Default: 'all'."
        ),
    )
    args = parser.parse_args()
    merge_results(
        args.method, classification_type=args.classification, task_set=args.task_set
    )
