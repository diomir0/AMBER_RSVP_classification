#!/usr/bin/env python3
"""
Main entry point for AMBER ERP classification.

Loads extracted features from data_for_classification/*.npz, runs
classification for a given denoising method, and saves results.

Usage:
    python classify_main.py RAW
    python classify_main.py ASR --classification bysub
    python classify_main.py ICLabel --classification allsubs
    python classify_main.py MARA --classification bytask
"""

import argparse
import itertools
import os
import pickle
import sys
import time

import numpy as np

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from classify_allsubs import classify_allsubs
from classify_bysub import classify_bysub
from config import (
    BALANCING,
    CLASSIFIER_NAMES,
    DATA_OUT_ROOT,
    DENOISING_METHODS,
    MA_WIN,
    N_FOLDS,
    N_JOBS,
    RANDOM_STATE,
    RESULTS_ROOT,
    TIME_SWITCH,
    get_data_version,
    get_exp_version,
    get_results_dir,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def build_classifiers(n_classes: int) -> dict:
    """Return the dict of classifier name -> sklearn estimator."""
    xgb_objective = "binary:logistic" if n_classes == 2 else "multi:softmax"
    return {
        "LDA": LinearDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "LR": LogisticRegression(class_weight="balanced", max_iter=4000),
        "Tree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(n_estimators=100),
        "XGB": XGBClassifier(
            objective=xgb_objective,
            booster="gbtree",
            eval_metric="auc",
            max_depth=4,
            n_estimators=100,
        ),
        "RF": RandomForestClassifier(
            n_estimators=100, max_depth=4, class_weight="balanced"
        ),
        "SVC_lin": SVC(kernel="linear", probability=True, gamma="scale", max_iter=4000),
        "SVC_rbf": SVC(kernel="rbf", probability=True, gamma="scale", max_iter=4000),
    }


def load_features(method: str, task_set: str = "all"):
    """Load extracted features from .npz file for a given method and task set."""
    data_version = get_data_version(method, task_set)
    filepath = os.path.join(DATA_OUT_ROOT, f"data_{data_version}.npz")

    if not os.path.exists(filepath):
        # Try without task_set suffix (backward compatibility)
        data_version_legacy = get_data_version(method)
        filepath_legacy = os.path.join(DATA_OUT_ROOT, f"data_{data_version_legacy}.npz")
        if os.path.exists(filepath_legacy):
            filepath = filepath_legacy
            data_version = data_version_legacy
        else:
            # Try alternative extensions
            filepath_mat = os.path.join(DATA_OUT_ROOT, f"data_{data_version}.mat")
            if not os.path.exists(filepath_mat):
                filepath_mat = os.path.join(
                    DATA_OUT_ROOT, f"data_{data_version_legacy}.mat"
                )
            if os.path.exists(filepath_mat):
                from scipy import io

                data_mat = io.loadmat(filepath_mat)
                return data_mat, "mat"

            print(f"[ERROR] Feature file not found: {filepath}")
            print("  Run extract_features.py first.")
            sys.exit(1)

    data = np.load(filepath, allow_pickle=True)
    return data, "npz"


def main():
    parser = argparse.ArgumentParser(description="AMBER ERP classification")
    parser.add_argument(
        "method", choices=list(DENOISING_METHODS.keys()), help="Denoising method"
    )
    parser.add_argument(
        "--classification",
        choices=["bysub", "allsubs", "bytask"],
        default=None,
        help="Classification type (default: bysub for notime, allsubs for temporal)",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        default=None,
        help="Classifiers to use (default: all)",
    )
    parser.add_argument(
        "--balancing",
        default=BALANCING,
        choices=["smote", "ros", "none"],
        help="Class balancing strategy",
    )
    parser.add_argument(
        "--time-switch",
        default=None,
        help="Override time_switch: '' for temporal, '_notime' for stat params",
    )
    parser.add_argument(
        "--task-set",
        default="all",
        help=(
            "Task-set identifier (e.g. 'all', 'standard', 'artifact', 'artifact_X4X6'). "
            "Determines which .npz feature file to load. Default: 'all'."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=N_JOBS,
        help=(
            "Number of parallel jobs for classification "
            "(-1 = all CPUs, 1 = sequential). Default: from config."
        ),
    )
    args = parser.parse_args()

    method = args.method
    balancing = args.balancing
    time_switch = args.time_switch if args.time_switch is not None else TIME_SWITCH
    task_set = args.task_set

    # Determine classification type
    classification_type = args.classification
    if classification_type is None:
        classification_type = "bysub" + time_switch

    # Load features
    print(f"=== Loading features for method: {method}, task set: {task_set} ===")
    data_file, file_format = load_features(method, task_set)

    if file_format == "npz":
        X = data_file["X"]
        yi = data_file["yi"]
        IDs = data_file["IDs"]
        feature_names_arr = data_file["X_features"]
        times_arr = data_file["time_info_orig"]
        y_cats = data_file["y"] if "y" in data_file else None
        y_task = data_file["y_task"] if "y_task" in data_file else None
    else:  # mat
        from scipy import io

        data_mat = data_file
        X = data_mat["X"]
        yi = data_mat["yi"]
        IDs = data_mat["IDs"]
        feature_names_arr = data_mat["X_features"]
        times_arr = data_mat["time_info_orig"]
        y_cats = data_mat["y_cats"] if "y_cats" in data_mat else None
        y_task = None

    # Parse feature names
    if isinstance(feature_names_arr, np.ndarray):
        feature_names = [str(f).strip() for f in feature_names_arr.flat]
    else:
        feature_names = [feature_names_arr]

    times = times_arr.flatten() if times_arr.ndim > 0 else times_arr

    # Determine data shape
    if "notime" in time_switch:
        nSamples = 1
    else:
        # Temporal features: X is (trials, features, time_points)
        X = np.transpose(X, (0, 2, 1)) if X.ndim == 3 and X.shape[2] > X.shape[1] else X
        nSamples = X.shape[2] if X.ndim == 3 else 1

    nTrials = X.shape[0]
    nFeatures = len(feature_names)

    # Parse subject IDs and labels
    if IDs.ndim == 2:
        subID = IDs[:, 0].astype(int)
    else:
        subID = IDs.flatten().astype(int)

    labels = yi.flatten().astype(int)
    # Ensure binary: 0 = frequent, 1 = rare
    if np.any(labels > 1):
        labels[labels % 2 == 0] = 0  # even -> frequent
        labels[labels % 2 == 1] = 1  # odd -> rare

    nSubs = len(np.unique(subID))
    nClasses = len(np.unique(labels))

    # Determine target
    if classification_type.startswith("bysub"):
        target = "stim"
    elif classification_type.startswith("allsubs"):
        target = "stim"
    elif classification_type.startswith("bytask"):
        target = "task"
        if y_task is not None:
            # Create 4-class labels: frequent_standard, rare_standard, frequent_artifact, rare_artifact
            labels = y_task.flatten().astype(int) * 2 + labels
            nClasses = len(np.unique(labels))
        else:
            print("[ERROR] Task labels not available for bytask classification")
            sys.exit(1)
    else:
        target = "stim"

    # Validate bytask classification with task set
    if classification_type.startswith("bytask"):
        has_standard = y_task is not None and 0 in y_task
        has_artifact = y_task is not None and 1 in y_task
        if not (has_standard and has_artifact):
            print(
                f"[WARNING] bytask classification with task_set='{task_set}': "
                f"only one task type present. bytask requires both standard "
                f"and artifact recordings. Consider using 'allsubs' instead."
            )

    data_version = get_data_version(method, task_set)
    exp_version = get_exp_version(method, task_set)
    path_out = get_results_dir(method, classification_type, task_set)

    classifiers_to_use = args.classifiers if args.classifiers else CLASSIFIER_NAMES
    classifiers = build_classifiers(nClasses)

    print(f"  Data version: {data_version}")
    print(f"  Classification: {classification_type}")
    print(f"  Target: {target}")
    print(f"  Trials: {nTrials}, Features: {nFeatures}, Subjects: {nSubs}")
    print(f"  Classes: {nClasses}, Labels: {np.unique(labels)}")
    print(f"  Classifiers: {classifiers_to_use}")

    np.random.seed(RANDOM_STATE)

    for classifier_name in classifiers_to_use:
        if classifier_name not in classifiers:
            print(f"  [WARN] Unknown classifier '{classifier_name}', skipping")
            continue
        classifier = classifiers[classifier_name]
        t = time.time()

        print(f"\n--- {classifier_name} ---")

        if classification_type.startswith("bysub"):
            results, subject_ids = classify_bysub(
                X,
                classifier,
                classifier_name,
                nSubs,
                nSamples,
                nFeatures,
                nClasses,
                subID,
                labels,
                balancing,
                MA_WIN,
                time_switch,
                n_jobs=args.n_jobs,
            )
            extra = subject_ids
        elif classification_type.startswith(
            "allsubs"
        ) or classification_type.startswith("bytask"):
            results = classify_allsubs(
                X,
                classifier,
                classifier_name,
                nSamples,
                nFeatures,
                nClasses,
                labels,
                balancing,
                time_switch,
                n_jobs=args.n_jobs,
            )
            extra = []
        else:
            continue

        duration = time.time() - t

        # Save results
        result_filename = (
            f"classificaton_results_{classification_type}"
            f"_dat{data_version}_exp{exp_version}"
            f"_{balancing}_{target}_ma{MA_WIN}_{classifier_name}.pkl"
        )
        with open(os.path.join(path_out, result_filename), "wb") as f:
            pickle.dump(
                [results, subID, labels, times, duration, extra], f, protocol=-1
            )

    # Save general info
    label_names = [f"class_{c}" for c in sorted(np.unique(labels))]
    general_info = {
        "classification_type": classification_type,
        "balancing": balancing,
        "classifiers_used": classifiers_to_use,
        "feature_names": feature_names,
        "label_names": label_names,
        "target": target,
        "ma_win": MA_WIN,
        "times": times,
        "subID": subID,
        "labels": labels,
        "denoising_method": method,
        "nClasses": nClasses,
    }
    info_filename = (
        f"classificaton_results_{classification_type}"
        f"_dat{data_version}_exp{exp_version}"
        f"_{balancing}_{target}_ma{MA_WIN}_general_info.pkl"
    )
    with open(os.path.join(path_out, info_filename), "wb") as f:
        pickle.dump(general_info, f, protocol=-1)

    print(f"\n=== Finished classification for method: {method} ===")
    print(f"  Results saved to: {path_out}")


if __name__ == "__main__":
    main()
