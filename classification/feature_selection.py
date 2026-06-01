"""
Feature selection using mutual information.

Adapted from VEP_classification_comp/feature_selection.py for the AMBER dataset.
Supports both per-subject and all-subjects feature selection.
"""

from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


def feature_selection_bysub(
    data,
    classifier,
    classifier_name,
    nSubs,
    nSamples,
    nFeatures,
    nClasses,
    ID,
    labels,
    balancing,
    ma_win,
    target,
    time_switch,
):
    """Per-subject feature selection using mutual information."""
    unique_ids = np.unique(ID)

    mi_total = np.empty((nFeatures, nSamples, nSubs))

    for isub, iid in enumerate(unique_ids):
        sub_idx = np.where(ID == iid)[0]
        data_sub = (
            np.squeeze(data[sub_idx, :])
            if "notime" in time_switch
            else np.squeeze(data[sub_idx, :, :])
        )

        mi_sub = np.empty((nFeatures, nSamples))

        for itime in range(nSamples):
            print(
                f"  {classifier_name} | sub {isub + 1}/{nSubs} | time {itime}/{nSamples - 1}"
            )

            mi_kfold = np.zeros(nFeatures)
            X = (
                data_sub
                if "notime" in time_switch
                else np.squeeze(data_sub[:, :, itime])
            )
            y = labels[sub_idx]

            nFolds = 10
            kfold = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=1)

            for train_ix, test_ix in kfold.split(X, y):
                X_train_orig, X_test_orig = X[train_ix], X[test_ix]
                y_train_orig, y_test_orig = y[train_ix], y[test_ix]

                imputer = SimpleImputer(strategy="median").fit(X_train_orig)
                X_train_orig = imputer.transform(X_train_orig)

                class_counts = Counter(y_train_orig)
                meanSamples = int(
                    np.mean([class_counts.get(0, 0), class_counts.get(1, 0)])
                )
                if meanSamples == 0:
                    continue

                try:
                    rus = RandomUnderSampler(
                        sampling_strategy={0: meanSamples}, random_state=0
                    )
                    X_train_temp, y_train_temp = rus.fit_resample(
                        X_train_orig, y_train_orig
                    )
                except ValueError:
                    X_train_temp, y_train_temp = X_train_orig, y_train_orig

                if balancing == "smote":
                    try:
                        overSmote = SMOTE(
                            sampling_strategy={1: meanSamples}, random_state=0
                        )
                        X_train, y_train = overSmote.fit_resample(
                            X_train_temp, y_train_temp
                        )
                    except ValueError:
                        X_train, y_train = X_train_temp, y_train_temp
                elif balancing == "ros":
                    try:
                        ros = RandomOverSampler(
                            sampling_strategy={1: meanSamples}, random_state=0
                        )
                        X_train, y_train = ros.fit_resample(X_train_temp, y_train_temp)
                    except ValueError:
                        X_train, y_train = X_train_temp, y_train_temp
                else:
                    X_train, y_train = X_train_orig, y_train_orig

                rng = np.random.default_rng()
                shuffle_indices = np.arange(len(y_train))
                rng.shuffle(shuffle_indices)
                X_train = X_train[shuffle_indices, :]
                y_train = y_train[shuffle_indices]

                scaler = MinMaxScaler().fit(X_train)
                X_train_norm = scaler.transform(X_train)

                mi = mutual_info_classif(X_train_norm, y_train)
                mi_kfold += mi

            mi_sub[:, itime] = mi_kfold / nFolds

        mi_total[:, :, isub] = mi_sub

    results = {"mi_total": mi_total}
    return results, unique_ids


def feature_selection_allsubs(
    data,
    classifier,
    classifier_name,
    nSamples,
    nFeatures,
    nClasses,
    labels,
    balancing,
    target,
    time_switch,
):
    """All-subjects feature selection using mutual information."""
    mi_total = np.empty((nFeatures, nSamples))

    for itime in range(nSamples):
        print(f"  {classifier_name} | allsubs | {target} | time {itime}/{nSamples - 1}")

        mi_kfold = np.zeros(nFeatures)
        X = data if "notime" in time_switch else np.squeeze(data[:, :, itime])
        y = np.copy(labels)

        nFolds = 10
        kfold = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=1)

        for train_ix, test_ix in kfold.split(X, y):
            X_train_orig, X_test_orig = X[train_ix], X[test_ix]
            y_train_orig, y_test_orig = y[train_ix], y[test_ix]

            imputer = SimpleImputer(strategy="median").fit(X_train_orig)
            X_train_orig = imputer.transform(X_train_orig)

            class_counts = Counter(y_train_orig)
            classes_present = sorted(class_counts.keys())
            meanSamples = int(
                np.mean([class_counts.get(c, 0) for c in classes_present])
            )
            if meanSamples == 0:
                continue

            if balancing == "smote":
                try:
                    overSmote = SMOTE(
                        sampling_strategy={c: meanSamples for c in classes_present},
                        random_state=0,
                    )
                    X_train, y_train = overSmote.fit_resample(
                        X_train_orig, y_train_orig
                    )
                except (ValueError, RuntimeError):
                    X_train, y_train = X_train_orig, y_train_orig
            elif balancing == "ros":
                try:
                    ros = RandomOverSampler(
                        sampling_strategy={c: meanSamples for c in classes_present},
                        random_state=0,
                    )
                    X_train, y_train = ros.fit_resample(X_train_orig, y_train_orig)
                except ValueError:
                    X_train, y_train = X_train_orig, y_train_orig
            else:
                X_train, y_train = X_train_orig, y_train_orig

            rng = np.random.default_rng()
            shuffle_indices = np.arange(len(y_train))
            rng.shuffle(shuffle_indices)
            X_train = X_train[shuffle_indices, :]
            y_train = y_train[shuffle_indices]

            scaler = MinMaxScaler().fit(X_train)
            X_train_norm = scaler.transform(X_train)

            mi = mutual_info_classif(X_train_norm, y_train)
            mi_kfold += mi

        mi_total[:, itime] = mi_kfold / nFolds

    results = {"mi_total": mi_total}
    return results, []
