"""
Per-subject classification: train and evaluate classifiers on each subject
independently, predicting rare vs frequent stimulus.

Adapted from VEP_classification_comp/classify_bysub.py for the AMBER dataset.
"""

from collections import Counter

import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler


def classify_bysub(
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
    time_switch,
):
    """Classify rare vs frequent per subject using 10-fold stratified CV.

    Parameters
    ----------
    data : np.ndarray
        Feature data. Shape (nTrials, nFeatures) for notime, or
        (nTrials, nFeatures, nSamples) for temporal features.
    classifier : sklearn classifier object
    classifier_name : str
    nSubs : int
    nSamples : int (1 for notime, else number of time points)
    nFeatures : int
    nClasses : int
    ID : np.ndarray, shape (nTrials,)
        Subject ID for each trial.
    labels : np.ndarray, shape (nTrials,)
        Class labels (0=frequent, 1=rare).
    balancing : str, "smote" | "ros" | "none"
    ma_win : int, moving-average window (0 = disabled)
    time_switch : str, "" or "_notime"

    Returns
    -------
    results : dict with metric arrays
    subject_ids : np.ndarray of unique subject IDs
    """
    unique_ids = np.unique(ID)
    subject_ids = unique_ids.copy()

    # Initialize metric arrays
    probs_total = np.empty((nClasses, nSamples, nSubs))
    cm_total = np.empty((nClasses * 2, nSamples, nSubs))
    acc_total = np.empty((nSamples, nSubs))
    acc_std_total = np.empty((nSamples, nSubs))
    accbal_total = np.empty((nSamples, nSubs))
    accbal_std_total = np.empty((nSamples, nSubs))
    auc_total = np.empty((nSamples, nSubs))
    auc_std_total = np.empty((nSamples, nSubs))
    precision_total = np.empty((nSamples, nSubs))
    recall_total = np.empty((nSamples, nSubs))
    f1_total = np.empty((nSamples, nSubs))
    importance_acc_total = np.empty((nFeatures, nSamples, nSubs))
    importance_acc_std_total = np.empty((nFeatures, nSamples, nSubs))
    importance_accbal_total = np.empty((nFeatures, nSamples, nSubs))
    importance_accbal_std_total = np.empty((nFeatures, nSamples, nSubs))
    importance_auc_total = np.empty((nFeatures, nSamples, nSubs))
    importance_auc_std_total = np.empty((nFeatures, nSamples, nSubs))
    importance_mi_total = np.empty((nFeatures, nSamples, nSubs))

    for isub, iid in enumerate(unique_ids):
        sub_idx = np.where(ID == iid)[0]
        data_sub = (
            np.squeeze(data[sub_idx, :])
            if "notime" in time_switch
            else np.squeeze(data[sub_idx, :, :])
        )

        # Per-subject accumulators
        probs_sub = np.empty((nClasses, nSamples))
        cm_sub = np.empty((nClasses * 2, nSamples))
        acc_sub = np.empty(nSamples)
        acc_std_sub = np.empty(nSamples)
        accbal_sub = np.empty(nSamples)
        accbal_std_sub = np.empty(nSamples)
        auc_sub = np.empty(nSamples)
        auc_std_sub = np.empty(nSamples)
        precision_sub = np.empty(nSamples)
        recall_sub = np.empty(nSamples)
        f1_sub = np.empty(nSamples)
        importance_acc_sub = np.empty((nFeatures, nSamples))
        importance_acc_std_sub = np.empty((nFeatures, nSamples))
        importance_accbal_sub = np.empty((nFeatures, nSamples))
        importance_accbal_std_sub = np.empty((nFeatures, nSamples))
        importance_auc_sub = np.empty((nFeatures, nSamples))
        importance_auc_std_sub = np.empty((nFeatures, nSamples))
        importance_mi_sub = np.empty((nFeatures, nSamples))

        # Moving average
        if ma_win != 0 and "notime" not in time_switch:
            npad = ((0, 0), (0, 0), (int(ma_win / 2), int(ma_win / 2)))
            data_sub_pad = np.pad(data_sub, pad_width=npad, mode="edge")
            data_sub_temp = np.cumsum(data_sub_pad, dtype=float, axis=2)
            data_sub_temp[:, :, ma_win:] = (
                data_sub_temp[:, :, ma_win:] - data_sub_temp[:, :, :-ma_win]
            )
            data_sub_ma = data_sub_temp[:, :, ma_win - 1 :] / ma_win
            data_sub_ma = data_sub_ma[:, :, :nSamples]
            data_sub = np.copy(data_sub_ma)

        for itime in range(nSamples):
            print(
                f"  {classifier_name} | sub {isub + 1}/{nSubs} (ID={iid}) | "
                f"time {itime}/{nSamples - 1}"
            )

            probs_kfold = np.zeros(nClasses)
            cm_kfold = np.zeros(nClasses * 2)
            acc_kfold = 0.0
            acc_std_kfold = []
            accbal_kfold = 0.0
            accbal_std_kfold = []
            auc_kfold = 0.0
            auc_std_kfold = []
            recall_kfold = 0.0
            precision_kfold = 0.0
            f1_kfold = 0.0
            importance_acc_mean_kfold = np.zeros(nFeatures)
            importance_acc_std_kfold = np.zeros(nFeatures)
            importance_accbal_mean_kfold = np.zeros(nFeatures)
            importance_accbal_std_kfold = np.zeros(nFeatures)
            importance_auc_mean_kfold = np.zeros(nFeatures)
            importance_auc_std_kfold = np.zeros(nFeatures)
            importance_mi_scores_kfold = np.zeros(nFeatures)

            X = (
                data_sub
                if "notime" in time_switch
                else np.squeeze(data_sub[:, :, itime])
            )
            y = labels[sub_idx]

            nFolds = 10
            kfold = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=0)

            for train_ix, test_ix in kfold.split(X, y):
                X_train_orig, X_test_orig = X[train_ix], X[test_ix]
                y_train_orig, y_test_orig = y[train_ix], y[test_ix]

                # Impute NaN values
                imputer = SimpleImputer(strategy="median").fit(X_train_orig)
                X_train_orig = imputer.transform(X_train_orig)
                X_test_orig = imputer.transform(X_test_orig)

                # Balance classes
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
                            sampling_strategy={0: meanSamples, 1: meanSamples},
                            random_state=0,
                        )
                        X_train, y_train = overSmote.fit_resample(
                            X_train_temp, y_train_temp
                        )
                    except ValueError:
                        X_train, y_train = X_train_temp, y_train_temp
                elif balancing == "ros":
                    try:
                        ros = RandomOverSampler(
                            sampling_strategy={0: meanSamples, 1: meanSamples},
                            random_state=0,
                        )
                        X_train, y_train = ros.fit_resample(X_train_temp, y_train_temp)
                    except ValueError:
                        X_train, y_train = X_train_temp, y_train_temp
                else:
                    X_train, y_train = X_train_orig, y_train_orig

                # Shuffle after balancing
                rng = np.random.default_rng()
                shuffle_indices = np.arange(len(y_train))
                rng.shuffle(shuffle_indices)
                X_train = X_train[shuffle_indices, :]
                y_train = y_train[shuffle_indices]

                # Normalize
                scaler = MinMaxScaler().fit(X_train)
                X_train_norm = scaler.transform(X_train)
                X_test_norm = scaler.transform(X_test_orig)

                # Train
                classifier.fit(X_train_norm, y_train)

                # Predict
                y_pred = classifier.predict(X_test_norm)
                y_pred_prob = classifier.predict_proba(X_test_norm)

                # Evaluate
                probs_kfold += np.mean(y_pred_prob, axis=0)
                cm_kfold += confusion_matrix(
                    y_test_orig, y_pred, labels=list(range(nClasses))
                ).reshape(nClasses * 2)
                acc_kfold += accuracy_score(y_test_orig, y_pred)
                acc_std_kfold.append(accuracy_score(y_test_orig, y_pred))
                accbal_kfold += balanced_accuracy_score(y_test_orig, y_pred)
                accbal_std_kfold.append(balanced_accuracy_score(y_test_orig, y_pred))

                try:
                    auc_kfold += roc_auc_score(y_test_orig, y_pred)
                    auc_std_kfold.append(roc_auc_score(y_test_orig, y_pred))
                except ValueError:
                    pass

                precision_kfold += precision_score(y_test_orig, y_pred, zero_division=0)
                recall_kfold += recall_score(y_test_orig, y_pred, zero_division=0)
                f1_kfold += f1_score(y_test_orig, y_pred, zero_division=0)

                # Feature importance
                try:
                    importance_mi_scores_kfold += mutual_info_classif(
                        X_train_norm, y_train
                    )
                except Exception:
                    pass

                try:
                    imp_acc = permutation_importance(
                        classifier, X_train_norm, y_train, scoring="accuracy"
                    )
                    importance_acc_mean_kfold += imp_acc.importances_mean
                    importance_acc_std_kfold += imp_acc.importances_std
                except Exception:
                    pass

                try:
                    imp_accbal = permutation_importance(
                        classifier, X_train_norm, y_train, scoring="balanced_accuracy"
                    )
                    importance_accbal_mean_kfold += imp_accbal.importances_mean
                    importance_accbal_std_kfold += imp_accbal.importances_std
                except Exception:
                    pass

                try:
                    imp_auc = permutation_importance(
                        classifier, X_train_norm, y_train, scoring="roc_auc"
                    )
                    importance_auc_mean_kfold += imp_auc.importances_mean
                    importance_auc_std_kfold += imp_auc.importances_std
                except Exception:
                    pass

            # Save per-timepoint averages
            probs_sub[:, itime] = probs_kfold / nFolds
            cm_sub[:, itime] = cm_kfold / nFolds
            acc_sub[itime] = acc_kfold / nFolds
            acc_std_sub[itime] = np.std(acc_std_kfold)
            accbal_sub[itime] = accbal_kfold / nFolds
            accbal_std_sub[itime] = np.std(accbal_std_kfold)
            auc_sub[itime] = auc_kfold / nFolds
            auc_std_sub[itime] = np.std(auc_std_kfold)
            recall_sub[itime] = recall_kfold / nFolds
            precision_sub[itime] = precision_kfold / nFolds
            f1_sub[itime] = f1_kfold / nFolds
            importance_acc_sub[:, itime] = importance_acc_mean_kfold / nFolds
            importance_acc_std_sub[:, itime] = importance_acc_std_kfold / nFolds
            importance_accbal_sub[:, itime] = importance_accbal_mean_kfold / nFolds
            importance_accbal_std_sub[:, itime] = importance_accbal_std_kfold / nFolds
            importance_auc_sub[:, itime] = importance_auc_mean_kfold / nFolds
            importance_auc_std_sub[:, itime] = importance_auc_std_kfold / nFolds
            importance_mi_sub[:, itime] = importance_mi_scores_kfold / nFolds

        # Save per-subject results
        probs_total[:, :, isub] = probs_sub
        cm_total[:, :, isub] = cm_sub
        acc_total[:, isub] = acc_sub
        acc_std_total[:, isub] = acc_std_sub
        accbal_total[:, isub] = accbal_sub
        accbal_std_total[:, isub] = accbal_std_sub
        auc_total[:, isub] = auc_sub
        auc_std_total[:, isub] = auc_std_sub
        precision_total[:, isub] = precision_sub
        recall_total[:, isub] = recall_sub
        f1_total[:, isub] = f1_sub
        importance_acc_total[:, :, isub] = importance_acc_sub
        importance_acc_std_total[:, :, isub] = importance_acc_std_sub
        importance_accbal_total[:, :, isub] = importance_accbal_sub
        importance_accbal_std_total[:, :, isub] = importance_accbal_std_sub
        importance_auc_total[:, :, isub] = importance_auc_sub
        importance_auc_std_total[:, :, isub] = importance_auc_std_sub
        importance_mi_total[:, :, isub] = importance_mi_sub

    results = {
        "probs_total": probs_total,
        "cm_total": cm_total,
        "acc_total": acc_total,
        "acc_std_total": acc_std_total,
        "accbal_total": accbal_total,
        "accbal_std_total": accbal_std_total,
        "auc_total": auc_total,
        "auc_std_total": auc_std_total,
        "precision_total": precision_total,
        "recall_total": recall_total,
        "f1_total": f1_total,
        "importance_acc_total": importance_acc_total,
        "importance_acc_std_total": importance_acc_std_total,
        "importance_accbal_total": importance_accbal_total,
        "importance_accbal_std_total": importance_accbal_std_total,
        "importance_auc_total": importance_auc_total,
        "importance_auc_std_total": importance_auc_std_total,
        "importance_mi_total": importance_mi_total,
    }

    return results, subject_ids
