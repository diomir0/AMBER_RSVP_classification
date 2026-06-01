"""
All-subjects classification: pool all subjects together and classify rare vs
frequent (2-class) or rare/frequent × task type (4-class).

Adapted from VEP_classification_comp/classify_allsubs.py for the AMBER dataset.
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


def classify_allsubs(
    data,
    classifier,
    classifier_name,
    nSamples,
    nFeatures,
    nClasses,
    labels,
    balancing,
    time_switch,
):
    """Classify rare vs frequent across all subjects using 10-fold stratified CV.

    Parameters
    ----------
    data : np.ndarray
        Feature data. Shape (nTrials, nFeatures) for notime, or
        (nTrials, nFeatures, nSamples) for temporal features.
    classifier : sklearn classifier object
    classifier_name : str
    nSamples : int (1 for notime, else number of time points)
    nFeatures : int
    nClasses : int (2 for stimulus, 4 for stimulus × task)
    labels : np.ndarray, shape (nTrials,)
        Class labels.
    balancing : str, "smote" | "ros" | "none"
    time_switch : str, "" or "_notime"

    Returns
    -------
    results : dict with metric arrays
    """
    probs_total = np.empty((nClasses, nSamples))
    acc_total = np.empty(nSamples)
    acc_std_total = np.empty(nSamples)
    accbal_total = np.empty(nSamples)
    accbal_std_total = np.empty(nSamples)
    auc_total = np.empty(nSamples)
    auc_std_total = np.empty(nSamples)
    precision_total = np.empty(nSamples)
    recall_total = np.empty(nSamples)
    f1_total = np.empty(nSamples)
    importance_acc_total = np.empty((nFeatures, nSamples))
    importance_acc_std_total = np.empty((nFeatures, nSamples))
    importance_accbal_total = np.empty((nFeatures, nSamples))
    importance_accbal_std_total = np.empty((nFeatures, nSamples))
    importance_auc_total = np.empty((nFeatures, nSamples))
    importance_auc_std_total = np.empty((nFeatures, nSamples))
    importance_mi_total = np.empty((nFeatures, nSamples))

    for itime in range(nSamples):
        print(f"  {classifier_name} | allsubs | time {itime}/{nSamples - 1}")

        probs_kfold = np.zeros(nClasses)
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
        importance_auc_mean_kfold = np.zeros(nFeatures)
        importance_auc_std_kfold = np.zeros(nFeatures)
        importance_accbal_mean_kfold = np.zeros(nFeatures)
        importance_accbal_std_kfold = np.zeros(nFeatures)
        importance_mi_scores_kfold = np.zeros(nFeatures)

        X = data if "notime" in time_switch else np.squeeze(data[:, :, itime])
        y = np.copy(labels)

        nFolds = 10
        kfold = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=1)

        for train_ix, test_ix in kfold.split(X, y):
            X_train_orig, X_test_orig = X[train_ix], X[test_ix]
            y_train_orig, y_test_orig = y[train_ix], y[test_ix]

            # Impute NaN values
            imputer = SimpleImputer(strategy="median").fit(X_train_orig)
            X_train_orig = imputer.transform(X_train_orig)
            X_test_orig = imputer.transform(X_test_orig)

            # Balance classes
            class_counts = Counter(y_train_orig)
            classes_present = sorted(class_counts.keys())
            meanSamples = int(
                np.mean([class_counts.get(c, 0) for c in classes_present])
            )
            if meanSamples == 0:
                continue

            if len(classes_present) == 2:
                undersample_key = classes_present[0]
                oversample_key = classes_present[1]
                try:
                    rus = RandomUnderSampler(
                        sampling_strategy={undersample_key: meanSamples}, random_state=0
                    )
                    X_train_temp, y_train_temp = rus.fit_resample(
                        X_train_orig, y_train_orig
                    )
                except ValueError:
                    X_train_temp, y_train_temp = X_train_orig, y_train_orig
            else:
                # Multi-class: undersample majority, oversample minority
                oversample_strategy = {c: meanSamples for c in classes_present}
                X_train_temp, y_train_temp = X_train_orig, y_train_orig

            if balancing == "smote":
                try:
                    overSmote = SMOTE(
                        sampling_strategy={c: meanSamples for c in classes_present},
                        random_state=0,
                    )
                    X_train, y_train = overSmote.fit_resample(
                        X_train_temp, y_train_temp
                    )
                except (ValueError, RuntimeError):
                    X_train, y_train = X_train_temp, y_train_temp
            elif balancing == "ros":
                try:
                    ros = RandomOverSampler(
                        sampling_strategy={c: meanSamples for c in classes_present},
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
            acc_kfold += accuracy_score(y_test_orig, y_pred)
            acc_std_kfold.append(accuracy_score(y_test_orig, y_pred))
            accbal_kfold += balanced_accuracy_score(y_test_orig, y_pred)
            accbal_std_kfold.append(balanced_accuracy_score(y_test_orig, y_pred))

            try:
                if nClasses == 2:
                    auc_kfold += roc_auc_score(y_test_orig, y_pred)
                else:
                    auc_kfold += roc_auc_score(
                        y_test_orig, y_pred_prob, average="weighted", multi_class="ovr"
                    )
                auc_std_kfold.append(
                    roc_auc_score(y_test_orig, y_pred)
                    if nClasses == 2
                    else roc_auc_score(
                        y_test_orig, y_pred_prob, average="weighted", multi_class="ovr"
                    )
                )
            except ValueError:
                pass

            precision_kfold += precision_score(
                y_test_orig, y_pred, average="weighted", zero_division=0
            )
            recall_kfold += recall_score(
                y_test_orig, y_pred, average="weighted", zero_division=0
            )
            f1_kfold += f1_score(
                y_test_orig, y_pred, average="weighted", zero_division=0
            )

            # Feature importance
            try:
                importance_mi_scores_kfold += mutual_info_classif(X_train_norm, y_train)
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
                scoring = "roc_auc" if nClasses == 2 else "roc_auc_ovr"
                imp_auc = permutation_importance(
                    classifier, X_train_norm, y_train, scoring=scoring
                )
                importance_auc_mean_kfold += imp_auc.importances_mean
                importance_auc_std_kfold += imp_auc.importances_std
            except Exception:
                pass

        # Save averages
        probs_total[:, itime] = probs_kfold / nFolds
        acc_total[itime] = acc_kfold / nFolds
        acc_std_total[itime] = np.std(acc_std_kfold)
        accbal_total[itime] = accbal_kfold / nFolds
        accbal_std_total[itime] = np.std(accbal_std_kfold)
        auc_total[itime] = auc_kfold / nFolds
        auc_std_total[itime] = np.std(auc_std_kfold)
        recall_total[itime] = recall_kfold / nFolds
        precision_total[itime] = precision_kfold / nFolds
        f1_total[itime] = f1_kfold / nFolds
        importance_acc_total[:, itime] = importance_acc_mean_kfold / nFolds
        importance_acc_std_total[:, itime] = importance_acc_std_kfold / nFolds
        importance_accbal_total[:, itime] = importance_accbal_mean_kfold / nFolds
        importance_accbal_std_total[:, itime] = importance_accbal_std_kfold / nFolds
        importance_auc_total[:, itime] = importance_auc_mean_kfold / nFolds
        importance_auc_std_total[:, itime] = importance_auc_std_kfold / nFolds
        importance_mi_total[:, itime] = importance_mi_scores_kfold / nFolds

    results = {
        "probs_total": probs_total,
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

    return results
