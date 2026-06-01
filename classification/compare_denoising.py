#!/usr/bin/env python3
"""
Compare classification performance (AUROC) across denoising methods.

Adapted from VEP_classification_comp/compare_denoising.py for the AMBER dataset.

Usage:
    python compare_denoising.py
    python compare_denoising.py --classification bysub_notime
"""

import argparse
import os
import pickle
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BALANCING,
    DENOISING_COLORS,
    DENOISING_LABELS,
    DENOISING_METHODS,
    MA_WIN,
    RESULTS_ROOT,
    TIME_SWITCH,
    get_data_version,
    get_exp_version,
    get_results_dir,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_merged_results(
    method, classification_type, balancing, ma_win, time_switch, task_set="all"
):
    """Load merged classification results for a given denoising method."""
    exp_version = get_exp_version(method, task_set)
    data_version = get_data_version(method, task_set)
    path_results = get_results_dir(method, classification_type, task_set)

    merged_filename = (
        f"merged_classificaton_results_dat{data_version}_exp{exp_version}"
        f"_{balancing}_stim_ma{ma_win}.pkl"
    )
    filepath = os.path.join(path_results, merged_filename)

    if not os.path.exists(filepath):
        print(f"  [WARN] File not found for {method}: {filepath}")
        return None, None, None

    with open(filepath, "rb") as f:
        [general_info, df_duration, df_probs, df_metrics, df_importances] = pickle.load(
            f
        )

    return general_info, df_metrics, df_duration


def create_comparison_barplot(
    results,
    classifiers,
    metric="AUROC",
    path_out=".",
    classification_type="bysub_notime",
    balancing="smote",
):
    """Create a grouped bar plot comparing AUROC across denoising methods."""
    methods_with_data = list(results.keys())
    if not methods_with_data:
        return

    n_classifiers = len(classifiers)
    n_methods = len(methods_with_data)

    auroc_means = np.full((n_methods, n_classifiers), np.nan)
    auroc_sems = np.full((n_methods, n_classifiers), np.nan)

    for i, method in enumerate(methods_with_data):
        df_metrics = results[method]["df_metrics"]
        for j, clf in enumerate(classifiers):
            clf_data = df_metrics[df_metrics["model"] == clf][metric]
            clf_data_clean = clf_data.dropna()
            if len(clf_data_clean) > 0:
                auroc_means[i, j] = clf_data_clean.mean()
                auroc_sems[i, j] = clf_data_clean.std() / np.sqrt(len(clf_data_clean))

    # Grouped bar plot
    fig, ax = plt.subplots(figsize=(max(12, n_classifiers * 1.2), 6))
    bar_width = 0.8 / n_methods
    x = np.arange(n_classifiers)

    for i, method in enumerate(methods_with_data):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            auroc_means[i],
            bar_width,
            label=DENOISING_LABELS.get(method, method),
            color=DENOISING_COLORS.get(method, None),
            yerr=auroc_sems[i],
            capsize=3,
            alpha=0.85,
        )

    ax.set_xlabel("Classifier", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(
        f"{metric} Comparison Across Denoising Methods\n"
        f"(Classification: {classification_type})",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=45, ha="right")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(path_out, f"comparison_{metric}_by_classifier.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Heatmap
    fig, ax = plt.subplots(
        figsize=(max(8, n_classifiers * 0.8), max(4, n_methods * 0.8))
    )
    im = ax.imshow(auroc_means, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_xticks(np.arange(n_classifiers))
    ax.set_yticks(np.arange(n_methods))
    ax.set_xticklabels(classifiers, rotation=45, ha="right")
    ax.set_yticklabels([DENOISING_LABELS.get(m, m) for m in methods_with_data])
    for i in range(n_methods):
        for j in range(n_classifiers):
            if not np.isnan(auroc_means[i, j]):
                ax.text(
                    j,
                    i,
                    f"{auroc_means[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if auroc_means[i, j] > 0.75 else "black",
                )
    ax.set_title(f"{metric}: Denoising Method × Classifier", fontsize=13)
    fig.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    plt.savefig(
        os.path.join(path_out, f"comparison_{metric}_heatmap.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Average across classifiers
    method_means = np.nanmean(auroc_means, axis=1)
    method_sems = np.nanmean(auroc_sems, axis=1)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        range(n_methods),
        method_means,
        yerr=method_sems,
        capsize=5,
        color=[DENOISING_COLORS.get(m, "#333333") for m in methods_with_data],
        alpha=0.85,
    )
    for bar, val, sem in zip(bars, method_means, method_sems):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + sem + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xticks(range(n_methods))
    ax.set_xticklabels(
        [DENOISING_LABELS.get(m, m) for m in methods_with_data], fontsize=10
    )
    ax.set_ylabel(f"Mean {metric}", fontsize=12)
    ax.set_title(
        f"Average {metric} Across Classifiers by Denoising Method", fontsize=13
    )
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(path_out, f"comparison_{metric}_average_by_method.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    return auroc_means, auroc_sems, classifiers, methods_with_data


def perform_statistical_tests(results, classifiers, metric="AUROC", path_out="."):
    """Perform paired t-tests between denoising methods."""
    methods_with_data = list(results.keys())
    if len(methods_with_data) < 2:
        return

    stat_results = []
    for clf in classifiers:
        method_data = {}
        for method in methods_with_data:
            df_metrics = results[method]["df_metrics"]
            clf_data = (
                df_metrics[df_metrics["model"] == clf][metric]
                .dropna()
                .values.astype(float)
            )
            if len(clf_data) > 0:
                method_data[method] = clf_data

        for i, m1 in enumerate(methods_with_data):
            for j, m2 in enumerate(methods_with_data):
                if i >= j:
                    continue
                if m1 not in method_data or m2 not in method_data:
                    continue
                d1 = method_data[m1]
                d2 = method_data[m2]
                n_min = min(len(d1), len(d2))
                if n_min > 1:
                    t_stat, p_value = stats.ttest_rel(d1[:n_min], d2[:n_min])
                else:
                    t_stat, p_value = np.nan, np.nan
                stat_results.append(
                    {
                        "classifier": clf,
                        "method_1": m1,
                        "method_2": m2,
                        "mean_1": np.mean(d1),
                        "mean_2": np.mean(d2),
                        "diff": np.mean(d1) - np.mean(d2),
                        "t_stat": t_stat,
                        "p_value": p_value,
                        "n": n_min,
                    }
                )

    if stat_results:
        df_stats = pd.DataFrame(stat_results)
        n_comparisons = len(df_stats)
        df_stats["p_value_corrected"] = df_stats["p_value"] * n_comparisons
        df_stats["p_value_corrected"] = df_stats["p_value_corrected"].clip(upper=1.0)
        df_stats["significant"] = df_stats["p_value_corrected"] < 0.05
        df_stats.to_csv(
            os.path.join(path_out, f"statistical_comparisons_{metric}.csv"), index=False
        )
        print("\n=== Statistical Comparisons ===")
        print(
            df_stats[
                [
                    "classifier",
                    "method_1",
                    "method_2",
                    "mean_1",
                    "mean_2",
                    "p_value_corrected",
                    "significant",
                ]
            ].to_string()
        )


def main():
    parser = argparse.ArgumentParser(description="Compare denoising methods")
    parser.add_argument("--classification", default=None)
    parser.add_argument(
        "--task-set",
        default="all",
        help=(
            "Task-set identifier (e.g. 'all', 'standard', 'artifact', 'artifact_X4X6'). "
            "Default: 'all'."
        ),
    )
    parser.add_argument("--metric", default="AUROC")
    args = parser.parse_args()

    time_switch = TIME_SWITCH
    balancing = BALANCING
    ma_win = MA_WIN
    classification_type = args.classification or ("bysub" + time_switch)
    task_set = args.task_set
    metric = args.metric

    path_compare_out = os.path.join(RESULTS_ROOT, "denoising_comparison")
    os.makedirs(path_compare_out, exist_ok=True)

    # Load results for each method
    results = {}
    for method in DENOISING_METHODS:
        print(f"Loading results for {method}...")
        general_info, df_metrics, df_duration = load_merged_results(
            method, classification_type, balancing, ma_win, time_switch, task_set
        )
        if df_metrics is not None:
            valid_classifiers = df_metrics["model"].unique().tolist()
            results[method] = {
                "general_info": general_info,
                "df_metrics": df_metrics,
                "df_duration": df_duration,
                "valid_classifiers": valid_classifiers,
            }

    if not results:
        print(
            "[ERROR] No results found. Run classify_main.py and merge_results.py first."
        )
        return

    # Get common classifiers
    common = set(list(results.values())[0]["valid_classifiers"])
    for r in list(results.values())[1:]:
        common = common.intersection(set(r["valid_classifiers"]))
    classifiers = sorted(common)

    create_comparison_barplot(
        results, classifiers, metric, path_compare_out, classification_type, balancing
    )
    perform_statistical_tests(results, classifiers, metric, path_compare_out)

    print(f"\n=== Comparison complete. Results saved to {path_compare_out} ===")


if __name__ == "__main__":
    main()
