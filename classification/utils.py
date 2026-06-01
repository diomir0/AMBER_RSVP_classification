"""
Utility and plotting functions for the AMBER ERP classification pipeline.

Adapted from VEP_classification_comp/utils.py.
"""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.titlesize"] = 10
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 6
plt.rcParams["lines.linewidth"] = 1

colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:cyan",
    "tab:olive",
    "tab:brown",
    "tab:gray",
]
color_cycle = cycler(color=colors)
plt.rcParams["legend.labelcolor"] = "black"
plt.rcParams["axes.prop_cycle"] = color_cycle


def plot_features_notime(
    data,
    labels,
    feature_names,
    label_names,
    classification_type,
    fig_path=None,
    data_version=None,
):
    """Plot statistical ERP parameters grouped by class (violin plots)."""
    nFeatures = data.shape[1]

    # Clean feature names
    feature_names_edit = [fn.replace("_", ": ") for fn in feature_names]

    fig, axs = plt.subplots(2, max(4, nFeatures // 2), figsize=(max(8, nFeatures), 5))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    unique_labels = np.unique(labels)
    for i, ax in enumerate(fig.axes[:nFeatures]):
        if i >= nFeatures:
            ax.axis("off")
            continue
        data_groups = [data[labels == lbl, i] for lbl in unique_labels]
        vp = ax.violinplot(
            data_groups,
            positions=range(1, len(unique_labels) + 1),
            showmeans=False,
            showextrema=False,
        )
        for vph in vp["bodies"]:
            vph.set_alpha(0.7)
        quartile1 = [np.percentile(g, [25]) for g in data_groups]
        medians = [np.percentile(g, [50]) for g in data_groups]
        quartile3 = [np.percentile(g, [75]) for g in data_groups]
        ax.scatter(
            range(1, len(unique_labels) + 1),
            medians,
            marker="o",
            color="k",
            s=5,
            zorder=3,
        )
        for j in range(len(unique_labels)):
            ax.vlines(j + 1, quartile1[j], quartile3[j], color="k", linestyle="-", lw=1)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(feature_names_edit[i], fontsize=7, pad=2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(direction="in", labelsize=6)

    if fig_path:
        plt.savefig(
            os.path.join(fig_path, f"plot_features_dat{data_version}.png"),
            bbox_inches="tight",
        )
    plt.close("all")


def plot_metrics_averaged(
    df_metrics,
    clfs_included,
    metrics_included,
    classification_type,
    fig_path=None,
    data_version=None,
):
    """Plot evaluation metrics averaged over subjects."""
    if "_notime" in classification_type:
        df_avg = (
            df_metrics.drop("time", axis=1, errors="ignore")
            .groupby("model", as_index=False)[metrics_included]
            .mean()
        )
    else:
        df_avg = df_metrics.groupby(["time", "model"], as_index=False)[
            metrics_included
        ].mean()

    nMetrics = len(metrics_included)
    fig, axs = plt.subplots(1, nMetrics, figsize=(3 * nMetrics, 4))
    if nMetrics == 1:
        axs = [axs]
    plt.subplots_adjust(wspace=0.4)

    for im, metric in enumerate(metrics_included):
        if "_notime" in classification_type:
            sns.scatterplot(
                x=range(len(clfs_included)),
                y=metric,
                hue="model",
                data=df_avg,
                ax=axs[im],
                s=40,
                hue_order=clfs_included,
            )
            axs[im].set_xticks(range(len(clfs_included)))
            axs[im].set_xticklabels(clfs_included, rotation=45, ha="right", fontsize=6)
        else:
            sns.lineplot(
                x="time",
                y=metric,
                hue="model",
                data=df_avg,
                ax=axs[im],
                hue_order=clfs_included,
                lw=0.7,
            )
        axs[im].set_title(metric, fontsize=8)
        if metric in ["Accuracy", "AUROC"]:
            axs[im].axhline(y=0.5, color="gray", ls="--", lw=0.5)

    if fig_path:
        plt.savefig(
            os.path.join(fig_path, f"plot_metrics_{data_version}.png"),
            bbox_inches="tight",
        )
    plt.close("all")
