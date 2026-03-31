#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

# run command:python Scripts/plot_accuracy_selected_windows.py \
#   --summary_csv results_final_early_models/metrics_summary.csv \
#   --out_png paper_figures/accuracy_selected_windows.png \
#   --windows 1 2 5 10 20 30 50 60 \
#   --models RF XGB \
#   --title "Accuracy vs Early Window"

font_size = 18
rc = {"text.usetex": False, "font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold",
          "font.serif": ["Palatino"], "xtick.labelsize": font_size, 'figure.figsize': (10, 8),
          "ytick.labelsize": font_size, 'axes.grid': True, 'axes.facecolor': 'white',
          'grid.linestyle': '--', 'grid.linewidth': 1.5, 'lines.linewidth': 2.5, "axes.linewidth":2.5,
          'axes.axisbelow': True}
plt.rcParams.update(rc)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary_csv",
        type=Path,
        required=True,
        help="Path to metrics_summary.csv",
    )
    ap.add_argument(
        "--out_png",
        type=Path,
        required=True,
        help="Output PNG path",
    )
    ap.add_argument(
        "--windows",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Selected windows to plot",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=["RF", "XGB"],
        help="Models to include",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="Accuracy vs Early Window",
        help="Plot title",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    syn = False

    summary_df = pd.read_csv(args.summary_csv)
    # summary_df = summary_df[summary_df["window_sec"].isin(args.windows)].copy()
    # summary_df = summary_df[summary_df["model"].isin(args.models)].copy()
    # summary_df = summary_df.sort_values(["model", "window_sec"])

    for metric in ["accuracy", "macro_f1"]:
        plt.figure(figsize=(10, 6))
        cmap = plt.get_cmap("tab10")
        for i, model_name in enumerate(sorted(summary_df["model"].unique())):
            sub = summary_df[summary_df["model"] == model_name].sort_values("window_sec")
            sub = sub.select_dtypes(include='number').groupby('window_sec', as_index=False).mean()
            plt.plot(sub["window_sec"], sub[metric], marker="o", label=model_name, color=cmap(i), alpha=0.9)
            if syn:
                plt.plot(sub["window_sec"], sub[f"{metric}_syn"], marker="*", label=f"{model_name}_syn",
                         color=cmap(i), alpha=0.5)

        plt.xlabel("Time window (s)", fontsize=24)
        plt.ylabel(metric.replace("_", " ").title(), fontsize=24)
        plt.title(f"{metric.replace('_', ' ').title()} vs Time Window", fontsize=24)
        plt.xticks(sorted(summary_df["window_sec"].unique()))
        plt.ylim([.75, .95])
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.legend(loc="best", frameon=True, fontsize=24)
        args.out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.show()
        out_file = args.out_png.parent / f"lineplot_{metric}_plot.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_file}")
        plt.close('all')

if __name__ == "__main__":
    main()