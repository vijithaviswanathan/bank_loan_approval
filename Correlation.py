import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns

from Config import CORR_COLS, FIGURE_DIR
from Utils import load_artifact


def create_corr_matrix_plot(train_df):
    df = train_df[CORR_COLS].apply(lambda c: c.astype(float))
    corr_matrix = df.corr()

    my_colors = LinearSegmentedColormap.from_list(
        "r_style", ["#4B0082", "#9370DB", "#FFFFFF", "#ADD8E6", "#00008B"], N=100
    )

    fig, ax = plt.subplots(figsize=(10, 9))
    sns.heatmap(
        corr_matrix,
        cmap=my_colors,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 7, "color": "black"},
        square=True,
        linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.set_title("Fig. 10. Correlation Matrix.", pad=20)

    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig10_correlation_matrix.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    return corr_matrix


def main():
    train_df = load_artifact("train_df")
    corr_matrix = create_corr_matrix_plot(train_df)
    print(corr_matrix)
    return corr_matrix


if __name__ == "__main__":
    main()