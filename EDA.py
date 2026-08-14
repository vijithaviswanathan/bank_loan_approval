import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Config import ORIGINAL_COLS, FIGURE_DIR, TARGET_COL
from Utils import load_artifact, section


def figure_6a_attribute_overview(train_data):
    filtered_cols = [c for c in ORIGINAL_COLS if c in train_data.columns]
    filtered_summary = train_data[filtered_cols].notna().sum()

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(range(len(filtered_summary)), filtered_summary.values,
                   color="skyblue", edgecolor="gold")
    ax.set_ylim(0, 850)
    ax.set_title("Dataset Overview on Amount")
    ax.set_ylabel("Amount")
    ax.set_xticks(range(len(filtered_summary)))
    ax.set_xticklabels(filtered_cols, rotation=45, ha="right")
    ax.set_xlabel("")
    for rect, val in zip(bars, filtered_summary.values):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 10, str(val),
                ha="center", va="bottom", fontsize=8)
    fig.text(0.5, 0.02, "Attribute_Name", ha="center")
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig6a_dataset_overview.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def figure_6b_loan_status_pie(train_data):
    loan_status_counts = train_data[TARGET_COL].value_counts()

    fig, ax = plt.subplots(figsize=(6, 6))
    labels = [f"{idx} : {cnt}" for idx, cnt in loan_status_counts.items()]
    ax.pie(loan_status_counts.values, labels=labels,
           colors=["royalblue", "gold"], startangle=90)
    ax.set_title("Loan_Status")
    ax.legend(labels=["Yes", "No"], loc="upper right")
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig6b_loan_status_pie.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    train_data = load_artifact("train_data")
    figure_6a_attribute_overview(train_data)
    figure_6b_loan_status_pie(train_data)


if __name__ == "__main__":
    main()