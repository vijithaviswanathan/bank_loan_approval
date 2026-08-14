import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from Config import FIGURE_DIR
from Utils import load_artifact, save_artifact


def calculate_detailed_metrics(predictions, actual):
    predictions = np.asarray(predictions)
    actual = np.asarray(actual)

    TP = np.sum((predictions == "Yes") & (actual == "Yes"))
    TN = np.sum((predictions == "No") & (actual == "No"))
    FP = np.sum((predictions == "Yes") & (actual == "No"))
    FN = np.sum((predictions == "No") & (actual == "Yes"))

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else np.nan

    precision_yes = TP / (TP + FP) if (TP + FP) else np.nan
    recall_yes = TP / (TP + FN) if (TP + FN) else np.nan
    f1_yes = (2 * precision_yes * recall_yes / (precision_yes + recall_yes)
              if precision_yes and recall_yes else np.nan)

    precision_no = TN / (TN + FN) if (TN + FN) else np.nan
    recall_no = TN / (TN + FP) if (TN + FP) else np.nan
    f1_no = (2 * precision_no * recall_no / (precision_no + recall_no)
             if precision_no and recall_no else np.nan)

    conf_matrix = pd.DataFrame(
        [[TP, FN], [FP, TN]],
        index=["Yes", "No"], columns=["Yes", "No"],
    )
    conf_matrix.index.name = "Predicted"
    conf_matrix.columns.name = "Actual"

    return {
        "confusion_matrix": conf_matrix,
        "accuracy": accuracy,
        "precision": {"yes": precision_yes, "no": precision_no},
        "recall": {"yes": recall_yes, "no": recall_no},
        "f1_score": {"yes": f1_yes, "no": f1_no},
    }


def plot_confusion_matrix(conf_matrix, title, filename):
    total = conf_matrix.values.sum()
    pct = conf_matrix / total * 100

    annot = np.empty_like(conf_matrix.values, dtype=object)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            annot[i, j] = f"{conf_matrix.values[i, j]}\n({pct.values[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(conf_matrix, annot=annot, fmt="", cmap="Blues",
                cbar=True, ax=ax, linewidths=0.5, linecolor="white")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Actual Class")
    ax.set_ylabel("Predicted Class")
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    valid_simplified = load_artifact("valid_simplified")
    actual = valid_simplified["Loan_Status"].astype(str).values

    model_info = {
        "dt": ("Decision Tree", load_artifact("dt_preds")),
        "rf": ("Random Forest", load_artifact("rf_preds")),
        "lr": ("Logistic Regression", load_artifact("lr_preds")),
        "svm": ("Support Vector Machine", load_artifact("svm_preds")),
        "knn": ("K-Nearest Neighbors", load_artifact("knn_preds")),
        "gnb": ("Gaussian Naive Bayes", load_artifact("gnb_preds")),
        "ada": ("AdaBoost", load_artifact("ada_preds")),
        "gb": ("Gradient Boosting", load_artifact("gb_preds")),
        "dnn": ("Deep Neural Network", load_artifact("dnn_preds")),
        "lstm": ("LSTM", load_artifact("lstm_preds")),
        "rnn": ("RNN", load_artifact("rnn_preds")),
        "all_ensemble": ("All Models Ensemble", load_artifact("all_ensemble_preds")),
        "top_ensemble": ("Top 3 Models Ensemble", load_artifact("top_ensemble_preds")),
    }

    metrics_list = {key: calculate_detailed_metrics(preds, actual)
                     for key, (name, preds) in model_info.items()}

    metrics_summary = pd.DataFrame([
        {
            "Model": model_info[key][0],
            "Accuracy": round(metrics_list[key]["accuracy"], 4),
            "Precision_Yes": round(metrics_list[key]["precision"]["yes"], 4),
            "Precision_No": round(metrics_list[key]["precision"]["no"], 4),
            "Recall_Yes": round(metrics_list[key]["recall"]["yes"], 4),
            "Recall_No": round(metrics_list[key]["recall"]["no"], 4),
            "F1_Score_Yes": round(metrics_list[key]["f1_score"]["yes"], 4),
            "F1_Score_No": round(metrics_list[key]["f1_score"]["no"], 4),
        }
        for key in model_info
    ])

    for key in ["rf", "ada", "gb", "top_ensemble"]:
        name = model_info[key][0]
        cm = metrics_list[key]["confusion_matrix"]
        plot_confusion_matrix(cm, f"{name} Confusion Matrix",
                               f"fig_confmat_{key}.png")

    metrics_summary = metrics_summary.sort_values("Accuracy", ascending=False)
    print(metrics_summary.to_string(index=False))

    save_artifact(metrics_summary, "metrics_summary")
    save_artifact(metrics_list, "metrics_list")

    return metrics_summary


if __name__ == "__main__":
    main()