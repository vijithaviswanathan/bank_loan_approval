import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from Config import FIGURE_DIR
from Utils import load_artifact


def main():
    valid_simplified = load_artifact("valid_simplified")
    X_valid = load_artifact("X_valid")
    actual = (valid_simplified["Loan_Status"].astype(str).values == "Yes").astype(int)

    dt_model = load_artifact("dt_model")
    rf_model = load_artifact("rf_model")
    lr_model = load_artifact("lr_model")
    gb_model = load_artifact("gb_model")
    ada_model = load_artifact("ada_model")
    svm_model = load_artifact("svm_model")

    classes = list(rf_model.classes_)
    yes_idx = classes.index("Yes")

    probs = {
        "dt": dt_model.predict_proba(X_valid)[:, yes_idx],
        "rf": rf_model.predict_proba(X_valid)[:, list(rf_model.classes_).index("Yes")],
        "lr": lr_model.predict_proba(X_valid)[:, list(lr_model.classes_).index("Yes")],
        "gb": gb_model.predict_proba(X_valid)[:, list(gb_model.classes_).index("Yes")],
        "ada": ada_model.predict_proba(X_valid)[:, list(ada_model.classes_).index("Yes")],
        "svm": svm_model.predict_proba(X_valid)[:, list(svm_model.classes_).index("Yes")],
    }

    model_labels = {
        "rf": "Random Forest", "ada": "AdaBoost", "gb": "Gradient Boosting",
        "dt": "Decision Tree", "lr": "Logistic Regression", "svm": "SVM",
    }
    colors = {"rf": "blue", "ada": "red", "gb": "green",
              "dt": "orange", "lr": "purple", "svm": "brown"}

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    roc_curves = {}
    for key in ["rf", "ada", "gb", "dt", "lr", "svm"]:
        fpr, tpr, _ = roc_curve(actual, probs[key])
        roc_auc = auc(fpr, tpr)
        roc_curves[key] = (fpr, tpr, roc_auc)
        ax.plot(fpr, tpr, color=colors[key],
                label=f"{model_labels[key]} (AUC = {roc_auc:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves for Loan Prediction Models")
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_roc_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    for key, (fpr, tpr, roc_auc) in roc_curves.items():
        print(f"{model_labels[key]}: AUC = {roc_auc:.3f}")

    return roc_curves


if __name__ == "__main__":
    main()