import os

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Config import FIGURE_DIR, MODEL_DIR
from Utils import load_artifact, section
from Evaluation import calculate_detailed_metrics, plot_confusion_matrix
from TreeModels import majority_vote


class FinalEnsemble:
    def __init__(self, models, model_names):
        self.models = models
        self.model_names = model_names

    def predict(self, predict_fn_map):
        preds_per_model = [predict_fn_map[name](self.models[name]) for name in self.model_names]
        return majority_vote(*preds_per_model)


def main():
    valid_simplified = load_artifact("valid_simplified")

    model_accuracy_map = {
        "Decision Tree": load_artifact("dt_accuracy"),
        "Random Forest": load_artifact("rf_accuracy"),
        "Logistic Regression": load_artifact("lr_accuracy"),
        "SVM": load_artifact("svm_accuracy"),
        "KNN": load_artifact("knn_accuracy"),
        "Gaussian Naive Bayes": load_artifact("gnb_accuracy"),
        "AdaBoost": load_artifact("ada_accuracy"),
        "Gradient Boosting": load_artifact("gb_accuracy"),
        "DNN": load_artifact("dnn_accuracy"),
        "LSTM": load_artifact("lstm_accuracy"),
        "RNN": load_artifact("rnn_accuracy"),
        "All Models Ensemble": load_artifact("all_ensemble_accuracy"),
        "Top 3 Models Ensemble": load_artifact("top_ensemble_accuracy"),
    }

    model_comparison = pd.DataFrame(
        {"Model": list(model_accuracy_map.keys()),
         "Accuracy": list(model_accuracy_map.values())}
    ).sort_values("Accuracy", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(model_comparison)))
    ax.barh(model_comparison["Model"][::-1], model_comparison["Accuracy"][::-1], color=colors[::-1])
    for i, (model, acc) in enumerate(zip(model_comparison["Model"][::-1], model_comparison["Accuracy"][::-1])):
        ax.text(acc + 0.005, i, f"{acc:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, model_comparison["Accuracy"].max() * 1.1)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Model")
    ax.set_title("Model Performance Comparison\nAccuracy of Different Models for Loan Prediction")
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_model_comparison_final.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    # Confusion matrices for top 3 models
    preds_map = {
        "Decision Tree": load_artifact("dt_preds"),
        "Random Forest": load_artifact("rf_preds"),
        "Logistic Regression": load_artifact("lr_preds"),
        "SVM": load_artifact("svm_preds"),
        "KNN": load_artifact("knn_preds"),
        "Gaussian Naive Bayes": load_artifact("gnb_preds"),
        "AdaBoost": load_artifact("ada_preds"),
        "Gradient Boosting": load_artifact("gb_preds"),
        "DNN": load_artifact("dnn_preds"),
        "LSTM": load_artifact("lstm_preds"),
        "RNN": load_artifact("rnn_preds"),
        "All Models Ensemble": load_artifact("all_ensemble_preds"),
        "Top 3 Models Ensemble": load_artifact("top_ensemble_preds"),
    }

    actual = valid_simplified["Loan_Status"].astype(str).values
    top_3_models = model_comparison["Model"].head(3).tolist()

    for model_name in top_3_models:
        preds = preds_map[model_name]
        metrics = calculate_detailed_metrics(preds, actual)
        safe_name = model_name.lower().replace(" ", "_")
        plot_confusion_matrix(metrics["confusion_matrix"], f"{model_name} Confusion Matrix",
                               f"fig_final_confmat_{safe_name}.png")

    top_models = load_artifact("top_models")  
    model_objs = {}
    for key in top_models:
        try:
            model_objs[key] = load_artifact(f"{key}_model")
        except FileNotFoundError:
            pass

    final_ensemble = FinalEnsemble(models=model_objs, model_names=list(model_objs.keys()))
    joblib.dump(final_ensemble, os.path.join(MODEL_DIR, "final_loan_prediction_ensemble.joblib"))
    print(f"Saved final ensemble -> {os.path.join(MODEL_DIR, 'final_loan_prediction_ensemble.joblib')}")

    print("\nFinal model comparison table:")
    print(model_comparison.to_string(index=False))

    return model_comparison


if __name__ == "__main__":
    main()