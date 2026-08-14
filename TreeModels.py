import os

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from Config import MODEL_FEATURE_COLS, MODEL_DIR, SEED_MODELS_1
from Utils import load_artifact, save_artifact, build_design_matrix


def train_decision_tree(X_train, y_train, seed=SEED_MODELS_1):
    dt_model = DecisionTreeClassifier(random_state=seed)
    dt_model.fit(X_train, y_train)
    return dt_model


def train_random_forest(X_train, y_train, seed=SEED_MODELS_1, n_estimators=100):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    rf_model.fit(X_train, y_train)
    return rf_model


def majority_vote(*pred_arrays):
    stacked = np.array(pred_arrays) 
    result = []
    for col in stacked.T:
        vals, counts = np.unique(col, return_counts=True)
        result.append(vals[np.argmax(counts)])
    return np.array(result)


def main():
    train_simplified = load_artifact("train_simplified")
    valid_simplified = load_artifact("valid_simplified")

    X_train, X_valid = build_design_matrix(train_simplified, valid_simplified, MODEL_FEATURE_COLS)
    y_train = train_simplified["Loan_Status"].astype(str)
    y_valid = valid_simplified["Loan_Status"].astype(str)

    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    dt_preds = dt_model.predict(X_valid)
    rf_preds = rf_model.predict(X_valid)

    dt_accuracy = (dt_preds == y_valid.values).mean()
    rf_accuracy = (rf_preds == y_valid.values).mean()

    print(f"Decision Tree Accuracy: {dt_accuracy}")
    print(f"Random Forest Accuracy: {rf_accuracy}")

    ensemble_preds = majority_vote(dt_preds, rf_preds)
    ensemble_accuracy = (ensemble_preds == y_valid.values).mean()
    print(f"Ensemble Accuracy: {ensemble_accuracy}")

    best_accuracy = max(dt_accuracy, rf_accuracy, ensemble_accuracy)
    if ensemble_accuracy == best_accuracy:
        best_model_name = "Ensemble"
        best_preds = ensemble_preds
        best_model = {"dt": dt_model, "rf": rf_model}
    elif rf_accuracy == best_accuracy:
        best_model_name = "Random Forest"
        best_preds = rf_preds
        best_model = rf_model
    else:
        best_model_name = "Decision Tree"
        best_preds = dt_preds
        best_model = dt_model

    print(f"Best model: {best_model_name} with accuracy: {best_accuracy}")

    conf_matrix = confusion_matrix(y_valid, best_preds, labels=["No", "Yes"])
    print("Confusion Matrix for Best Model:")
    print(conf_matrix)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_loan_prediction_model.joblib"))

    save_artifact(dt_model, "dt_model")
    save_artifact(rf_model, "rf_model")
    save_artifact(dt_preds, "dt_preds")
    save_artifact(rf_preds, "rf_preds")
    save_artifact(dt_accuracy, "dt_accuracy")
    save_artifact(rf_accuracy, "rf_accuracy")
    save_artifact(X_train, "X_train")
    save_artifact(X_valid, "X_valid")

    return dt_model, rf_model


if __name__ == "__main__":
    main()