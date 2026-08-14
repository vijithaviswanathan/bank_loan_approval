import os
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from Config import MODEL_DIR, SEED_MODELS_2
from Utils import load_artifact, save_artifact
from TreeModels import majority_vote


def train_logistic_regression(X_train, y_train):
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    return lr_model


def train_svm(X_train, y_train, seed=SEED_MODELS_2):
    svm_model = SVC(kernel="rbf", probability=True, random_state=seed)
    svm_model.fit(X_train, y_train)
    return svm_model


def train_knn(X_train, y_train, X_valid):
    k_value = math.floor(math.sqrt(len(X_train)))
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(X_train_scaled, y_train)
    knn_preds = knn_model.predict(X_valid_scaled)
    return knn_model, knn_preds, scaler, k_value


def train_gaussian_nb(X_train, y_train):
    gnb_model = GaussianNB()
    gnb_model.fit(X_train, y_train)
    return gnb_model


def train_adaboost(X_train, y_train, seed=SEED_MODELS_2):
    ada_model = AdaBoostClassifier(n_estimators=50, random_state=seed)
    ada_model.fit(X_train, y_train)
    return ada_model


def train_gradient_boosting(X_train, y_train, seed=SEED_MODELS_2):
    gb_model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed
    )
    gb_model.fit(X_train, y_train)
    return gb_model


def main():
    X_train = load_artifact("X_train")
    X_valid = load_artifact("X_valid")
    train_simplified = load_artifact("train_simplified")
    valid_simplified = load_artifact("valid_simplified")
    dt_preds = load_artifact("dt_preds")
    rf_preds = load_artifact("rf_preds")
    dt_accuracy = load_artifact("dt_accuracy")
    rf_accuracy = load_artifact("rf_accuracy")

    y_train = train_simplified["Loan_Status"].astype(str)
    y_valid = valid_simplified["Loan_Status"].astype(str)

    # 1. Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    lr_preds = lr_model.predict(X_valid)

    # 2. SVM
    svm_model = train_svm(X_train, y_train)
    svm_preds = svm_model.predict(X_valid)

    # 3. KNN
    knn_model, knn_preds, knn_scaler, k_value = train_knn(X_train, y_train, X_valid)

    # 4. Gaussian Naive Bayes
    gnb_model = train_gaussian_nb(X_train, y_train)
    gnb_preds = gnb_model.predict(X_valid)

    # 5. AdaBoost
    ada_model = train_adaboost(X_train, y_train)
    ada_preds = ada_model.predict(X_valid)

    # 6. Gradient Boosting
    gb_model = train_gradient_boosting(X_train, y_train)
    gb_preds = gb_model.predict(X_valid)

    # Accuracy for each model
    lr_accuracy = (lr_preds == y_valid.values).mean()
    svm_accuracy = (svm_preds == y_valid.values).mean()
    knn_accuracy = (knn_preds == y_valid.values).mean()
    gnb_accuracy = (gnb_preds == y_valid.values).mean()
    ada_accuracy = (ada_preds == y_valid.values).mean()
    gb_accuracy = (gb_preds == y_valid.values).mean()

    print(f"Logistic Regression Accuracy: {lr_accuracy}")
    print(f"SVM Accuracy: {svm_accuracy}")
    print(f"KNN Accuracy: {knn_accuracy}")
    print(f"Gaussian Naive Bayes Accuracy: {gnb_accuracy}")
    print(f"AdaBoost Accuracy: {ada_accuracy}")
    print(f"Gradient Boosting Accuracy: {gb_accuracy}")

    # All-models majority vote ensemble
    all_ensemble_preds = majority_vote(dt_preds, rf_preds, lr_preds, svm_preds,
                                        knn_preds, gnb_preds, ada_preds, gb_preds)
    all_ensemble_accuracy = (all_ensemble_preds == y_valid.values).mean()
    print(f"All Models Ensemble Accuracy: {all_ensemble_accuracy}")

    # Top 3 models ensemble
    model_accuracies = {
        "dt": dt_accuracy, "rf": rf_accuracy, "lr": lr_accuracy, "svm": svm_accuracy,
        "knn": knn_accuracy, "gnb": gnb_accuracy, "ada": ada_accuracy, "gb": gb_accuracy,
    }
    preds_by_name = {
        "dt": dt_preds, "rf": rf_preds, "lr": lr_preds, "svm": svm_preds,
        "knn": knn_preds, "gnb": gnb_preds, "ada": ada_preds, "gb": gb_preds,
    }
    top_models = sorted(model_accuracies, key=model_accuracies.get, reverse=True)[:3]
    print(f"Top 3 models: {', '.join(top_models)}")

    top_ensemble_preds = majority_vote(*[preds_by_name[m] for m in top_models])
    top_ensemble_accuracy = (top_ensemble_preds == y_valid.values).mean()
    print(f"Top 3 Models Ensemble Accuracy: {top_ensemble_accuracy}")

    conf_matrix_ensemble = pd.crosstab(
        pd.Series(top_ensemble_preds, name="Predicted"),
        pd.Series(y_valid.values, name="Actual"),
    )
    print("Confusion Matrix for Top 3 Ensemble:")
    print(conf_matrix_ensemble)
    
    model_lookup = {"dt": None, "rf": None, "lr": lr_model, "svm": svm_model,
                     "knn": {"model": knn_model, "scaler": knn_scaler, "k": k_value},
                     "gnb": gnb_model, "ada": ada_model, "gb": gb_model}
    best_ensemble = {"models": {m: model_lookup[m] for m in top_models}, "top_models": top_models}
    joblib.dump(best_ensemble, os.path.join(MODEL_DIR, "best_ensemble_loan_prediction_model.joblib"))

    for name, obj in [
        ("lr_model", lr_model), ("svm_model", svm_model), ("knn_model", knn_model),
        ("gnb_model", gnb_model), ("ada_model", ada_model), ("gb_model", gb_model),
        ("lr_preds", lr_preds), ("svm_preds", svm_preds), ("knn_preds", knn_preds),
        ("gnb_preds", gnb_preds), ("ada_preds", ada_preds), ("gb_preds", gb_preds),
        ("lr_accuracy", lr_accuracy), ("svm_accuracy", svm_accuracy), ("knn_accuracy", knn_accuracy),
        ("gnb_accuracy", gnb_accuracy), ("ada_accuracy", ada_accuracy), ("gb_accuracy", gb_accuracy),
        ("all_ensemble_preds", all_ensemble_preds), ("all_ensemble_accuracy", all_ensemble_accuracy),
        ("top_ensemble_preds", top_ensemble_preds), ("top_ensemble_accuracy", top_ensemble_accuracy),
        ("top_models", top_models), ("knn_scaler", knn_scaler), ("k_value", k_value),
    ]:
        save_artifact(obj, name)

    return top_models, top_ensemble_accuracy

if __name__ == "__main__":
    main()