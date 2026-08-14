import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping

from Config import SEED_DL
from Utils import load_artifact, save_artifact, section
from TreeModels import majority_vote

tf.random.set_seed(SEED_DL)
np.random.seed(SEED_DL)


def prepare_data_dl(data, categorical_vars):
    dl_data = data.copy()

    for var in categorical_vars:
        dummies = pd.get_dummies(dl_data[var], prefix=var)
        dl_data = pd.concat([dl_data, dummies], axis=1)

    dl_data = dl_data.drop(columns=categorical_vars)

    if "Loan_Status" in dl_data.columns:
        dl_data["Loan_Status"] = dl_data["Loan_Status"].map({"No": 0, "Yes": 1}).astype(float)

    return dl_data


def build_dnn(input_shape):
    model = Sequential([
        Dense(64, activation="relu", input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(input_shape):
    model = Sequential([
        LSTM(32, input_shape=(1, input_shape), return_sequences=True),
        Dropout(0.3),
        LSTM(16),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_rnn(input_shape):
    model = Sequential([
        SimpleRNN(32, input_shape=(1, input_shape), return_sequences=True),
        Dropout(0.3),
        SimpleRNN(16),
        Dropout(0.2),
        Dense(8, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def calculate_metrics(predictions, actual):
    pred_numeric = (np.asarray(predictions) == "Yes").astype(int)
    actual_numeric = (np.asarray(actual) == "Yes").astype(int)

    true_pos = np.sum((pred_numeric == 1) & (actual_numeric == 1))
    false_pos = np.sum((pred_numeric == 1) & (actual_numeric == 0))
    true_neg = np.sum((pred_numeric == 0) & (actual_numeric == 0))
    false_neg = np.sum((pred_numeric == 0) & (actual_numeric == 1))

    precision_yes = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else np.nan
    recall_yes = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else np.nan
    f1_yes = (2 * precision_yes * recall_yes / (precision_yes + recall_yes)
              if precision_yes and recall_yes else np.nan)

    precision_no = true_neg / (true_neg + false_neg) if (true_neg + false_neg) else np.nan
    recall_no = true_neg / (true_neg + false_pos) if (true_neg + false_pos) else np.nan
    f1_no = (2 * precision_no * recall_no / (precision_no + recall_no)
             if precision_no and recall_no else np.nan)

    return {
        "precision": {"yes": precision_yes, "no": precision_no},
        "recall": {"yes": recall_yes, "no": recall_no},
        "f1_score": {"yes": f1_yes, "no": f1_no},
    }


def main():
    train_simplified = load_artifact("train_simplified")
    valid_simplified = load_artifact("valid_simplified")

    categorical_vars = ["Gender", "Married", "Education",
                         "Self_Employed", "Credit_History", "Property_Area"]

    train_dl = prepare_data_dl(train_simplified, categorical_vars)
    valid_dl = prepare_data_dl(valid_simplified, categorical_vars)
    valid_dl = valid_dl.reindex(columns=train_dl.columns, fill_value=0)

    x_train = train_dl.drop(columns=["Loan_Status"]).astype(float).values
    y_train = train_dl["Loan_Status"].values

    x_valid = valid_dl.drop(columns=["Loan_Status"]).astype(float).values
    y_valid = valid_dl["Loan_Status"].values

    input_shape = x_train.shape[1]
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    # 1. DNN
    dnn_model = build_dnn(input_shape)
    dnn_model.fit(x_train, y_train, epochs=50, batch_size=32,
                  validation_split=0.2, callbacks=[early_stop], verbose=0)

    # 2. LSTM 
    x_train_lstm = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_valid_lstm = x_valid.reshape((x_valid.shape[0], 1, x_valid.shape[1]))

    lstm_model = build_lstm(input_shape)
    lstm_model.fit(x_train_lstm, y_train, epochs=50, batch_size=32,
                   validation_split=0.2, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                   verbose=0)

    # 3. Simple RNN
    rnn_model = build_rnn(input_shape)
    rnn_model.fit(x_train_lstm, y_train, epochs=50, batch_size=32,
                  validation_split=0.2, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                  verbose=0)

    dnn_probs = dnn_model.predict(x_valid, verbose=0)
    dnn_preds = np.where(dnn_probs.ravel() > 0.5, "Yes", "No")

    lstm_probs = lstm_model.predict(x_valid_lstm, verbose=0)
    lstm_preds = np.where(lstm_probs.ravel() > 0.5, "Yes", "No")

    rnn_probs = rnn_model.predict(x_valid_lstm, verbose=0)
    rnn_preds = np.where(rnn_probs.ravel() > 0.5, "Yes", "No")

    y_valid_labels = np.where(y_valid == 1, "Yes", "No")

    dnn_accuracy = (dnn_preds == y_valid_labels).mean()
    lstm_accuracy = (lstm_preds == y_valid_labels).mean()
    rnn_accuracy = (rnn_preds == y_valid_labels).mean()

    print(f"DNN Accuracy: {dnn_accuracy}")
    print(f"LSTM Accuracy: {lstm_accuracy}")
    print(f"RNN Accuracy: {rnn_accuracy}")

    # Add DL preds to all models ensemble
    dt_preds = load_artifact("dt_preds")
    rf_preds = load_artifact("rf_preds")
    lr_preds = load_artifact("lr_preds")
    svm_preds = load_artifact("svm_preds")
    knn_preds = load_artifact("knn_preds")
    gnb_preds = load_artifact("gnb_preds")
    ada_preds = load_artifact("ada_preds")
    gb_preds = load_artifact("gb_preds")

    all_ensemble_dl_preds = majority_vote(
        dt_preds, rf_preds, lr_preds, svm_preds, knn_preds, gnb_preds,
        ada_preds, gb_preds, dnn_preds, lstm_preds, rnn_preds,
    )
    all_ensemble_dl_accuracy = (all_ensemble_dl_preds == y_valid_labels).mean()
    print(f"All Models with DL Ensemble Accuracy: {all_ensemble_dl_accuracy}")

    all_ensemble_preds = load_artifact("all_ensemble_preds")
    top_ensemble_preds = load_artifact("top_ensemble_preds")
    actual_labels = valid_simplified["Loan_Status"].astype(str).values

    metrics_results = {
        "dt": calculate_metrics(dt_preds, actual_labels),
        "rf": calculate_metrics(rf_preds, actual_labels),
        "lr": calculate_metrics(lr_preds, actual_labels),
        "svm": calculate_metrics(svm_preds, actual_labels),
        "knn": calculate_metrics(knn_preds, actual_labels),
        "gnb": calculate_metrics(gnb_preds, actual_labels),
        "ada": calculate_metrics(ada_preds, actual_labels),
        "gb": calculate_metrics(gb_preds, actual_labels),
        "dnn": calculate_metrics(dnn_preds, actual_labels),
        "lstm": calculate_metrics(lstm_preds, actual_labels),
        "rnn": calculate_metrics(rnn_preds, actual_labels),
        "all_ensemble": calculate_metrics(all_ensemble_preds, actual_labels),
        "top_ensemble": calculate_metrics(top_ensemble_preds, actual_labels),
        "all_ensemble_dl": calculate_metrics(all_ensemble_dl_preds, actual_labels),
    }

    rows = []
    for model, m in metrics_results.items():
        rows.append({
            "Model": model,
            "Precision_Yes": round(m["precision"]["yes"], 2),
            "Precision_No": round(m["precision"]["no"], 2),
            "Recall_Yes": round(m["recall"]["yes"], 2),
            "Recall_No": round(m["recall"]["no"], 2),
            "F1_Score_Yes": round(m["f1_score"]["yes"], 2),
            "F1_Score_No": round(m["f1_score"]["no"], 2),
        })
    metrics_table = pd.DataFrame(rows)
    print(metrics_table)

    for name, obj in [
        ("dnn_model", dnn_model), ("lstm_model", lstm_model), ("rnn_model", rnn_model),
        ("dnn_preds", dnn_preds), ("lstm_preds", lstm_preds), ("rnn_preds", rnn_preds),
        ("dnn_accuracy", dnn_accuracy), ("lstm_accuracy", lstm_accuracy), ("rnn_accuracy", rnn_accuracy),
        ("all_ensemble_dl_preds", all_ensemble_dl_preds),
        ("all_ensemble_dl_accuracy", all_ensemble_dl_accuracy),
        ("metrics_results", metrics_results), ("metrics_table", metrics_table),
    ]:
        save_artifact(obj, name)

    return metrics_table


if __name__ == "__main__":
    main()