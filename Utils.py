import os
import pickle

from Config import ARTIFACT_DIR


def save_artifact(obj, name):
    path = os.path.join(ARTIFACT_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[Utils] Saved artifact -> {path}")
    return path


def load_artifact(name):
    path = os.path.join(ARTIFACT_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[Utils] Loaded artifact <- {path}")
    return obj


def build_design_matrix(train_df, valid_df, feature_cols):
    import pandas as pd

    train_X = pd.get_dummies(train_df[feature_cols], drop_first=False)
    valid_X = pd.get_dummies(valid_df[feature_cols], drop_first=False)

    valid_X = valid_X.reindex(columns=train_X.columns, fill_value=0)

    return train_X.astype(float), valid_X.astype(float)