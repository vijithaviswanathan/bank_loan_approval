import re

import numpy as np
import pandas as pd

from Config import MODEL_FEATURE_COLS
from Utils import load_artifact, save_artifact, section


def relabel_loan_status(train_subset, valid_subset):
    train_subset = train_subset.copy()
    valid_subset = valid_subset.copy()

    mapping = {0: "No", 1: "Yes", "0": "No", "1": "Yes"}
    train_subset["Loan_Status"] = train_subset["Loan_Status"].astype(str).astype(int).map({0: "No", 1: "Yes"})
    valid_subset["Loan_Status"] = valid_subset["Loan_Status"].astype(str).astype(int).map({0: "No", 1: "Yes"})

    train_subset["Loan_Status"] = pd.Categorical(train_subset["Loan_Status"], categories=["No", "Yes"])
    valid_subset["Loan_Status"] = pd.Categorical(valid_subset["Loan_Status"], categories=["No", "Yes"])

    print("Updated Loan_Status levels:", list(train_subset["Loan_Status"].cat.categories))
    return train_subset, valid_subset


def clean_dependents_and_drop_encoded(train_subset, valid_subset):
    train_subset = train_subset.copy()
    valid_subset = valid_subset.copy()

    print("Unique values in Dependents column:", train_subset["Dependents"].unique())

    train_subset["Dependents"] = train_subset["Dependents"].astype(str).str.replace(r"3\+", "3", regex=True)
    valid_subset["Dependents"] = valid_subset["Dependents"].astype(str).str.replace(r"3\+", "3", regex=True)

    train_subset["Dependents"] = pd.Categorical(train_subset["Dependents"])
    valid_subset["Dependents"] = pd.Categorical(valid_subset["Dependents"],
                                                 categories=train_subset["Dependents"].cat.categories)

    encoded_cols = [c for c in train_subset.columns if c.endswith("_Encoded")]
    if encoded_cols:
        train_subset = train_subset.drop(columns=encoded_cols)
        valid_subset = valid_subset.drop(columns=[c for c in encoded_cols if c in valid_subset.columns])

    cat_vars = ["Gender", "Married", "Dependents", "Education",
                "Self_Employed", "Property_Area", "Credit_History"]
    for var in cat_vars:
        if not isinstance(train_subset[var].dtype, pd.CategoricalDtype):
            train_subset[var] = pd.Categorical(train_subset[var])
        if not isinstance(valid_subset[var].dtype, pd.CategoricalDtype):
            valid_subset[var] = pd.Categorical(valid_subset[var], categories=train_subset[var].cat.categories)

    return train_subset, valid_subset


def impute_remaining(train_subset, valid_subset):
    train_subset = train_subset.copy()
    valid_subset = valid_subset.copy()

    print("Missing values check:")
    print(train_subset.isna().sum())

    for col in train_subset.columns:
        n_missing = train_subset[col].isna().sum()
        if n_missing > 0:
            if pd.api.types.is_numeric_dtype(train_subset[col]):
                mean_val = train_subset[col].mean(skipna=True)
                train_subset[col] = train_subset[col].fillna(mean_val)
                if col in valid_subset.columns:
                    valid_subset[col] = valid_subset[col].fillna(mean_val)
                print(f"Imputed missing values in numeric column {col} with mean: {mean_val}")
            else:
                mode_val = train_subset[col].mode(dropna=True).iloc[0]
                train_subset[col] = train_subset[col].fillna(mode_val)
                if col in valid_subset.columns:
                    valid_subset[col] = valid_subset[col].fillna(mode_val)
                print(f"Imputed missing values in categorical column {col} with mode: {mode_val}")

    train_subset["Dependents"] = train_subset["Dependents"].astype(str).str.strip()
    valid_subset["Dependents"] = valid_subset["Dependents"].astype(str).str.strip()
    train_subset["Dependents"] = train_subset["Dependents"].replace("3+", "3")
    valid_subset["Dependents"] = valid_subset["Dependents"].replace("3+", "3")
    train_subset["Dependents"] = pd.to_numeric(train_subset["Dependents"], errors="coerce")
    valid_subset["Dependents"] = pd.to_numeric(valid_subset["Dependents"], errors="coerce")

    print("After imputation, missing values:")
    print(train_subset.isna().sum())
    print(valid_subset.isna().sum())

    return train_subset, valid_subset


def drop_na_rows(train_subset, valid_subset):
    na_rows = train_subset[train_subset.isna().any(axis=1)]
    print("Rows with NA values:", len(na_rows))

    train_subset_clean = train_subset.dropna().reset_index(drop=True)

    valid_na_rows = valid_subset[valid_subset.isna().any(axis=1)]
    if len(valid_na_rows) > 0:
        valid_subset_clean = valid_subset.dropna().reset_index(drop=True)
    else:
        valid_subset_clean = valid_subset.reset_index(drop=True)

    return train_subset_clean, valid_subset_clean


def prepare_data(data):
    data = data.copy()

    data["Gender"] = data["Gender"].astype(str)
    data.loc[data["Gender"] == "", "Gender"] = "Male"
    data["Gender"] = pd.Categorical(data["Gender"])

    data["Married"] = data["Married"].astype(str)
    data.loc[data["Married"] == "", "Married"] = "Yes"
    data["Married"] = pd.Categorical(data["Married"])

    data["Self_Employed"] = data["Self_Employed"].astype(str)
    data.loc[data["Self_Employed"] == "", "Self_Employed"] = "No"
    data["Self_Employed"] = pd.Categorical(data["Self_Employed"])

    if "Dependents" in data.columns:
        data["Dependents"] = data["Dependents"].astype(str).str.strip()
        data["Dependents"] = data["Dependents"].replace("3+", "3")
        data["Dependents"] = pd.to_numeric(data["Dependents"], errors="coerce")

    data["Credit_History"] = pd.Categorical(data["Credit_History"])

    return data


def main():
    train_subset = load_artifact("train_subset")
    valid_subset = load_artifact("valid_subset")

    train_subset, valid_subset = relabel_loan_status(train_subset, valid_subset)
    train_subset, valid_subset = clean_dependents_and_drop_encoded(train_subset, valid_subset)
    train_subset, valid_subset = impute_remaining(train_subset, valid_subset)
    train_subset_clean, valid_subset_clean = drop_na_rows(train_subset, valid_subset)

    train_simplified = prepare_data(train_subset_clean)
    valid_simplified = prepare_data(valid_subset_clean)

    print("train_simplified shape:", train_simplified.shape)
    print("valid_simplified shape:", valid_simplified.shape)

    save_artifact(train_simplified, "train_simplified")
    save_artifact(valid_simplified, "valid_simplified")

    return train_simplified, valid_simplified


if __name__ == "__main__":
    main()