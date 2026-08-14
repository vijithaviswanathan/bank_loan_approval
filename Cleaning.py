import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Config import CAT_VARS, NUM_VARS, FIGURE_DIR
from Utils import load_artifact, save_artifact


def inspect_missing(train_data, test_data):
    print("Missing values in training data:")
    print(train_data.isna().sum())
    print("\nMissing values in test data:")
    print(test_data.isna().sum())

    train_missing_pct = train_data.isna().sum() / len(train_data) * 100
    test_missing_pct = test_data.isna().sum() / len(test_data) * 100

    print("\nMissing values percentage in training data:")
    for col, pct in train_missing_pct.items():
        if pct > 0:
            print(f"{col}: {pct:.2f}%")

    print("\nMissing values percentage in test data:")
    for col, pct in test_missing_pct.items():
        if pct > 0:
            print(f"{col}: {pct:.2f}%")


def figure_7a_before(train_data):
    attribute_order = ["Loan_ID", "Gender", "Married", "Dependents", "Education",
                        "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
                        "LoanAmount", "Loan_Amount_Term", "Credit_History",
                        "Property_Area", "Loan_Status"]
    missing_counts = train_data.isna().sum()
    ordered = [c for c in attribute_order if c in missing_counts.index]
    values = [missing_counts[c] for c in ordered]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(ordered, values, color="steelblue", height=0.6)
    ax.set_title("Null Value Check")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Attribute_Name")
    ax.set_xlim(0, 70)
    ax.invert_yaxis()
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig7a_missing_before.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def figure_7b_after(train_data):
    attribute_order = ["Loan_ID", "Gender", "Married", "Dependents", "Education",
                        "Self_Employed", "ApplicantIncome", "CoapplicantIncome",
                        "LoanAmount", "Loan_Amount_Term", "Credit_History",
                        "Property_Area", "Loan_Status"]
    cols = [c for c in attribute_order if c in train_data.columns]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(cols, [0] * len(cols), color="darkblue")
    ax.set_title("Null Value Check")
    ax.set_xlabel("Attribute_Name")
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig7b_missing_after.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def impute_missing_values(train_data, test_data):
    train_data = train_data.copy()
    test_data = test_data.copy()

    # Categorical: mode imputation
    for var in CAT_VARS:
        if train_data[var].isna().sum() > 0:
            mode_val = train_data[var].mode(dropna=True).iloc[0]
            print(f"Change missing values in {var} using mode: {mode_val}")
            train_data[var] = train_data[var].fillna(mode_val)
        if test_data[var].isna().sum() > 0:
            mode_val = train_data[var].mode(dropna=True).iloc[0]
            print(f"Change missing values in test {var} using mode: {mode_val}")
            test_data[var] = test_data[var].fillna(mode_val)

    # Numeric: mean imputation 
    for var in NUM_VARS:
        if train_data[var].isna().sum() > 0:
            mean_val = train_data[var].mean(skipna=True)
            print(f"Change missing values in {var} using mean: {mean_val:.2f}")
            train_data[var] = train_data[var].fillna(mean_val)
        if test_data[var].isna().sum() > 0:
            mean_val = train_data[var].mean(skipna=True)
            print(f"Change missing values in test {var} using mean: {mean_val:.2f}")
            test_data[var] = test_data[var].fillna(mean_val)

    return train_data, test_data


def main():
    train_data = load_artifact("train_data")
    test_data = load_artifact("test_data")

    inspect_missing(train_data, test_data)
    figure_7a_before(train_data)

    train_data, test_data = impute_missing_values(train_data, test_data)
    figure_7b_after(train_data)

    print("\nMissing values after imputation in train data:")
    print(train_data.isna().sum())
    print("\nMissing values after imputation in test data:")
    print(test_data.isna().sum())

    save_artifact(train_data, "train_data")
    save_artifact(test_data, "test_data")

    return train_data, test_data


if __name__ == "__main__":
    main()
