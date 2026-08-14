import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from Config import NUM_VARS, FIGURE_DIR
from Utils import load_artifact, save_artifact


def skewness(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = np.nanmean(x)
    s = np.nanstd(x, ddof=1)  
    z = (x - m) / s
    return np.nansum(z ** 3) / n


def add_log_features(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    print("Skewness before transformation:")
    for var in NUM_VARS:
        print(f"{var}: {skewness(train_df[var]):.2f}")

    train_df["LoanAmount_Log"] = np.log1p(train_df["LoanAmount"])
    test_df["LoanAmount_Log"] = np.log1p(test_df["LoanAmount"])

    train_df["ApplicantIncome_Log"] = np.log1p(train_df["ApplicantIncome"])
    test_df["ApplicantIncome_Log"] = np.log1p(test_df["ApplicantIncome"])

    train_df["CoapplicantIncome_Log"] = np.log1p(train_df["CoapplicantIncome"])
    test_df["CoapplicantIncome_Log"] = np.log1p(test_df["CoapplicantIncome"])

    train_df["Loan_Amount_Term_Log"] = np.log1p(train_df["Loan_Amount_Term"])
    test_df["Loan_Amount_Term_Log"] = np.log1p(test_df["Loan_Amount_Term"])

    # TotalIncome feature 
    train_df["TotalIncome"] = train_df["ApplicantIncome"] + train_df["CoapplicantIncome"]
    test_df["TotalIncome"] = test_df["ApplicantIncome"] + test_df["CoapplicantIncome"]
    train_df["TotalIncome_Log"] = np.log1p(train_df["TotalIncome"])
    test_df["TotalIncome_Log"] = np.log1p(test_df["TotalIncome"])

    print("\nSkewness after log transformation:")
    for var in ["LoanAmount_Log", "ApplicantIncome_Log", "CoapplicantIncome_Log", "TotalIncome_Log"]:
        print(f"{var}: {skewness(train_df[var]):.2f}")

    return train_df, test_df


def plot_transformation(data, var_name):
    log_var = f"{var_name}_Log"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.histplot(data[var_name].dropna(), bins=30, stat="density",
                 color="steelblue", alpha=0.5, ax=axes[0])
    sns.kdeplot(data[var_name].dropna(), color="darkblue", linewidth=1.2, ax=axes[0])
    axes[0].set_title(f"Before Log Transformation ({var_name})", fontsize=10)
    axes[0].set_xlabel(var_name)
    axes[0].set_ylabel("Density")

    sns.histplot(data[log_var].dropna(), bins=30, stat="density",
                 color="steelblue", alpha=0.5, ax=axes[1])
    sns.kdeplot(data[log_var].dropna(), color="darkblue", linewidth=1.2, ax=axes[1])
    axes[1].set_title(f"After Log Transformation ({var_name})", fontsize=10)
    axes[1].set_xlabel(log_var)
    axes[1].set_ylabel("Density")

    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, f"fig8_9_{var_name}_log_transform.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    train_df = load_artifact("train_df")
    test_df = load_artifact("test_df")

    train_df, test_df = add_log_features(train_df, test_df)

    for var in ["ApplicantIncome", "CoapplicantIncome", "LoanAmount",
                "Loan_Amount_Term", "TotalIncome"]:
        plot_transformation(train_df, var)

    save_artifact(train_df, "train_df")
    save_artifact(test_df, "test_df")

    return train_df, test_df


if __name__ == "__main__":
    main()