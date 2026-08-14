import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from Utils import load_artifact, save_artifact

RNG = np.random.default_rng() 

def smote_sampling(minority_class, n_synthetic, k=5, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    numeric_cols = [c for c in minority_class.columns
                     if pd.api.types.is_numeric_dtype(minority_class[c])
                     and c != "Loan_Status"]
    numeric_data = minority_class[numeric_cols].reset_index(drop=True)

    if len(numeric_data) < 2:
        raise ValueError("Need at least 2 minority samples for SMOTE.")

    dist_matrix = cdist(numeric_data.values, numeric_data.values, metric="euclidean")
    np.fill_diagonal(dist_matrix, np.inf)

    minority_class = minority_class.reset_index(drop=True)
    synthetic_rows = []

    n_rows = len(minority_class)
    k_eff = min(k, n_rows - 1)

    for _ in range(n_synthetic):
        idx = rng.integers(0, n_rows)
        nhbrs = np.argsort(dist_matrix[idx])[:k_eff]
        nhbr_idx = rng.choice(nhbrs)

        new_sample = minority_class.iloc[idx].copy()
        alpha = rng.uniform(0, 1)
        for col in numeric_cols:
            new_sample[col] = (minority_class.iloc[idx][col] +
                                alpha * (minority_class.iloc[nhbr_idx][col] -
                                         minority_class.iloc[idx][col]))
        synthetic_rows.append(new_sample)

    return pd.DataFrame(synthetic_rows).reset_index(drop=True)


def balance_with_smote(train_df, seed=None):
    rng = np.random.default_rng(seed)

    print("Class distribution before balancing:")
    print(train_df["Loan_Status"].value_counts())

    class_0 = train_df[train_df["Loan_Status"] == 0]
    class_1 = train_df[train_df["Loan_Status"] == 1]

    n_synthetic = len(class_1) - len(class_0)
    synthetic_samples = smote_sampling(class_0, n_synthetic, rng=rng)
    balanced_df = pd.concat([train_df, synthetic_samples], ignore_index=True)

    print("\nClass distribution after balancing:")
    print(balanced_df["Loan_Status"].value_counts())

    before_counts = train_df["Loan_Status"].value_counts()
    after_counts = balanced_df["Loan_Status"].value_counts()
    before_ratio = before_counts.min() / before_counts.max()
    after_ratio = after_counts.min() / after_counts.max()
    print(f"\nClass balance improved from {before_ratio:.2f} to {after_ratio:.2f}")

    return balanced_df


def main():
    train_df = load_artifact("train_df")

    balanced_df = balance_with_smote(train_df)
    train_df = balanced_df

    save_artifact(train_df, "train_df")
    return train_df


if __name__ == "__main__":
    main()