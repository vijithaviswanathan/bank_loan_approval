import pandas as pd

from Config import CAT_COLS_TO_ENCODE, TARGET_COL
from Utils import load_artifact, save_artifact, section


def encode_target(train_df):
    train_df = train_df.copy()
    if TARGET_COL in train_df.columns:
        print("Converting Loan_Status: Y -> 1, N -> 0")
        train_df[TARGET_COL] = train_df[TARGET_COL].apply(lambda v: 1 if v == "Y" else 0)
        train_df[TARGET_COL] = train_df[TARGET_COL].astype("category")
    return train_df


def encode_categoricals(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    print("Categorical variables to factors then integers:")
    for col in CAT_COLS_TO_ENCODE:
        print(f"- {col}: {', '.join(map(str, train_df[col].unique()))}")

        train_df[col] = train_df[col].astype("category")
        # Fit categories on TRAIN, apply the exact same categories to TEST
        train_categories = train_df[col].cat.categories
        test_df[col] = pd.Categorical(test_df[col], categories=train_categories)

        # R's as.integer(factor) is 1-indexed -> replicate with +1
        train_df[f"{col}_Encoded"] = train_df[col].cat.codes + 1
        test_df[f"{col}_Encoded"] = test_df[col].cat.codes + 1

    return train_df, test_df


def main():
    train_df = load_artifact("train_df")
    test_df = load_artifact("test_df")

    train_df = encode_target(train_df)
    train_df, test_df = encode_categoricals(train_df, test_df)

    save_artifact(train_df, "train_df")
    save_artifact(test_df, "test_df")

    return train_df, test_df


if __name__ == "__main__":
    main()