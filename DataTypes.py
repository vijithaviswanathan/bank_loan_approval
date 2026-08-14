import numpy as np
import pandas as pd

from Config import INTEGER_COLS, CHARACTER_COLS
from Utils import load_artifact, save_artifact, section


def _r_as_integer(series):
    return series.astype(float).apply(lambda v: np.nan if pd.isna(v) else int(v)).astype("Int64")


def fix_data_types(df):
    fixed_df = df.copy()

    for col in INTEGER_COLS:
        if col in fixed_df.columns:
            if col == "Credit_History":
                fixed_df[col] = _r_as_integer(fixed_df[col].astype(str).replace("nan", np.nan).astype(float))
            else:
                fixed_df[col] = _r_as_integer(fixed_df[col])
            print(f"Converted {col} to INTEGER")

    for col in CHARACTER_COLS:
        if col in fixed_df.columns:
            mask = fixed_df[col].notna()
            fixed_df.loc[mask, col] = fixed_df.loc[mask, col].astype(str)
            print(f"Converted {col} to CHARACTER")

    print("\nNA CHECK:")
    final_na_counts = fixed_df.isna().sum()
    cols_with_final_nas = final_na_counts[final_na_counts > 0]
    if len(cols_with_final_nas) > 0:
        print("WARNING: These columns still contain NAs after conversion and imputation:")
        for col, cnt in cols_with_final_nas.items():
            print(f"- {col}: {cnt} NAs")
    else:
        print("No NAs in the dataset.")

    return fixed_df


def main():
    train_data = load_artifact("train_data")
    test_data = load_artifact("test_data")

    train_df = fix_data_types(train_data)
    test_df = fix_data_types(test_data)

    print(train_df.info())

    save_artifact(train_df, "train_df")
    save_artifact(test_df, "test_df")

    return train_df, test_df


if __name__ == "__main__":
    main()