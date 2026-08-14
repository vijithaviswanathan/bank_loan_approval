from sklearn.preprocessing import MinMaxScaler

from Config import NUM_COLS_FOR_SCALING
from Utils import load_artifact, save_artifact


def min_max_scale(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    scaler = MinMaxScaler()
    train_df[NUM_COLS_FOR_SCALING] = scaler.fit_transform(train_df[NUM_COLS_FOR_SCALING])
    test_df[NUM_COLS_FOR_SCALING] = scaler.transform(test_df[NUM_COLS_FOR_SCALING])

    return train_df, test_df, scaler


def drop_loan_id(train_df, test_df):
    train_ids = train_df["Loan_ID"].copy()
    test_ids = test_df["Loan_ID"].copy()
    train_df = train_df.drop(columns=["Loan_ID"])
    test_df = test_df.drop(columns=["Loan_ID"])
    return train_df, test_df, train_ids, test_ids


def main():
    train_df = load_artifact("train_df")
    test_df = load_artifact("test_df")

    train_df, test_df, scaler = min_max_scale(train_df, test_df)
    train_df, test_df, train_ids, test_ids = drop_loan_id(train_df, test_df)

    print("\n===== Final data summary =====")
    print(f"Train data dimensions: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"Test data dimensions: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    print("\nPreprocessed train data (first 5 rows):")
    print(train_df.iloc[:5, :min(10, train_df.shape[1])])

    save_artifact(train_df, "train_df")
    save_artifact(test_df, "test_df")
    save_artifact(train_ids, "train_ids")
    save_artifact(test_ids, "test_ids")
    save_artifact(scaler, "minmax_scaler")

    return train_df, test_df


if __name__ == "__main__":
    main()