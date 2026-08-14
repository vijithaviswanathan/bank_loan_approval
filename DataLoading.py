import pandas as pd

from Config import TRAIN_RAW_PATH, TEST_RAW_PATH, TARGET_COL
from Utils import save_artifact


def load_data(train_path=TRAIN_RAW_PATH, test_path=TEST_RAW_PATH):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print("Train columns:", ", ".join(train.columns))
    print("\nTest columns:", ", ".join(test.columns))

    if TARGET_COL in train.columns:
        train[TARGET_COL] = train[TARGET_COL].astype("category")

    print(f"\nTrain shape: {train.shape[0]} rows, {train.shape[1]} columns")
    print(f"\nTest shape: {test.shape[0]} rows, {test.shape[1]} columns")

    return train, test


def main():
    train_data, test_data = load_data()

    save_artifact(train_data, "train_data")
    save_artifact(test_data, "test_data")

    return train_data, test_data


if __name__ == "__main__":
    main()