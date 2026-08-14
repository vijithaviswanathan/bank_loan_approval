from sklearn.model_selection import train_test_split

from Config import SEED_SPLIT_1, SEED_SPLIT_2, TARGET_COL
from Utils import load_artifact, save_artifact, section


def simple_random_split(train_df, seed=SEED_SPLIT_1):
    final_train, final_test = train_test_split(
        train_df, train_size=0.75, random_state=seed, shuffle=True
    )
    print(f"Final training set: {final_train.shape[0]} rows, {final_train.shape[1]} columns")
    print(f"Final testing set (validation set): {final_test.shape[0]} rows, {final_test.shape[1]} columns")
    return final_train, final_test


def stratified_split(train_df, seed=SEED_SPLIT_2):
    train_subset, valid_subset = train_test_split(
        train_df, train_size=0.75, random_state=seed, shuffle=True,
        stratify=train_df[TARGET_COL],
    )
    print(f"Training set dimensions: {train_subset.shape[0]} rows, {train_subset.shape[1]} columns")
    print(f"Validation set dimensions: {valid_subset.shape[0]} rows, {valid_subset.shape[1]} columns")
    return train_subset, valid_subset


def main():
    train_df = load_artifact("train_df")

    final_train, final_test = simple_random_split(train_df)
    save_artifact(final_train, "final_train")
    save_artifact(final_test, "final_test")

    train_subset, valid_subset = stratified_split(train_df)
    save_artifact(train_subset, "train_subset")
    save_artifact(valid_subset, "valid_subset")

    return train_subset, valid_subset


if __name__ == "__main__":
    main()