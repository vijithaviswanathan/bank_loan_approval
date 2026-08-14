import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Config import FIGURE_DIR
from Utils import load_artifact, section


def plot_single_importance(importance_df, title, filename, ylabel="Importance"):
    importance_df = importance_df.sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(importance_df))))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel(ylabel)
    ax.set_ylabel("Feature")
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def normalize(x):
    x = np.asarray(x, dtype=float)
    valid = x[~np.isnan(x)]
    if len(valid) == 0:
        return x
    if len(valid) <= 1:
        return x
    if valid.max() == valid.min():
        return np.full_like(x, 50.0)
    return (x - valid.min()) / (valid.max() - valid.min()) * 100


def main():
    rf_model = load_artifact("rf_model")
    gb_model = load_artifact("gb_model")
    dt_model = load_artifact("dt_model")
    X_train = load_artifact("X_train")

    feature_names = list(X_train.columns)

    rf_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    plot_single_importance(rf_importance_df, "Random Forest - Feature Importance",
                            "fig_rf_feature_importance.png")

    gb_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": gb_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    plot_single_importance(gb_importance_df, "Gradient Boosting - Feature Importance",
                            "fig_gb_feature_importance.png", ylabel="Relative Influence")

    dt_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": dt_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    plot_single_importance(dt_importance_df, "Decision Tree - Feature Importance",
                            "fig_dt_feature_importance.png")

    # Combine and normalize across models
    all_features = sorted(set(rf_importance_df["Feature"]) |
                           set(gb_importance_df["Feature"]) |
                           set(dt_importance_df["Feature"]))
    feature_importance = pd.DataFrame({"Feature": all_features})

    def lookup(df, feature):
        row = df[df["Feature"] == feature]
        return row["Importance"].iloc[0] if len(row) else np.nan

    feature_importance["RF"] = feature_importance["Feature"].apply(lambda f: lookup(rf_importance_df, f))
    feature_importance["GB"] = feature_importance["Feature"].apply(lambda f: lookup(gb_importance_df, f))
    feature_importance["DT"] = feature_importance["Feature"].apply(lambda f: lookup(dt_importance_df, f))

    feature_importance["RF_norm"] = normalize(feature_importance["RF"])
    feature_importance["GB_norm"] = normalize(feature_importance["GB"])
    feature_importance["DT_norm"] = normalize(feature_importance["DT"])

    norm_cols = ["RF_norm", "GB_norm", "DT_norm"]
    feature_importance["Avg_Importance"] = feature_importance[norm_cols].mean(axis=1, skipna=True)
    feature_importance = feature_importance.sort_values("Avg_Importance", ascending=False)

    top_features = feature_importance.head(10)

    plot_data = top_features.melt(id_vars="Feature", value_vars=norm_cols,
                                   var_name="Model", value_name="Importance")
    plot_data["Model"] = plot_data["Model"].str.replace("_norm", "", regex=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    models = plot_data["Model"].unique()
    feature_order = top_features["Feature"].tolist()
    y_pos = np.arange(len(feature_order))
    bar_height = 0.8 / len(models)

    for i, model in enumerate(models):
        subset = plot_data[plot_data["Model"] == model].set_index("Feature").reindex(feature_order)
        ax.barh(y_pos + i * bar_height, subset["Importance"], height=bar_height, label=model)

    ax.set_yticks(y_pos + bar_height * (len(models) - 1) / 2)
    ax.set_yticklabels(feature_order)
    ax.set_xlabel("Normalized Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance Across Models")
    ax.legend(title="Model", loc="lower right")
    fig.tight_layout()
    out_path = os.path.join(FIGURE_DIR, "fig_feature_importance_across_models.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

    return feature_importance


if __name__ == "__main__":
    main()