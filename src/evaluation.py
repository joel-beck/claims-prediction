import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import ModelPaths
from modeling import (
    extract_column_transformer,
    extract_model,
    load_model,
    remove_all_pipeline_prefixes,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

sns.set_theme(style="whitegrid")


def plot_random_forest_predictions(predictions_frame: pd.DataFrame) -> None:
    sns.scatterplot(predictions_frame, x="true_values", y="random_forest").set(
        xlabel="True Values",
        ylabel="Predictions",
        title="Random Forest Predictions vs. True Values",
    )


def plot_claims_vs_age(df: pd.DataFrame) -> None:
    sns.scatterplot(
        # remove outliers for plotting
        data=df.loc[df["claim_amount"] < 1e6],
        x="driver_age",
        y="claim_amount",
    )


def plot_performance_metrics(predictions_frame: pd.DataFrame) -> None:
    metrics = (
        predictions_frame.melt(id_vars="true_values", var_name="model", value_name="predictions")
        .groupby("model", as_index=False)
        .agg(
            mse=(
                "predictions",
                lambda predictions: mean_squared_error(
                    predictions, predictions_frame["true_values"]
                ),
            ),
            mae=(
                "predictions",
                lambda predictions: mean_absolute_error(
                    predictions, predictions_frame["true_values"]
                ),
            ),
        )
        .melt(id_vars="model", var_name="metric", value_name="value")
    )

    sns.catplot(metrics, x="value", y="model", col="metric", kind="bar", sharex=False).set(
        xlabel=None, ylabel=None
    )

    plt.show()


def collect_coefficients_data() -> pd.DataFrame:
    linear_regression_fit = load_model(ModelPaths.linear_regression)
    linear_regression_model = extract_model(linear_regression_fit)

    column_transformer = extract_column_transformer(linear_regression_fit)
    feature_names = column_transformer.get_feature_names_out()

    lasso_regression_fit = load_model(ModelPaths.lasso_regression)
    lasso_regression_model = extract_model(lasso_regression_fit)

    random_forest_fit = load_model(ModelPaths.random_forest)
    random_forest_model = extract_model(random_forest_fit)

    xgboost_fit = load_model(ModelPaths.xgboost)
    xgboost_model = extract_model(xgboost_fit)

    return pd.DataFrame(
        {
            "feature": remove_all_pipeline_prefixes(feature_names),
            "linear_regression": linear_regression_model.coef_.round(3),
            "lasso_regression": lasso_regression_model.coef_.round(3),
            "random_forest": random_forest_model.feature_importances_.round(3),
            "xgboost": xgboost_model.feature_importances_.round(3),
        }
    )


def plot_coefficients(coefficients_data: pd.DataFrame) -> None:
    coefficients = coefficients_data[["feature", "linear_regression", "lasso_regression"]].melt(
        id_vars="feature", var_name="model", value_name="coefficient"
    )

    sns.catplot(
        coefficients,
        x="coefficient",
        y="feature",
        col="model",
        kind="bar",
        sharex=False,
        height=8,
        aspect=0.5,
    ).set(ylabel=None, xlabel="Coefficient").set_titles("{col_name}")

    plt.show()


def plot_feature_importances(coefficients_data: pd.DataFrame) -> None:
    feature_importances = coefficients_data[["feature", "random_forest", "xgboost"]].melt(
        id_vars="feature", var_name="model", value_name="feature_importance"
    )

    sns.catplot(
        feature_importances,
        x="feature_importance",
        y="feature",
        col="model",
        kind="bar",
        sharex=False,
        height=8,
        aspect=0.5,
    ).set(ylabel=None, xlabel="Feature Importance").set_titles("{col_name}")

    plt.show()
