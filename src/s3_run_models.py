import polars as pl
from config import DataPaths
from modeling import (
    BaselineModel,
    Predictions,
    collect_predictions,
    fit,
    get_column_transformer,
    get_true_values,
    predict,
    split_data,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor


def main() -> None:
    df_complete = pl.read_parquet(DataPaths.processed.complete).to_pandas()
    data_split = split_data(df_complete)
    true_values = get_true_values(data_split)

    column_transformer = get_column_transformer()

    baseline_model = BaselineModel()
    baseline_fit = fit(data_split, column_transformer, baseline_model)
    baseline_predictions = predict(data_split, baseline_fit)

    linear_regression_model = LinearRegression()
    linear_regression_fit = fit(data_split, column_transformer, linear_regression_model)
    linear_regression_predictions = predict(data_split, linear_regression_fit)

    ridge_regression_model = Ridge()
    ridge_regression_fit = fit(data_split, column_transformer, ridge_regression_model)
    ridge_regression_predictions = predict(data_split, ridge_regression_fit)

    random_forest_model = RandomForestRegressor()
    random_forest_fit = fit(data_split, column_transformer, random_forest_model)
    random_forest_predictions = predict(data_split, random_forest_fit)

    xgboost_model = XGBRegressor()
    xgboost_fit = fit(data_split, column_transformer, xgboost_model)
    xgb_predictions = predict(data_split, xgboost_fit)

    predictions_frame = collect_predictions(
        true_values,
        Predictions("baseline", baseline_predictions),
        Predictions("linear_regression", linear_regression_predictions),
        Predictions("ridge_regression", ridge_regression_predictions),
        Predictions("random_forest", random_forest_predictions),
        Predictions("xgboost", xgb_predictions),
    )

    predictions_frame.to_parquet(DataPaths.results.predictions)


if __name__ == "__main__":
    main()
