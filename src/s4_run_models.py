import pandas as pd
from config import DataPaths, ModelPaths, ParamGrids
from modeling import (
    BaselineModel,
    Predictions,
    collect_predictions,
    fit,
    get_column_transformer,
    predict,
    save_model,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor


def main() -> None:
    X_train: pd.DataFrame = pd.read_parquet(DataPaths.processed.X_train)
    y_train: pd.Series = pd.read_parquet(DataPaths.processed.y_train).squeeze()
    X_test: pd.DataFrame = pd.read_parquet(DataPaths.processed.X_test)
    true_test_values: pd.Series = pd.read_parquet(DataPaths.results.true_values).squeeze()

    column_transformer = get_column_transformer()
    param_grids = ParamGrids()

    baseline_model = BaselineModel()
    baseline_fit = fit(X_train, y_train, column_transformer, baseline_model)
    baseline_predictions = predict(X_test, baseline_fit)
    save_model(baseline_fit, ModelPaths.baseline)

    linear_regression_model = LinearRegression()
    linear_regression_fit = fit(X_train, y_train, column_transformer, linear_regression_model)
    linear_regression_predictions = predict(X_test, linear_regression_fit)
    save_model(linear_regression_fit, ModelPaths.linear_regression)

    ridge_regression_model = Ridge()
    ridge_regression_fit = fit(
        X_train, y_train, column_transformer, ridge_regression_model, param_grids.ridge_regression
    )
    ridge_regression_predictions = predict(X_test, ridge_regression_fit)
    save_model(ridge_regression_fit, ModelPaths.ridge_regression)

    random_forest_model = RandomForestRegressor()
    random_forest_fit = fit(
        X_train, y_train, column_transformer, random_forest_model, param_grids.random_forest
    )
    random_forest_predictions = predict(X_test, random_forest_fit)
    save_model(random_forest_fit, ModelPaths.random_forest)

    xgboost_model = XGBRegressor()
    xgboost_fit = fit(X_train, y_train, column_transformer, xgboost_model, param_grids.xgboost)
    xgb_predictions = predict(X_test, xgboost_fit)
    save_model(xgboost_fit, ModelPaths.xgboost)

    predictions_frame = collect_predictions(
        true_test_values,
        Predictions("baseline", baseline_predictions),
        Predictions("linear_regression", linear_regression_predictions),
        Predictions("ridge_regression", ridge_regression_predictions),
        Predictions("random_forest", random_forest_predictions),
        Predictions("xgboost", xgb_predictions),
    )
    predictions_frame.to_parquet(DataPaths.results.predictions)


if __name__ == "__main__":
    main()
