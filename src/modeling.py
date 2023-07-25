import numpy as np
import pandas as pd
import polars as pl
from config import DataPaths, ModelConfig
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_column_transformer() -> ColumnTransformer:
    """
    Standardizes all numeric Variables and transforms all categorical Variables to Dummy
    Variables.
    """
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer(
        [
            ("scaler", scaler, make_column_selector(dtype_include="number")),  # type: ignore
            ("encoder", encoder, make_column_selector(dtype_include="object")),  # type: ignore
        ],
    )


def fit_baseline_model(y_train: pd.Series, y_test: pd.Series) -> pd.Series:
    """
    Fits a baseline model that always predicts the mean of the training target.
    """
    return pd.Series(
        np.repeat(y_train.mean(), len(y_test)),
    )


def main() -> None:
    df_complete = pl.read_parquet(DataPaths.processed.complete).to_pandas()

    df_train, df_test = train_test_split(
        df_complete,
        test_size=0.2,
        random_state=ModelConfig.seed,
    )

    X_train = df_train.drop(ModelConfig.target, axis=1)
    y_train = df_train[ModelConfig.target]

    X_test = df_test.drop(ModelConfig.target, axis=1)
    y_test = df_test[ModelConfig.target]

    true_values = X_test["claim_amount_per_year"]

    baseline_log_predictions = fit_baseline_model(y_train, y_test)
    baseline_predictions = np.exp(baseline_log_predictions)

    mean_squared_error(true_values, baseline_predictions)
    mean_absolute_error(true_values, baseline_predictions)

    column_transformer = get_column_transformer()
    linear_regression = LinearRegression()

    pipeline = Pipeline(
        [
            ("column_transformer", column_transformer),
            ("linear_regression", linear_regression),
        ]
    )

    pipeline.fit(X_train, y_train)

    linear_regression_log_predictions = pipeline.predict(X_test)
    linear_regression_predictions = np.exp(linear_regression_log_predictions)

    mean_squared_error(true_values, linear_regression_predictions)
    mean_absolute_error(true_values, linear_regression_predictions)
    r2_score(true_values, linear_regression_predictions)


if __name__ == "__main__":
    main()
