from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self, TypeAlias, overload

import numpy as np
import pandas as pd
from config import CvConfig, ModelConfig, PipelineKeys
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ParamGrid: TypeAlias = dict[str, Sequence]


class RegressionModel(Protocol):
    def fit(self, X_train: Any, y_train: Any) -> Self:
        ...

    def predict(self, X_test: Any) -> Any:
        ...


class NotFittedError(Exception):
    def __str__(self) -> str:
        return "The model has not been fitted yet. Call `.fit()` before `.predict()`."


@dataclass
class BaselineModel:
    mean_prediction: float | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Self:  # noqa: ARG002
        self.mean_prediction = y_train.mean()

        return self

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.mean_prediction is None:
            raise NotFittedError

        return np.repeat(self.mean_prediction, X_test.shape[0])


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class Predictions:
    model_name: str
    predictions: pd.Series


def split_data(df: pd.DataFrame) -> DataSplit:
    df_train, df_test = train_test_split(
        df,
        test_size=ModelConfig.test_size,
        random_state=ModelConfig.seed,
    )

    X_train = df_train.drop(ModelConfig.target, axis=1)
    y_train = df_train[ModelConfig.target]

    X_test = df_test.drop(ModelConfig.target, axis=1)
    y_test = df_test[ModelConfig.target]

    return DataSplit(X_train, X_test, y_train, y_test)


def get_true_values(data_split: DataSplit) -> pd.Series:
    return data_split.X_test["claim_amount_per_year"]


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


def setup_pipeline(column_transformer: ColumnTransformer, model: RegressionModel) -> Pipeline:
    return Pipeline(
        [
            (PipelineKeys.column_transformer, column_transformer),
            (PipelineKeys.model, model),
        ]
    )


def fit_without_cv(data_split: DataSplit, pipeline: Pipeline) -> Pipeline:
    return pipeline.fit(data_split.X_train, data_split.y_train)


def setup_cv(pipeline: Pipeline, param_grid: ParamGrid) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        cv=CvConfig.n_folds,
        n_iter=CvConfig.n_iter,
        scoring=CvConfig.scoring,
        refit=CvConfig.refit,
        return_train_score=True,
        random_state=CvConfig.random_state,
        verbose=CvConfig.verbose,
    )


def fit_with_cv(
    data_split: DataSplit,
    pipeline: Pipeline,
    param_grid: ParamGrid,
) -> RandomizedSearchCV:
    cv = setup_cv(pipeline, param_grid)
    return cv.fit(data_split.X_train, data_split.y_train)


@overload
def fit(
    data_split: DataSplit,
    column_transformer: ColumnTransformer,
    model: RegressionModel,
) -> Pipeline:
    ...


@overload
def fit(
    data_split: DataSplit,
    column_transformer: ColumnTransformer,
    model: RegressionModel,
    param_grid: ParamGrid,
) -> RandomizedSearchCV:
    ...


def fit(
    data_split: DataSplit,
    column_transformer: ColumnTransformer,
    model: RegressionModel,
    param_grid: ParamGrid | None = None,
) -> Pipeline | RandomizedSearchCV:
    pipeline = setup_pipeline(column_transformer, model)

    if param_grid is None:
        return fit_without_cv(data_split, pipeline)

    return fit_with_cv(data_split, pipeline, param_grid)


def predict(data_split: DataSplit, model: Pipeline | RandomizedSearchCV) -> pd.Series:
    return pd.Series(model.predict(data_split.X_test))


def collect_predictions(true_values: pd.Series, *predictions: Predictions) -> pd.DataFrame:
    predictions_dict = {prediction.model_name: prediction.predictions for prediction in predictions}
    predictions_dict["true_values"] = true_values

    return pd.DataFrame(predictions_dict)
