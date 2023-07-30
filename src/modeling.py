from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self, overload

import joblib
import numpy as np
import pandas as pd
from config import CvConfig, ParamGrid, PipelineKeys
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class NotFittedError(Exception):
    def __str__(self) -> str:
        return "The model has not been fitted yet. Call `.fit()` before `.predict()`."


class RegressionModel(Protocol):
    def fit(self, X_train: Any, y_train: Any) -> Self:
        ...

    def predict(self, X_test: Any) -> Any:
        ...


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
class Predictions:
    model_name: str
    predictions: np.ndarray


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


def fit_without_cv(X_train: pd.DataFrame, y_train: pd.Series, pipeline: Pipeline) -> Pipeline:
    return pipeline.fit(X_train, y_train)


def prefix_param_grid_keys(param_grid: ParamGrid) -> ParamGrid:
    """
    Prefixes all keys of a param_grid with the name of the pipeline step they belong to
    (e.g. "model__") Allows to set up the param grids with the hyperparameter names only
    without knowledge about the specific pipeline steps.
    """
    return {f"{PipelineKeys.model}__{key}": value for key, value in param_grid.items()}


def setup_cv(pipeline: Pipeline, param_grid: ParamGrid) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=prefix_param_grid_keys(param_grid),
        cv=CvConfig.n_folds,
        n_iter=CvConfig.n_iter,
        scoring=CvConfig.scoring,
        refit=CvConfig.refit,
        return_train_score=True,
        random_state=CvConfig.random_state,
        verbose=CvConfig.verbose_level,
    )


def fit_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline,
    param_grid: ParamGrid,
) -> RandomizedSearchCV:
    cv = setup_cv(pipeline, param_grid)
    return cv.fit(X_train, y_train)


@overload
def fit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    column_transformer: ColumnTransformer,
    model: RegressionModel,
) -> Pipeline:
    ...


@overload
def fit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    column_transformer: ColumnTransformer,
    model: RegressionModel,
    param_grid: ParamGrid,
) -> RandomizedSearchCV:
    ...


def fit(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    column_transformer: ColumnTransformer,
    model: RegressionModel,
    param_grid: ParamGrid | None = None,
) -> Pipeline | RandomizedSearchCV:
    pipeline = setup_pipeline(column_transformer, model)

    if param_grid is None:
        return fit_without_cv(X_train, y_train, pipeline)

    return fit_with_cv(X_train, y_train, pipeline, param_grid)


def predict(X_test: pd.DataFrame, model: Pipeline | RandomizedSearchCV) -> pd.Series:
    return pd.Series(model.predict(X_test))


def save_model(model: Pipeline | RandomizedSearchCV, path: Path) -> None:
    return joblib.dump(model, path)


def load_model(path: Path) -> Pipeline | RandomizedSearchCV:
    return joblib.load(path)


def extract_pipeline_from_cv(cv: RandomizedSearchCV) -> Pipeline:
    return cv.best_estimator_


def remove_all_pipeline_prefixes(keys: Sequence[str] | np.ndarray) -> list[str]:
    """
    Removes all prefixes added from scikit-learn Pipelines to display only the original
    key value. Pipeline prefixes are prepended by double underscores (e.g. "model__").
    """
    return [key.split("__")[-1] for key in keys]


def extract_hyperparams(cv: RandomizedSearchCV) -> dict[str, int | float]:
    """
    Extracts hyperparameter keys and values with the original key names.
    """
    return dict(
        zip(
            remove_all_pipeline_prefixes(list(cv.best_params_.keys())),
            cv.best_params_.values(),
        )
    )


def extract_column_transformer(pipeline: Pipeline | RandomizedSearchCV) -> ColumnTransformer:
    if isinstance(pipeline, RandomizedSearchCV):
        pipeline = extract_pipeline_from_cv(pipeline)

    return pipeline.named_steps.column_transformer


def extract_model(pipeline: Pipeline | RandomizedSearchCV) -> RegressionModel:
    if isinstance(pipeline, RandomizedSearchCV):
        pipeline = extract_pipeline_from_cv(pipeline)

    return pipeline.named_steps.model


def collect_predictions(true_values: pd.Series, *predictions: Predictions) -> pd.DataFrame:
    predictions_dict = {prediction.model_name: prediction.predictions for prediction in predictions}
    # reset index to align with predictions correctly
    predictions_dict["true_values"] = true_values.to_numpy()

    return pd.DataFrame(predictions_dict)
