from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np

ParamGrid: TypeAlias = dict[str, Sequence | np.ndarray]

PROJECT_PATH = Path(__file__).parent.parent


@dataclass(frozen=True)
class RawDataPaths:
    predictors_arff: Path = PROJECT_PATH / "data" / "freMTPL2freq.arff"
    predictors_parquet: Path = PROJECT_PATH / "data" / "predictor_data.parquet"
    target_arff: Path = PROJECT_PATH / "data" / "freMTPL2sev.arff"
    target_parquet: Path = PROJECT_PATH / "data" / "target_data.parquet"


@dataclass(frozen=True)
class ProcessedDataPaths:
    complete: Path = PROJECT_PATH / "data" / "complete_data.parquet"
    X_train: Path = PROJECT_PATH / "data" / "X_train.parquet"
    y_train: Path = PROJECT_PATH / "data" / "y_train.parquet"
    X_test: Path = PROJECT_PATH / "data" / "X_test.parquet"
    y_test: Path = PROJECT_PATH / "data" / "y_test.parquet"


@dataclass(frozen=True)
class ResultsPaths:
    predictions: Path = PROJECT_PATH / "data" / "predictions.parquet"
    true_values: Path = PROJECT_PATH / "data" / "true_values.parquet"


@dataclass(frozen=True)
class DataPaths:
    raw: RawDataPaths = RawDataPaths()  # noqa: RUF009
    processed: ProcessedDataPaths = ProcessedDataPaths()  # noqa: RUF009
    results: ResultsPaths = ResultsPaths()  # noqa: RUF009


@dataclass
class ModelPaths:
    baseline: Path = PROJECT_PATH / "models" / "baseline.joblib"
    linear_regression: Path = PROJECT_PATH / "models" / "linear_regression.joblib"
    ridge_regression: Path = PROJECT_PATH / "models" / "ridge_regression.joblib"
    random_forest: Path = PROJECT_PATH / "models" / "random_forest.joblib"
    xgboost: Path = PROJECT_PATH / "models" / "xgboost.joblib"


@dataclass
class ModelConfig:
    target: str = "claim_amount_per_year"
    log_target: str = "log_claim_amount_per_year"
    predictors: list[str] = field(
        default_factory=lambda: [
            "number_claims",
            "driver_age_groups",
            "log_bonus_malus",
            "vehicle_age",
            "vehicle_brand",
            "vehicle_power",
            "is_diesel",
            "area_code",
            "region",
            "log_population_density",
        ]
    )
    test_size: float = 0.2
    split_seed: int = 42


@dataclass(frozen=True)
class PipelineKeys:
    column_transformer: str = "column_transformer"
    model: str = "model"


@dataclass
class CvConfig:
    n_folds: int = 5
    n_iter: int = 1
    scoring: tuple[str, ...] = ("neg_mean_absolute_error", "neg_mean_squared_error", "r2")
    refit: str = "neg_mean_squared_error"
    random_state: int = 123
    verbose: int = 2


@dataclass
class ParamGrids:
    ridge_regression: ParamGrid = field(default_factory=lambda: {"alpha": np.logspace(-3, 3, 7)})
    random_forest: ParamGrid = field(
        default_factory=lambda: {
            "n_estimators": np.arange(100, 1000, 100),
            "max_depth": [None, *list(np.arange(1, 10))],
            "min_samples_leaf": np.arange(1, 10),
        }
    )
    xgboost: ParamGrid = field(
        # could add alpha, lambda and gamma for tuning
        default_factory=lambda: {
            "n_estimators": np.arange(100, 1000, 100),
            "max_depth": np.arange(1, 10),
            "learning_rate": np.logspace(-3, 0, 4),
            "subsample": np.arange(0.5, 1.0, 0.1),
            "colsample_bytree": np.arange(0.5, 1.0, 0.1),
        }
    )
