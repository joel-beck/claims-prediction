from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class ResultsPaths:
    predictions: Path = PROJECT_PATH / "data" / "predictions.parquet"


@dataclass(frozen=True)
class DataPaths:
    raw: RawDataPaths = RawDataPaths()  # noqa: RUF009
    processed: ProcessedDataPaths = ProcessedDataPaths()  # noqa: RUF009
    results: ResultsPaths = ResultsPaths()  # noqa: RUF009


@dataclass(frozen=True)
class PipelineKeys:
    column_transformer: str = "column_transformer"
    model: str = "model"


@dataclass
class ModelConfig:
    target: str = "log_claim_amount_per_year"
    test_size: float = 0.2
    seed: int = 42


@dataclass
class CvConfig:
    n_folds: int = 1
    n_iter: int = 2
    scoring: tuple[str, ...] = ("neg_mean_absolute_error", "neg_mean_squared_error", "r2")
    refit: str = "neg_mean_squared_error"
    random_state: int = 123
    verbose: int = 2
