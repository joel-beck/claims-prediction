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
class DataPaths:
    raw: RawDataPaths = RawDataPaths()  # noqa: RUF009
    processed: ProcessedDataPaths = ProcessedDataPaths()  # noqa: RUF009
