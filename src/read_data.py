from pathlib import Path
from typing import TypeAlias

import arff
import polars as pl
from config import DataPaths

SchemaMapping: TypeAlias = dict[str, pl.DataType]


def read_arff_format(path: Path, schema: SchemaMapping) -> pl.DataFrame:
    with path.open() as f:
        data = arff.load(f)

    return pl.DataFrame(data["data"], schema=schema)


def sum_claims_per_contract(target_df: pl.DataFrame) -> pl.DataFrame:
    return target_df.groupby("contract_id").sum()


def main() -> None:
    target_schema: SchemaMapping = {
        "contract_id": pl.Int64(),
        "claim_amount": pl.Float64(),
    }

    predictors_schema: SchemaMapping = {
        "contract_id": pl.Int64(),
        "number_claims": pl.Int64(),
        "contract_duration": pl.Float64(),
        "area_code": pl.Utf8(),
        "vehicle_power": pl.Int64(),
        "vehicle_age": pl.Int64(),
        "driver_age": pl.Int64(),
        "bonus_malus": pl.Int64(),
        "vehicle_brand": pl.Utf8(),
        "vehicle_gas": pl.Utf8(),
        "population_density": pl.Int64(),
        "region": pl.Utf8(),
    }

    df_predictors = read_arff_format(DataPaths.raw.predictors_arff, predictors_schema)
    df_target = read_arff_format(DataPaths.raw.target_arff, target_schema).pipe(
        sum_claims_per_contract
    )

    df_predictors.write_parquet(DataPaths.raw.predictors_parquet)
    df_target.write_parquet(DataPaths.raw.target_parquet)


if __name__ == "__main__":
    main()
