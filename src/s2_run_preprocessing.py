import polars as pl
from config import DataPaths
from preprocessing import (
    add_binned_columns,
    add_boolean_vehicle_gas,
    add_claim_amount_per_year,
    add_log_transformations,
    join_target_data,
)


def main() -> None:
    df_predictors = pl.scan_parquet(DataPaths.raw.predictors_parquet)
    df_target = pl.scan_parquet(DataPaths.raw.target_parquet)

    output_columns = [
        "contract_id",
        "claim_amount_per_year",
        "log_claim_amount_per_year",
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

    df_complete = (
        df_predictors.pipe(join_target_data, df_target)
        .pipe(add_claim_amount_per_year)
        .pipe(add_log_transformations)
        .pipe(add_binned_columns)
        .pipe(add_boolean_vehicle_gas)
        .select(output_columns)
        .collect()
    )

    df_complete.write_parquet(DataPaths.processed.complete)


if __name__ == "__main__":
    main()
