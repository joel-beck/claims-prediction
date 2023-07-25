import polars as pl
from config import DataPaths


def join_target_data(df_predictors: pl.DataFrame, df_target: pl.DataFrame) -> pl.DataFrame:
    return df_predictors.join(df_target, on="contract_id", how="inner")


def add_claim_amount_per_year(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        claim_amount_per_year=pl.col("claim_amount") / pl.col("contract_duration")
    )


def add_log_transformations(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        log_claim_amount_per_year=pl.col("claim_amount_per_year").log(),
        log_bonus_malus=pl.col("bonus_malus").log(),
        log_population_density=pl.col("population_density").log(),
    )


def main() -> None:
    df_predictors = pl.read_parquet(DataPaths.raw.predictors_parquet)
    df_target = pl.read_parquet(DataPaths.raw.target_parquet)

    output_columns = [
        "contract_id",
        "claim_amount_per_year",
        "log_claim_amount_per_year",
        "number_claims",
        "driver_age",
        "log_bonus_malus",
        "vehicle_age",
        "vehicle_brand",
        "vehicle_power",
        "vehicle_gas",
        "area_code",
        "region",
        "log_population_density",
    ]

    df_complete = (
        df_predictors.pipe(join_target_data, df_target)
        .pipe(add_claim_amount_per_year)
        .pipe(add_log_transformations)
        .select(output_columns)
    )

    df_complete.write_parquet(DataPaths.processed.complete)


if __name__ == "__main__":
    main()
