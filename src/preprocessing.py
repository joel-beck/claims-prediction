import polars as pl
from config import DataPaths


def join_target_data(df_predictors: pl.LazyFrame, df_target: pl.LazyFrame) -> pl.LazyFrame:
    return df_predictors.join(df_target, on="contract_id", how="inner")


def add_claim_amount_per_year(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        claim_amount_per_year=pl.col("claim_amount") / pl.col("contract_duration")
    )


def add_log_transformations(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        log_claim_amount_per_year=pl.col("claim_amount_per_year").log(),
        log_bonus_malus=pl.col("bonus_malus").log(),
        log_population_density=pl.col("population_density").log(),
    )


def bin_column(expression: pl.Expr, bins: list[float]) -> pl.Expr:
    return expression.cut(breaks=bins)


def add_binned_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Cuts integer columns with many unique values or a skewed distribution into bins.

    The number of claims is binned into the categories "1", "2", and "3 or more". The
    driver age is binned into age groups of 5 years.
    """
    return df.with_columns(
        number_claims=bin_column(pl.col("number_claims"), bins=[0, 1, 2]),
        driver_age_groups=bin_column(pl.col("driver_age"), bins=list(range(15, 105, 5))),
    )


def binary_to_boolean(expression: pl.Expr) -> pl.Expr:
    return expression.cast(pl.Boolean)


def add_boolean_columns(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.with_columns(
        is_diesel=binary_to_boolean(pl.col("vehicle_gas") == "Diesel"),
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
        .pipe(add_boolean_columns)
        .select(output_columns)
        .collect()
    )

    df_complete.write_parquet(DataPaths.processed.complete)


if __name__ == "__main__":
    main()
