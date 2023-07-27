import polars as pl
from config import DataPaths, ModelConfig
from sklearn.model_selection import train_test_split


def main() -> None:
    df_model = pl.read_parquet(DataPaths.processed.model).to_pandas()
    model_config = ModelConfig()

    df_train, df_test = train_test_split(
        df_model,
        test_size=model_config.test_size,
        random_state=model_config.split_seed,
    )

    X_train = df_train[model_config.predictors]
    X_train.to_parquet(DataPaths.processed.X_train)

    # can only save dataframes to parquet, not series
    y_train = df_train[[model_config.log_target]]
    y_train.to_parquet(DataPaths.processed.y_train)

    X_test = df_test[model_config.predictors]
    X_test.to_parquet(DataPaths.processed.X_test)

    y_test = df_test[[model_config.log_target]]
    y_test.to_parquet(DataPaths.processed.y_test)

    true_test_values = df_test[[model_config.target]]
    true_test_values.to_parquet(DataPaths.results.true_values)


if __name__ == "__main__":
    main()
