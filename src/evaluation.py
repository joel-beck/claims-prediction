from config import ModelPaths
from modeling import (
    extract_column_transformer,
    extract_hyperparams,
    extract_model,
    load_model,
    remove_all_pipeline_prefixes,
)

baseline_fit = load_model(ModelPaths.baseline)
baseline_model = extract_model(baseline_fit)

column_transformer = extract_column_transformer(baseline_fit)
# feature names the same for all models
feature_names = column_transformer.get_feature_names_out()

linear_regression_fit = load_model(ModelPaths.linear_regression)
linear_regression_model = extract_model(linear_regression_fit)
linear_regression_model.intercept_
linear_regression_model.coef_

dict(
    zip(
        remove_all_pipeline_prefixes(feature_names),
        linear_regression_model.coef_.round(3),
    )
)

ridge_regression_fit = load_model(ModelPaths.ridge_regression)
ridge_regression_hyperparams = extract_hyperparams(ridge_regression_fit)
ridge_regression_model = extract_model(ridge_regression_fit)

dict(
    zip(
        remove_all_pipeline_prefixes(feature_names),
        ridge_regression_model.coef_.round(3),
    )
)

random_forest_fit = load_model(ModelPaths.random_forest)
random_forest_hyperparams = extract_hyperparams(random_forest_fit)
random_forest_model = extract_model(random_forest_fit)

dict(
    zip(
        remove_all_pipeline_prefixes(feature_names),
        random_forest_model.feature_importances_.round(3),
    )
)

xgboost_fit = load_model(ModelPaths.xgboost)
xgboost_hyperparams = extract_hyperparams(xgboost_fit)
xgboost_model = extract_model(xgboost_fit)

dict(
    zip(
        remove_all_pipeline_prefixes(feature_names),
        xgboost_model.feature_importances_.round(3),
    )
)
