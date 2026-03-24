import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from risk_stratifier.models import (
    get_logistic_lasso_pipeline_and_hyperparameters,
    get_logistic_ridge_pipeline_and_hyperparameters,
    get_xgboost_pipeline_and_hyperparameters
)

# constants and helpers

N_ROWS = 1000
RANDOM_SEED = 42
CATEGORICAL_FEATURES = ["gender", "region", "diagnosis_code"]
NUMERIC_FEATURES = ["age", "bmi", "blood_pressure", "cholesterol"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _get_sub_pipelines(pipeline):
    """Return (preprocessor, cat_pipeline, num_pipeline) from a pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    lookup = {t[0]: t[1] for t in preprocessor.transformers}
    return preprocessor, lookup["cat"], lookup["num"]


def _fit_predict(pipeline, df, target):
    """Fit pipeline and return (class predictions, probability matrix)."""
    pipeline.fit(df[ALL_FEATURES], target)
    preds  = pipeline.predict(df[ALL_FEATURES])
    probas = pipeline.predict_proba(df[ALL_FEATURES])
    return preds, probas


# Fixtures

@pytest.fixture(scope="session")
def sample_data():
    """
    ~1 000-row dataset with a binary target and a mixture of integer,
    float, and string columns. ~5 % NAs injected into numeric columns
    and ~3 % into a categorical column.
    Target is mildly imbalanced (~25 % positive), mirroring healthcare data.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = N_ROWS

    df = pd.DataFrame({
        "age":            rng.integers(18, 90, size=n).astype(float),
        "cholesterol":    rng.integers(150, 300, size=n).astype(float),
        "bmi":            rng.uniform(17.0, 45.0, size=n).round(1),
        "blood_pressure": rng.uniform(60.0, 180.0, size=n).round(1),
        "gender":         rng.choice(["Male", "Female", "Other"], size=n),
        "region":         rng.choice(["North", "South", "East", "West"], size=n),
        "diagnosis_code": rng.choice(["C34", "C50", "C18", "C71", "C20"], size=n),
    })

    for col in ["bmi", "blood_pressure", "age", "cholesterol"]:
        df.loc[rng.choice(n, size=int(0.05 * n), replace=False), col] = np.nan

    df.loc[rng.choice(n, size=int(0.03 * n), replace=False), "region"] = np.nan

    target = pd.Series(rng.choice([0, 1], size=n, p=[0.75, 0.25]), name="target")
    return df, target


@pytest.fixture(scope="session")
def ridge_pipeline_and_params():
    return get_logistic_ridge_pipeline_and_hyperparameters(
        CATEGORICAL_FEATURES, NUMERIC_FEATURES
    )


@pytest.fixture(scope="session")
def lasso_pipeline_and_params():
    return get_logistic_lasso_pipeline_and_hyperparameters(
        CATEGORICAL_FEATURES, NUMERIC_FEATURES
    )


@pytest.fixture(scope="session")
def xgb_pipeline_and_params():
    return get_xgboost_pipeline_and_hyperparameters(
        CATEGORICAL_FEATURES, NUMERIC_FEATURES
    )

# Test sample data

class TestSampleData:

    def test_row_count(self, sample_data):
        df, target = sample_data
        assert len(df) == N_ROWS
        assert len(target) == N_ROWS

    def test_target_is_binary(self, sample_data):
        _, target = sample_data
        assert set(target.unique()).issubset({0, 1})

    def test_target_has_both_classes(self, sample_data):
        _, target = sample_data
        assert target.sum() > 0,        "No positive examples in target"
        assert (target == 0).sum() > 0, "No negative examples in target"

    def test_float_columns_dtype(self, sample_data):
        df, _ = sample_data
        assert df["bmi"].dtype == np.float64
        assert df["blood_pressure"].dtype == np.float64

    def test_string_columns_dtype(self, sample_data):
        df, _ = sample_data
        for col in CATEGORICAL_FEATURES:
            assert df[col].dtype == object, f"Column {col!r} is not object dtype"

    def test_numeric_missings_present(self, sample_data):
        df, _ = sample_data
        assert df[NUMERIC_FEATURES].isna().any().any()

    def test_categorical_missing_present(self, sample_data):
        df, _ = sample_data
        assert df["region"].isna().any()

# Test return types

class TestReturnTypes:

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_returns_pipeline_and_dict(self, request, fixture_name):
        pipeline, param_grid = request.getfixturevalue(fixture_name)
        assert isinstance(pipeline, Pipeline)
        assert isinstance(param_grid, dict)

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_param_grid_values_are_non_empty_lists(self, request, fixture_name):
        _, param_grid = request.getfixturevalue(fixture_name)
        for key, values in param_grid.items():
            assert isinstance(values, list), f"{key!r}: expected list, got {type(values)}"
            assert len(values) > 0,          f"{key!r}: param list is empty"

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_param_grid_keys_follow_pipeline_convention(self, request, fixture_name):
        _, param_grid = request.getfixturevalue(fixture_name)
        for key in param_grid:
            assert key.startswith("classifier__"), (
                f"Param grid key {key!r} does not start with 'classifier__'"
            )

# test pipeline structure

class TestPipelineStructure:

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_top_level_step_names(self, request, fixture_name):
        pipeline, _ = request.getfixturevalue(fixture_name)
        assert list(pipeline.named_steps.keys()) == ["preprocessor", "classifier"]

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_preprocessor_is_column_transformer(self, request, fixture_name):
        pipeline, _ = request.getfixturevalue(fixture_name)
        assert isinstance(pipeline.named_steps["preprocessor"], ColumnTransformer)

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_column_transformer_has_cat_and_num_branches(self, request, fixture_name):
        pipeline, _ = request.getfixturevalue(fixture_name)
        names = {t[0] for t in pipeline.named_steps["preprocessor"].transformers}
        assert "cat" in names and "num" in names

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_numeric_branch_has_median_imputer(self, request, fixture_name):
        pipeline, _ = request.getfixturevalue(fixture_name)
        _, _, num_p = _get_sub_pipelines(pipeline)
        assert "imputer" in num_p.named_steps
        assert num_p.named_steps["imputer"].strategy == "median"

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_numeric_branch_has_variance_filter(self, request, fixture_name):
        pipeline, _ = request.getfixturevalue(fixture_name)
        _, _, num_p = _get_sub_pipelines(pipeline)
        assert "var_filter" in num_p.named_steps

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
    ])
    def test_numeric_branch_has_standard_scaler(self, request, fixture_name):
        pipeline, _ = request.getfixturevalue(fixture_name)
        _, _, num_p = _get_sub_pipelines(pipeline)
        assert "scaler" in num_p.named_steps
        assert isinstance(num_p.named_steps["scaler"], StandardScaler)

    def test_xgb_numeric_branch_has_no_scaler(self, xgb_pipeline_and_params):
        pipeline, _ = xgb_pipeline_and_params
        _, _, num_p = _get_sub_pipelines(pipeline)
        assert "scaler" not in num_p.named_steps

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_categorical_branch_has_ohe(self, request, fixture_name):
        pipeline, _ = request.getfixturevalue(fixture_name)
        _, cat_p, _ = _get_sub_pipelines(pipeline)
        assert "onehot" in cat_p.named_steps

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_ohe_drops_first_category(self, request, fixture_name):
        pipeline, _ = request.getfixturevalue(fixture_name)
        _, cat_p, _ = _get_sub_pipelines(pipeline)
        assert cat_p.named_steps["onehot"].drop == "first"

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_ohe_ignores_unknown_categories(self, request, fixture_name):
        pipeline, _ = request.getfixturevalue(fixture_name)
        _, cat_p, _ = _get_sub_pipelines(pipeline)
        assert cat_p.named_steps["onehot"].handle_unknown == "ignore"

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
    ])
    def test_categorical_branch_has_standard_scaler(self, request, fixture_name):
        pipeline, _ = request.getfixturevalue(fixture_name)
        _, cat_p, _ = _get_sub_pipelines(pipeline)
        assert "scaler" in cat_p.named_steps
        assert isinstance(cat_p.named_steps["scaler"], StandardScaler)

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_categorical_branch_has_variance_filter(self, request, fixture_name):
        pipeline, _ = request.getfixturevalue(fixture_name)
        _, cat_p, _ = _get_sub_pipelines(pipeline)
        assert "var_filter" in cat_p.named_steps

# Test classifier config

class TestClassifierConfig:

    def _clf(self, pipeline):
        return pipeline.named_steps["classifier"]

    def test_ridge_classifier_is_logistic_regression(self, ridge_pipeline_and_params):
        assert isinstance(self._clf(ridge_pipeline_and_params[0]), LogisticRegression)

    def test_ridge_l1_ratio_is_zero(self, ridge_pipeline_and_params):
        assert self._clf(ridge_pipeline_and_params[0]).l1_ratio == 0.0

    def test_ridge_default_solver_is_saga(self, ridge_pipeline_and_params):
        assert self._clf(ridge_pipeline_and_params[0]).solver == "saga"

    def test_ridge_default_max_iter_is_5000(self, ridge_pipeline_and_params):
        assert self._clf(ridge_pipeline_and_params[0]).max_iter == 5000

    def test_ridge_custom_solver(self):
        pipe, _ = get_logistic_ridge_pipeline_and_hyperparameters(
            CATEGORICAL_FEATURES, NUMERIC_FEATURES, solver="lbfgs"
        )
        assert pipe.named_steps["classifier"].solver == "lbfgs"

    def test_ridge_custom_max_iter(self):
        pipe, _ = get_logistic_ridge_pipeline_and_hyperparameters(
            CATEGORICAL_FEATURES, NUMERIC_FEATURES, max_iter=1000
        )
        assert pipe.named_steps["classifier"].max_iter == 1000

    def test_lasso_classifier_is_logistic_regression(self, lasso_pipeline_and_params):
        assert isinstance(self._clf(lasso_pipeline_and_params[0]), LogisticRegression)

    def test_lasso_l1_ratio_is_one(self, lasso_pipeline_and_params):
        assert self._clf(lasso_pipeline_and_params[0]).l1_ratio == 1.0

    def test_lasso_default_solver_is_saga(self, lasso_pipeline_and_params):
        assert self._clf(lasso_pipeline_and_params[0]).solver == "saga"

    def test_lasso_default_max_iter_is_5000(self, lasso_pipeline_and_params):
        assert self._clf(lasso_pipeline_and_params[0]).max_iter == 5000

    def test_lasso_custom_solver(self):
        pipe, _ = get_logistic_lasso_pipeline_and_hyperparameters(
            CATEGORICAL_FEATURES, NUMERIC_FEATURES, solver="lbfgs"
        )
        assert pipe.named_steps["classifier"].solver == "lbfgs"

    def test_lasso_custom_max_iter(self):
        pipe, _ = get_logistic_lasso_pipeline_and_hyperparameters(
            CATEGORICAL_FEATURES, NUMERIC_FEATURES, max_iter=200
        )
        assert pipe.named_steps["classifier"].max_iter == 200

    def test_xgb_classifier_is_xgbclassifier(self, xgb_pipeline_and_params):
        assert isinstance(self._clf(xgb_pipeline_and_params[0]), xgb.XGBClassifier)

    def test_xgb_objective_binary_logistic(self, xgb_pipeline_and_params):
        assert self._clf(xgb_pipeline_and_params[0]).objective == "binary:logistic"

    def test_xgb_default_scale_pos_weight_is_one(self, xgb_pipeline_and_params):
        assert self._clf(xgb_pipeline_and_params[0]).scale_pos_weight == 1

    def test_xgb_custom_scale_pos_weight(self):
        pipe, _ = get_xgboost_pipeline_and_hyperparameters(
            CATEGORICAL_FEATURES, NUMERIC_FEATURES, positive_class_weighting=10
        )
        assert pipe.named_steps["classifier"].scale_pos_weight == 10


# test fit predict

class TestFitPredict:

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_predictions_correct_length(self, request, fixture_name, sample_data):
        pipeline, _ = request.getfixturevalue(fixture_name)
        df, target = sample_data
        preds, _ = _fit_predict(pipeline, df, target)
        assert len(preds) == N_ROWS

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_predictions_are_binary(self, request, fixture_name, sample_data):
        pipeline, _ = request.getfixturevalue(fixture_name)
        df, target = sample_data
        preds, _ = _fit_predict(pipeline, df, target)
        assert set(preds).issubset({0, 1})

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_proba_output_shape(self, request, fixture_name, sample_data):
        pipeline, _ = request.getfixturevalue(fixture_name)
        df, target = sample_data
        _, probas = _fit_predict(pipeline, df, target)
        assert probas.shape == (N_ROWS, 2)

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_probas_sum_to_one(self, request, fixture_name, sample_data):
        pipeline, _ = request.getfixturevalue(fixture_name)
        df, target = sample_data
        _, probas = _fit_predict(pipeline, df, target)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("fixture_name", [
        "ridge_pipeline_and_params",
        "lasso_pipeline_and_params",
        "xgb_pipeline_and_params",
    ])
    def test_probas_within_unit_interval(self, request, fixture_name, sample_data):
        pipeline, _ = request.getfixturevalue(fixture_name)
        df, target = sample_data
        _, probas = _fit_predict(pipeline, df, target)
        assert (probas >= 0).all() and (probas <= 1).all()

# test robustness

class TestRobustness:

    @pytest.mark.parametrize("factory_fn", [
        get_logistic_ridge_pipeline_and_hyperparameters,
        get_logistic_lasso_pipeline_and_hyperparameters,
        get_xgboost_pipeline_and_hyperparameters,
    ], ids=["ridge", "lasso", "xgb"])
    def test_unseen_category_at_predict_time(self, factory_fn, sample_data):
        df, target = sample_data
        pipeline, _ = factory_fn(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
        pipeline.fit(df[ALL_FEATURES], target)
        df_test = df.head(10).copy()
        df_test["diagnosis_code"] = "UNSEEN_CODE_XYZ"
        pipeline.predict(df_test[ALL_FEATURES])  # must not raise

    @pytest.mark.parametrize("factory_fn", [
        get_logistic_ridge_pipeline_and_hyperparameters,
        get_logistic_lasso_pipeline_and_hyperparameters,
        get_xgboost_pipeline_and_hyperparameters,
    ], ids=["ridge", "lasso", "xgb"])
    def test_all_nan_numeric_column_at_predict_time(self, factory_fn, sample_data):
        df, target = sample_data
        pipeline, _ = factory_fn(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
        pipeline.fit(df[ALL_FEATURES], target)
        df_test = df.head(20).copy()
        df_test["bmi"] = np.nan
        pipeline.predict(df_test[ALL_FEATURES])  # must not raise

    @pytest.mark.parametrize("factory_fn", [
        get_logistic_ridge_pipeline_and_hyperparameters,
        get_logistic_lasso_pipeline_and_hyperparameters,
        get_xgboost_pipeline_and_hyperparameters,
    ], ids=["ridge", "lasso", "xgb"])
    def test_highly_imbalanced_target(self, factory_fn, sample_data):
        df, _ = sample_data
        rng = np.random.default_rng(0)
        imbalanced_target = pd.Series(
            rng.choice([0, 1], size=N_ROWS, p=[0.98, 0.02])
        )
        pipeline, _ = factory_fn(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
        pipeline.fit(df[ALL_FEATURES], imbalanced_target)  # must not raise

    @pytest.mark.parametrize("factory_fn", [
        get_logistic_ridge_pipeline_and_hyperparameters,
        get_logistic_lasso_pipeline_and_hyperparameters,
        get_xgboost_pipeline_and_hyperparameters,
    ], ids=["ridge", "lasso", "xgb"])
    def test_single_category_column_survives_variance_filter(self, factory_fn, sample_data):
        df, target = sample_data
        df_mono = df.copy()
        df_mono["gender"] = "Male"
        pipeline, _ = factory_fn(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
        pipeline.fit(df_mono[ALL_FEATURES], target)  # must not raise
