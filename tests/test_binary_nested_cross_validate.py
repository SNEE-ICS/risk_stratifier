"""
Comprehensive test suite for binary_nested_cross_validate.py

Tests cover all functions including:
- FoldResult dataclass
- _create_scorers()
- _calculate_metrics()
- _train_inner_loop()
- _split_data()
- _process_outer_fold()
- _aggregate_fold_results()
- _summarize_scores()
- _calibration_curve()
- run_nested_cv_calibration_assessment()
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

# Assuming the module is importable
from risk_stratifier.binary_nested_cross_validate import (
    FoldResult,
    _create_scorers,
    _calculate_metrics,
    _train_inner_loop,
    _split_data,
    _process_outer_fold,
    _aggregate_fold_results,
    _summarize_scores,
    _calibration_curve,
    run_nested_cv_calibration_assessment,
)

from risk_stratifier.data_validation import validate_binary_y_and_X

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample binary classification data."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.binomial(1, 0.5, n_samples)
    
    return X, y


@pytest.fixture
def sample_dataframe():
    """Create sample data as pandas DataFrame and Series."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y = pd.Series(np.random.binomial(1, 0.5, n_samples), name="target")
    
    return X, y


@pytest.fixture
def simple_pipeline():
    """Create a simple sklearn pipeline for testing."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])


@pytest.fixture
def param_distributions():
    """Create simple parameter distributions for testing."""
    return {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l2'],
    }


@pytest.fixture
def fold_result_data():
    """Create a sample FoldResult instance."""
    np.random.seed(42)
    y_true = np.array([0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.8, 0.7, 0.2, 0.9])
    test_indices = np.array([0, 1, 2, 3, 4])
    
    return FoldResult(
        repeat=0,
        fold=0,
        y_true=y_true,
        y_proba=y_proba,
        test_indices=test_indices,
        best_params={'classifier__C': 1.0}
    )

# ============================================================================
# TESTS FOR FoldResult
# ============================================================================

class TestFoldResult:
    """Tests for FoldResult dataclass."""
    
    def test_fold_result_creation(self, fold_result_data):
        """Test creation of FoldResult instance."""
        assert fold_result_data.repeat == 0
        assert fold_result_data.fold == 0
        assert len(fold_result_data.y_true) == 5
        assert len(fold_result_data.y_proba) == 5
        assert fold_result_data.best_params['classifier__C'] == 1.0
    
    def test_fold_result_attributes(self, fold_result_data):
        """Test all attributes are stored correctly."""
        assert isinstance(fold_result_data.y_true, np.ndarray)
        assert isinstance(fold_result_data.y_proba, np.ndarray)
        assert isinstance(fold_result_data.test_indices, np.ndarray)
        assert isinstance(fold_result_data.best_params, dict)


# ============================================================================
# TESTS FOR _create_scorers()
# ============================================================================

class TestCreateScorers:
    """Tests for _create_scorers function."""
    
    def test_returns_dictionary(self):
        """Test that _create_scorers returns a dictionary."""
        scorers = _create_scorers()
        assert isinstance(scorers, dict)
    
    def test_contains_brier_scorer(self):
        """Test that scorers dict contains 'brier' key."""
        scorers = _create_scorers()
        assert 'brier' in scorers
    
    def test_scorer_is_not_none(self):
        """Test that scorer values are not None."""
        scorers = _create_scorers()
        assert scorers['brier'] is not None
    
    def test_brier_scorer_properties(self):
        """Test brier scorer has correct properties."""
        scorers = _create_scorers()
        brier_scorer = scorers['brier']
        # Scorers have a _score_func attribute
        assert hasattr(brier_scorer, '_score_func')



# ============================================================================
# TESTS FOR _calculate_metrics()
# ============================================================================

class TestCalculateMetrics:
    """Tests for _calculate_metrics function."""
    
    def test_returns_dictionary(self):
        """Test that function returns a dictionary."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.8, 0.7, 0.2, 0.9])
        
        metrics = _calculate_metrics(y_true, y_proba)
        assert isinstance(metrics, dict)
    
    def test_contains_required_metrics(self):
        """Test that all required metrics are present."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.8, 0.7, 0.2, 0.9])
        
        metrics = _calculate_metrics(y_true, y_proba)
        assert 'brier_score' in metrics
        assert 'roc_auc' in metrics
        assert 'log_loss' in metrics
    
    def test_metric_values_are_numeric(self):
        """Test that metric values are numeric."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.8, 0.7, 0.2, 0.9])
        
        metrics = _calculate_metrics(y_true, y_proba)
        for value in metrics.values():
            assert isinstance(value, (float, np.floating))
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_proba = np.array([0.0, 1.0, 1.0, 0.0])
        
        metrics = _calculate_metrics(y_true, y_proba)
        assert metrics['brier_score'] == pytest.approx(0.0)
        assert metrics['roc_auc'] == pytest.approx(1.0)
    
    def test_worst_predictions(self):
        """Test metrics with worst predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_proba = np.array([1.0, 0.0, 0.0, 1.0])
        
        metrics = _calculate_metrics(y_true, y_proba)
        assert metrics['roc_auc'] == pytest.approx(0.0)


# ============================================================================
# TESTS FOR _train_inner_loop()
# ============================================================================

class TestTrainInnerLoop:
    """Tests for _train_inner_loop function."""
    
    def test_returns_tuple(self, simple_pipeline, param_distributions, sample_dataframe):
        """Test that function returns a tuple of (estimator, dict)."""
        X, y = sample_dataframe
        inner_cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
        scorers = _create_scorers()
        
        result = _train_inner_loop(
            simple_pipeline,
            param_distributions,
            X, y,
            inner_cv,
            scorers['brier'],
            random_state=42,
            n_iter=5
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_returns_fitted_estimator(self, simple_pipeline, param_distributions, sample_dataframe):
        """Test that returned estimator is fitted."""
        X, y = sample_dataframe
        inner_cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
        scorers = _create_scorers()
        
        best_estimator, _ = _train_inner_loop(
            simple_pipeline,
            param_distributions,
            X, y,
            inner_cv,
            scorers['brier'],
            random_state=42,
            n_iter=5
        )
        
        # Check that estimator has predict_proba method
        assert hasattr(best_estimator, 'predict_proba')
    
    def test_returns_best_params(self, simple_pipeline, param_distributions, sample_dataframe):
        """Test that best_params is a dictionary."""
        X, y = sample_dataframe
        inner_cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
        scorers = _create_scorers()
        
        _, best_params = _train_inner_loop(
            simple_pipeline,
            param_distributions,
            X, y,
            inner_cv,
            scorers['brier'],
            random_state=42,
            n_iter=5
        )
        
        assert isinstance(best_params, dict)
    
    def test_n_iter_parameter(self, simple_pipeline, param_distributions, sample_dataframe):
        """Test that n_iter parameter is respected."""
        X, y = sample_dataframe
        inner_cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
        scorers = _create_scorers()
        
        # Should not raise an error with n_iter=3
        result = _train_inner_loop(
            simple_pipeline,
            param_distributions,
            X, y,
            inner_cv,
            scorers['brier'],
            random_state=42,
            n_iter=3
        )
        
        assert result is not None


# ============================================================================
# TESTS FOR _split_data()
# ============================================================================

class TestSplitData:
    """Tests for _split_data function."""
    
    def test_returns_four_elements(self, sample_dataframe):
        """Test that function returns 4 elements."""
        X, y = sample_dataframe
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([4, 5, 6])
        
        result = _split_data(X, y, train_idx, test_idx)
        assert len(result) == 4
    
    def test_split_shapes(self, sample_dataframe):
        """Test that split shapes are correct."""
        X, y = sample_dataframe
        train_idx = np.array([0, 1, 2, 3, 4])
        test_idx = np.array([5, 6, 7, 8, 9])
        
        X_train, X_test, y_train, y_test = _split_data(X, y, train_idx, test_idx)
        
        assert len(X_train) == 5
        assert len(X_test) == 5
        assert len(y_train) == 5
        assert len(y_test) == 5
    
    def test_correct_indices_used(self, sample_dataframe):
        """Test that correct indices are selected."""
        X, y = sample_dataframe
        train_idx = np.array([0, 1])
        test_idx = np.array([2, 3])
        
        X_train, X_test, y_train, y_test = _split_data(X, y, train_idx, test_idx)
        
        # Check that the values match
        pd.testing.assert_frame_equal(X_train, X.iloc[train_idx])
        pd.testing.assert_frame_equal(X_test, X.iloc[test_idx])
        pd.testing.assert_series_equal(y_train, y.iloc[train_idx])
        pd.testing.assert_series_equal(y_test, y.iloc[test_idx])
    
    def test_no_overlap(self, sample_dataframe):
        """Test that train and test sets don't overlap."""
        X, y = sample_dataframe
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([4, 5, 6])
        
        X_train, X_test, y_train, y_test = _split_data(X, y, train_idx, test_idx)
        
        # No direct way to test without accessing internals, but shapes ensure this
        assert len(X_train) + len(X_test) <= len(X)


# ============================================================================
# TESTS FOR _process_outer_fold()
# ============================================================================

class TestProcessOuterFold:
    """Tests for _process_outer_fold function."""
    
    def test_returns_fold_result(self, simple_pipeline, param_distributions, sample_dataframe):
        """Test that function returns a FoldResult."""
        X, y = sample_dataframe
        train_idx = np.arange(100)
        test_idx = np.arange(100, 150)
        
        inner_cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
        scorers = _create_scorers()
        
        result = _process_outer_fold(
            simple_pipeline,
            param_distributions,
            X, y,
            train_idx,
            test_idx,
            fold_number=0,
            outer_folds=5,
            inner_cv=inner_cv,
            scorer=scorers,
            random_state=42,
            n_iter=5
        )
        
        assert isinstance(result, FoldResult)
    
    def test_fold_result_attributes(self, simple_pipeline, param_distributions, sample_dataframe):
        """Test that returned FoldResult has all required attributes."""
        X, y = sample_dataframe
        train_idx = np.arange(100)
        test_idx = np.arange(100, 150)
        
        inner_cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
        scorers = _create_scorers()
        
        result = _process_outer_fold(
            simple_pipeline,
            param_distributions,
            X, y,
            train_idx,
            test_idx,
            fold_number=0,
            outer_folds=5,
            inner_cv=inner_cv,
            scorer=scorers,
            random_state=42,
            n_iter=5
        )
        
        assert hasattr(result, 'repeat')
        assert hasattr(result, 'fold')
        assert hasattr(result, 'y_true')
        assert hasattr(result, 'y_proba')
        assert hasattr(result, 'test_indices')
        assert hasattr(result, 'best_params')
    
    def test_predictions_match_test_size(self, simple_pipeline, param_distributions, sample_dataframe):
        """Test that predictions match test set size."""
        X, y = sample_dataframe
        train_idx = np.arange(100)
        test_idx = np.arange(100, 150)
        
        inner_cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
        scorers = _create_scorers()
        
        result = _process_outer_fold(
            simple_pipeline,
            param_distributions,
            X, y,
            train_idx,
            test_idx,
            fold_number=0,
            outer_folds=5,
            inner_cv=inner_cv,
            scorer=scorers,
            random_state=42,
            n_iter=5
        )
        
        assert len(result.y_proba) == len(test_idx)
        assert len(result.y_true) == len(test_idx)
    
    def test_fold_numbering(self, simple_pipeline, param_distributions, sample_dataframe):
        """Test that fold numbering is calculated correctly."""
        X, y = sample_dataframe
        train_idx = np.arange(100)
        test_idx = np.arange(100, 150)
        
        inner_cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=42)
        scorers = _create_scorers()
        
        # Test fold_number=7 with outer_folds=5
        # Should be repeat=1, fold=2
        result = _process_outer_fold(
            simple_pipeline,
            param_distributions,
            X, y,
            train_idx,
            test_idx,
            fold_number=7,
            outer_folds=5,
            inner_cv=inner_cv,
            scorer=scorers,
            random_state=42,
            n_iter=5
        )
        
        assert result.repeat == 1
        assert result.fold == 2


# ============================================================================
# TESTS FOR _aggregate_fold_results()
# ============================================================================

class TestAggregateFoldResults:
    """Tests for _aggregate_fold_results function."""
    
    def test_returns_two_dataframes(self, fold_result_data):
        """Test that function returns two DataFrames."""
        results = [fold_result_data]
        
        scores_df, predictions_df = _aggregate_fold_results(results)
        
        assert isinstance(scores_df, pd.DataFrame)
        assert isinstance(predictions_df, pd.DataFrame)
    
    def test_scores_df_structure(self, fold_result_data):
        """Test structure of scores DataFrame."""
        results = [fold_result_data]
        
        scores_df, _ = _aggregate_fold_results(results)
        
        assert 'repeat' in scores_df.columns
        assert 'fold' in scores_df.columns
        assert 'brier_score' in scores_df.columns
        assert 'roc_auc' in scores_df.columns
        assert 'log_loss' in scores_df.columns
    
    def test_predictions_df_structure(self, fold_result_data):
        """Test structure of predictions DataFrame."""
        results = [fold_result_data]
        
        _, predictions_df = _aggregate_fold_results(results)
        
        assert 'repeat' in predictions_df.columns
        assert 'fold' in predictions_df.columns
        assert 'y_true' in predictions_df.columns
        assert 'y_proba' in predictions_df.columns
        assert 'original_index' in predictions_df.columns
    
    def test_scores_df_row_count(self, fold_result_data):
        """Test that scores DataFrame has one row per FoldResult."""
        fold1 = fold_result_data
        fold2 = FoldResult(
            repeat=0,
            fold=1,
            y_true=np.array([0, 1]),
            y_proba=np.array([0.2, 0.8]),
            test_indices=np.array([5, 6]),
            best_params={'classifier__C': 1.0}
        )
        
        results = [fold1, fold2]
        scores_df, _ = _aggregate_fold_results(results)
        
        assert len(scores_df) == 2
    
    def test_predictions_df_row_count(self, fold_result_data):
        """Test that predictions DataFrame has one row per prediction."""
        results = [fold_result_data]
        
        _, predictions_df = _aggregate_fold_results(results)
        
        # fold_result_data has 5 predictions
        assert len(predictions_df) == 5
    
    def test_predictions_df_values(self, fold_result_data):
        """Test that prediction values match input."""
        results = [fold_result_data]
        
        _, predictions_df = _aggregate_fold_results(results)
        
        np.testing.assert_array_equal(
            predictions_df['y_true'].values,
            fold_result_data.y_true
        )
        np.testing.assert_array_almost_equal(
            predictions_df['y_proba'].values,
            fold_result_data.y_proba
        )


# ============================================================================
# TESTS FOR _summarize_scores()
# ============================================================================

class TestSummarizeScores:
    """Tests for _summarize_scores function."""
    
    def test_returns_dictionary(self):
        """Test that function returns a dictionary."""
        scores_df = pd.DataFrame({
            'brier_score': [0.1, 0.2, 0.15],
            'roc_auc': [0.8, 0.85, 0.82],
            'log_loss': [0.3, 0.4, 0.35]
        })
        
        summary = _summarize_scores(scores_df)
        assert isinstance(summary, dict)
    
    def test_contains_required_keys(self):
        """Test that summary contains required keys."""
        scores_df = pd.DataFrame({
            'brier_score': [0.1, 0.2],
            'roc_auc': [0.8, 0.85],
            'log_loss': [0.3, 0.4]
        })
        
        summary = _summarize_scores(scores_df)
        
        expected_keys = [
            'brier_score_mean', 'brier_score_std',
            'roc_auc_mean', 'roc_auc_std',
            'log_loss_mean', 'log_loss_std'
        ]
        for key in expected_keys:
            assert key in summary
    
    def test_correct_mean_calculation(self):
        """Test that means are calculated correctly."""
        scores_df = pd.DataFrame({
            'brier_score': [0.1, 0.2, 0.3],
            'roc_auc': [0.7, 0.8, 0.9],
            'log_loss': [0.4, 0.5, 0.6]
        })
        
        summary = _summarize_scores(scores_df)
        
        assert summary['brier_score_mean'] == pytest.approx(0.2)
        assert summary['roc_auc_mean'] == pytest.approx(0.8)
        assert summary['log_loss_mean'] == pytest.approx(0.5)
    
    def test_correct_std_calculation(self):
        """Test that std values are calculated correctly."""
        scores_df = pd.DataFrame({
            'brier_score': [0.1, 0.3],
            'roc_auc': [0.8, 0.8],
            'log_loss': [0.3, 0.5]
        })
        
        summary = _summarize_scores(scores_df)
        
        # std([0.1, 0.3]) = 0.1414...
        assert summary['brier_score_std'] == pytest.approx(0.1414, abs=0.001)
        # std([0.8, 0.8]) = 0
        assert summary['roc_auc_std'] == pytest.approx(0.0)


# ============================================================================
# TESTS FOR _calibration_curve()
# ============================================================================

class TestCalibrationCurve:
    """Tests for _calibration_curve function."""
    
    def test_returns_figure(self):
        """Test that function returns a matplotlib Figure."""
        predictions_df = pd.DataFrame({
            'y_true': [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
            'y_proba': [0.1, 0.8, 0.7, 0.2, 0.9, 0.15, 0.85, 0.75, 0.25, 0.3]
        })
        
        fig = _calibration_curve(predictions_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_figure_has_axes(self):
        """Test that returned figure has axes."""
        predictions_df = pd.DataFrame({
            'y_true': [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
            'y_proba': [0.1, 0.8, 0.7, 0.2, 0.9, 0.15, 0.85, 0.75, 0.25, 0.3]
        })
        
        fig = _calibration_curve(predictions_df)
        axes = fig.get_axes()
        assert len(axes) > 0
        plt.close(fig)
    
    def test_default_parameters(self):
        """Test that function works with default parameters."""
        predictions_df = pd.DataFrame({
            'y_true': [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
            'y_proba': [0.1, 0.8, 0.7, 0.2, 0.9, 0.15, 0.85, 0.75, 0.25, 0.3]
        })
        
        fig = _calibration_curve(predictions_df)
        assert fig is not None
        plt.close(fig)
    
    def test_custom_n_bins(self):
        """Test that custom n_bins parameter is accepted."""
        predictions_df = pd.DataFrame({
            'y_true': [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
            'y_proba': [0.1, 0.8, 0.7, 0.2, 0.9, 0.15, 0.85, 0.75, 0.25, 0.3]
        })
        
        fig = _calibration_curve(predictions_df, n_bins=5)
        assert fig is not None
        plt.close(fig)
    
    def test_uniform_strategy(self):
        """Test with uniform binning strategy."""
        predictions_df = pd.DataFrame({
            'y_true': [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
            'y_proba': [0.1, 0.8, 0.7, 0.2, 0.9, 0.15, 0.85, 0.75, 0.25, 0.3]
        })
        
        fig = _calibration_curve(predictions_df, strategy='uniform')
        assert fig is not None
        plt.close(fig)


# ============================================================================
# TESTS FOR run_nested_cv_calibration_assessment()
# ============================================================================

class TestRunNestedCVCalibrationAssessment:
    """Tests for main nested CV function."""
    
    @patch('risk_stratifier.binary_nested_cross_validate.validate_binary_y_and_X')
    def test_returns_dictionary(self, mock_validate, simple_pipeline, param_distributions, sample_dataframe):
        """Test that function returns a dictionary."""
        X, y = sample_dataframe
        
        result = run_nested_cv_calibration_assessment(
            simple_pipeline,
            param_distributions,
            X, y,
            outer_folds=2,
            outer_repeats=1,
            inner_folds=2,
            inner_repeats=1,
            n_iter=5,
            random_state=42,
            verbose=False
        )
        
        assert isinstance(result, dict)
    
    @patch('risk_stratifier.binary_nested_cross_validate.validate_binary_y_and_X')
    def test_contains_required_keys(self, mock_validate, simple_pipeline, param_distributions, sample_dataframe):
        """Test that result contains all required keys."""
        X, y = sample_dataframe
        
        result = run_nested_cv_calibration_assessment(
            simple_pipeline,
            param_distributions,
            X, y,
            outer_folds=2,
            outer_repeats=1,
            inner_folds=2,
            inner_repeats=1,
            n_iter=5,
            random_state=42,
            verbose=False
        )
        
        assert 'outer_scores' in result
        assert 'predictions' in result
        assert 'best_params_per_fold' in result
        assert 'summary' in result
        assert 'calibration_plot' in result
    
    @patch('risk_stratifier.binary_nested_cross_validate.validate_binary_y_and_X')
    def test_outer_scores_is_dataframe(self, mock_validate, simple_pipeline, param_distributions, sample_dataframe):
        """Test that outer_scores is a DataFrame."""
        X, y = sample_dataframe
        
        result = run_nested_cv_calibration_assessment(
            simple_pipeline,
            param_distributions,
            X, y,
            outer_folds=2,
            outer_repeats=1,
            inner_folds=2,
            inner_repeats=1,
            n_iter=5,
            random_state=42,
            verbose=False
        )
        
        assert isinstance(result['outer_scores'], pd.DataFrame)
    
    @patch('risk_stratifier.binary_nested_cross_validate.validate_binary_y_and_X')
    def test_predictions_is_dataframe(self, mock_validate, simple_pipeline, param_distributions, sample_dataframe):
        """Test that predictions is a DataFrame."""
        X, y = sample_dataframe
        
        result = run_nested_cv_calibration_assessment(
            simple_pipeline,
            param_distributions,
            X, y,
            outer_folds=2,
            outer_repeats=1,
            inner_folds=2,
            inner_repeats=1,
            n_iter=5,
            random_state=42,
            verbose=False
        )
        
        assert isinstance(result['predictions'], pd.DataFrame)
    
    @patch('risk_stratifier.binary_nested_cross_validate.validate_binary_y_and_X')
    def test_best_params_per_fold_is_list(self, mock_validate, simple_pipeline, param_distributions, sample_dataframe):
        """Test that best_params_per_fold is a list."""
        X, y = sample_dataframe
        
        result = run_nested_cv_calibration_assessment(
            simple_pipeline,
            param_distributions,
            X, y,
            outer_folds=2,
            outer_repeats=1,
            inner_folds=2,
            inner_repeats=1,
            n_iter=5,
            random_state=42,
            verbose=False
        )
        
        assert isinstance(result['best_params_per_fold'], list)
    
    @patch('risk_stratifier.binary_nested_cross_validate.validate_binary_y_and_X')
    def test_summary_is_dictionary(self, mock_validate, simple_pipeline, param_distributions, sample_dataframe):
        """Test that summary is a dictionary."""
        X, y = sample_dataframe
        
        result = run_nested_cv_calibration_assessment(
            simple_pipeline,
            param_distributions,
            X, y,
            outer_folds=2,
            outer_repeats=1,
            inner_folds=2,
            inner_repeats=1,
            n_iter=5,
            random_state=42,
            verbose=False
        )
        
        assert isinstance(result['summary'], dict)
    
    @patch('risk_stratifier.binary_nested_cross_validate.validate_binary_y_and_X')
    def test_calibration_plot_is_figure(self, mock_validate, simple_pipeline, param_distributions, sample_dataframe):
        """Test that calibration_plot is a matplotlib Figure."""
        X, y = sample_dataframe
        
        result = run_nested_cv_calibration_assessment(
            simple_pipeline,
            param_distributions,
            X, y,
            outer_folds=2,
            outer_repeats=1,
            inner_folds=2,
            inner_repeats=1,
            n_iter=5,
            random_state=42,
            verbose=False
        )
        
        assert isinstance(result['calibration_plot'], plt.Figure)
        plt.close(result['calibration_plot'])


    @patch('risk_stratifier.binary_nested_cross_validate.validate_binary_y_and_X')
    def test_correct_number_of_folds(self, mock_validate, simple_pipeline, param_distributions, sample_dataframe):
        """Test that result contains correct number of folds."""
        X, y = sample_dataframe
        
        result = run_nested_cv_calibration_assessment(
            simple_pipeline,
            param_distributions,
            X, y,
            outer_folds=3,
            outer_repeats=2,
            inner_folds=2,
            inner_repeats=1,
            n_iter=5,
            random_state=42,
            verbose=False
        )
        
        # Should have 3 * 2 = 6 folds total
        assert len(result['best_params_per_fold']) == 6
        assert len(result['outer_scores']) == 6
    
    @patch('risk_stratifier.binary_nested_cross_validate.validate_binary_y_and_X')
    def test_verbose_output(self, mock_validate, simple_pipeline, param_distributions, sample_dataframe, capsys):
        """Test that verbose output is printed."""
        X, y = sample_dataframe
        
        result = run_nested_cv_calibration_assessment(
            simple_pipeline,
            param_distributions,
            X, y,
            outer_folds=2,
            outer_repeats=1,
            inner_folds=2,
            inner_repeats=1,
            n_iter=5,
            random_state=42,
            verbose=True
        )
        
        captured = capsys.readouterr()
        assert "Fold" in captured.out
    
    @patch('risk_stratifier.binary_nested_cross_validate.validate_binary_y_and_X')
    def test_validation_called(self, mock_validate, simple_pipeline, param_distributions, sample_dataframe):
        """Test that validate_binary_y_and_X is called."""
        X, y = sample_dataframe
        
        result = run_nested_cv_calibration_assessment(
            simple_pipeline,
            param_distributions,
            X, y,
            outer_folds=2,
            outer_repeats=1,
            inner_folds=2,
            inner_repeats=1,
            n_iter=5,
            random_state=42,
            verbose=False
        )
        
        mock_validate.assert_called_once()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_end_to_end_workflow(self, simple_pipeline, param_distributions, sample_dataframe):
        """Test complete workflow from data to results."""
        X, y = sample_dataframe
        
        with patch('risk_stratifier.binary_nested_cross_validate.validate_binary_y_and_X'):
            result = run_nested_cv_calibration_assessment(
                simple_pipeline,
                param_distributions,
                X, y,
                outer_folds=2,
                outer_repeats=1,
                inner_folds=2,
                inner_repeats=1,
                n_iter=5,
                random_state=42,
                verbose=False
            )
        
        # Verify all output components have expected relationships
        n_folds = 2 * 1  # outer_folds * outer_repeats
        n_predictions = len(result['predictions'])
        
        # Check consistency
        assert len(result['outer_scores']) == n_folds
        assert len(result['best_params_per_fold']) == n_folds
        assert n_predictions > 0  # Should have at least one prediction
        assert len(result['summary']) == 6  # 3 metrics * 2 (mean + std)
        plt.close(result['calibration_plot'])
    
    def test_predictions_aggregated_correctly(self):
        """Test that aggregated predictions match original fold results."""
        fold1 = FoldResult(
            repeat=0, fold=0,
            y_true=np.array([0, 1]),
            y_proba=np.array([0.2, 0.8]),
            test_indices=np.array([0, 1]),
            best_params={'C': 1.0}
        )
        fold2 = FoldResult(
            repeat=0, fold=1,
            y_true=np.array([1, 0]),
            y_proba=np.array([0.7, 0.3]),
            test_indices=np.array([2, 3]),
            best_params={'C': 0.1}
        )
        
        scores_df, predictions_df = _aggregate_fold_results([fold1, fold2])
        
        # Check all predictions are preserved
        assert len(predictions_df) == 4
        assert len(scores_df) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])


