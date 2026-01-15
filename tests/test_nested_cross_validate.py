import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
import risk_stratifier.nested_cross_validate as testing_module
import pytest

#________________________________________________________________________________________

# testing FoldResult class

def test_foldresult_two_instances_same_data_fieldwise():
    y_true = np.array([0, 1])
    y_proba = np.array([0.3, 0.7])
    indices = np.array([5, 6])
    best_params = {"alpha": 0.1}

    r1 = testing_module.FoldResult(0, 0, y_true, y_proba, indices, best_params)
    r2 = testing_module.FoldResult(0, 0, y_true.copy(), y_proba.copy(), indices.copy(), best_params.copy())

    assert r1.repeat == r2.repeat
    assert r1.fold == r2.fold
    assert np.array_equal(r1.y_true, r2.y_true)
    assert np.array_equal(r1.y_proba, r2.y_proba)
    assert np.array_equal(r1.test_indices, r2.test_indices)
    assert r1.best_params == r2.best_params

#________________________________________________________________________________________

# test _make_scorer function

def test_create_scorers_returns_brier_scorer():
    scorers = testing_module._create_scorers()

    # It should be a dict with a 'brier' key
    assert isinstance(scorers, dict)
    assert "brier" in scorers

    # The object should be callable as a sklearn scorer
    scorer = scorers["brier"]
    assert callable(scorer)

def test_brier_scorer_matches_negative_brier_score():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(random_state=0).fit(X, y)

    scorers = testing_module._create_scorers()
    scorer = scorers["brier"]

    # scorer has the signature scorer(estimator, X, y)
    score = scorer(model, X, y)

    # Compute the "true" Brier score from probabilities
    y_proba = model.predict_proba(X)[:, 1]
    true_brier = brier_score_loss(y, y_proba)

    # Because greater_is_better=False, scorer should return negative brier
    assert np.isclose(score, -true_brier)

#________________________________________________________________________________________

# testing _calculate_metrics function

def test_calculate_metrics_returns_expected_values():
    # Simple binary example with non-trivial probabilities
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])

    result = testing_module._calculate_metrics(y_true, y_proba)

    # Check keys
    assert set(result.keys()) == {"brier_score", "roc_auc", "log_loss"}

    # Check each metric matches sklearn's direct computation
    expected_brier = brier_score_loss(y_true, y_proba)
    expected_roc_auc = roc_auc_score(y_true, y_proba)
    expected_log_loss = log_loss(y_true, y_proba)

    assert np.isclose(result["brier_score"], expected_brier)
    assert np.isclose(result["roc_auc"], expected_roc_auc)
    assert np.isclose(result["log_loss"], expected_log_loss)

def test_calculate_metrics_perfect_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.0, 1.0, 0.0, 1.0])  # perfect

    result = testing_module._calculate_metrics(y_true, y_proba)

    # Brier score and log loss should be 0 for perfect calibrated predictions
    assert np.isclose(result["brier_score"], 0.0)
    assert np.isclose(result["log_loss"], 0.0)

    # ROC-AUC should be 1 for perfect separation
    assert np.isclose(result["roc_auc"], 1.0)

#________________________________________________________________________________________

# testing _train_inner_loop

def test_train_inner_loop_returns_estimator_and_params():
    # Small toy dataset
    X_train = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y_train = pd.Series([0, 0, 1, 1])

    pipeline = LogisticRegression(max_iter=1000)

    # Very small search space to keep the test fast
    param_distributions = {
        "C": [0.001, 0.01, 0.1, 1.0],
    }

    inner_cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=0)

    scorers = testing_module._create_scorers()
    brier_scorer = scorers["brier"]

    best_estimator, best_params = testing_module._train_inner_loop(
        pipeline=pipeline,
        param_distributions=param_distributions,
        X_train=X_train,
        y_train=y_train,
        inner_cv=inner_cv,
        scorer=brier_scorer,
        random_state=0,
        n_iter=2,
        n_jobs=1,
    )

    # Types / structure
    assert best_estimator is not None
    assert isinstance(best_params, dict)
    assert "C" in best_params



#________________________________________________________________________________________

# testing _split_data function


def test_split_data_splits_by_indices_correctly():
    # Create simple labelled data so we can track rows
    X = pd.DataFrame(
        {"feature": [10, 11, 12, 13, 14]},
        index=[100, 101, 102, 103, 104],  # non‑default index to ensure .iloc is used
    )
    y = pd.Series(
        [0, 1, 0, 1, 0],
        index=[100, 101, 102, 103, 104],
        name="target",
    )

    train_idx = np.array([0, 2, 4])  # rows with original index 100, 102, 104
    test_idx = np.array([1, 3])      # rows with original index 101, 103

    X_train, X_test, y_train, y_test = testing_module._split_data(
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    # Types preserved
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # Shapes correct
    assert X_train.shape == (3, 1)
    assert X_test.shape == (2, 1)
    assert y_train.shape == (3,)
    assert y_test.shape == (2,)

    # Correct rows chosen (via index or values)
    assert list(X_train["feature"]) == [10, 12, 14]
    assert list(X_test["feature"]) == [11, 13]
    assert list(y_train) == [0, 0, 0]
    assert list(y_test) == [1, 1]

    # Indices correspond to iloc selection, not original labels
    assert list(X_train.index) == [100, 102, 104]
    assert list(X_test.index) == [101, 103]
    assert list(y_train.index) == [100, 102, 104]
    assert list(y_test.index) == [101, 103]

#________________________________________________________________________________________

# testing _process_outer_fold function

def test_process_outer_fold_returns_foldresult_with_correct_structure():
    # Small balanced dataset
    X = pd.DataFrame({"feature": np.linspace(0, 3, 60)})
    y = pd.Series([0] * 30 + [1] * 30)

    train_idx = np.arange(0, 40)  # first 40 rows
    test_idx = np.arange(40, 60)  # last 20 rows

    pipeline = LogisticRegression(max_iter=1000)
    param_distributions = {"C": [0.1, 1.0]}

    inner_cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=0)
    scorers = testing_module._create_scorers()

    result = testing_module._process_outer_fold(
        pipeline=pipeline,
        param_distributions=param_distributions,
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        fold_number=2,  # 3rd fold (0-indexed)
        outer_folds=5,
        inner_cv=inner_cv,
        scorer=scorers,
        random_state=0,
        n_iter=2,
        n_jobs=1,
    )

    # Result is a FoldResult instance
    assert isinstance(result, testing_module.FoldResult)

    # Repeat and fold calculated correctly: fold_number=2, outer_folds=5
    # repeat = 2 // 5 = 0
    # fold = 2 % 5 = 2
    assert result.repeat == 0
    assert result.fold == 2

    # Predictions and truth match test set size
    assert result.y_true.shape == (20,)
    assert result.y_proba.shape == (20,)
    assert len(result.test_indices) == 20

    # Test indices match what we provided
    assert np.array_equal(result.test_indices, test_idx)

    # y_true matches the test portion of y
    expected_y_test = y.iloc[test_idx].values
    assert np.array_equal(result.y_true, expected_y_test)

    # best_params is a dict and non-empty
    assert isinstance(result.best_params, dict)
    assert "C" in result.best_params

    # Probabilities are valid (between 0 and 1)
    assert np.all((result.y_proba >= 0) & (result.y_proba <= 1))

def test_process_outer_fold_calculates_repeat_and_fold_correctly():
    # Minimal data
    X = pd.DataFrame({"x": range(120)})
    y = pd.Series([0, 1] * 60)

    train_idx = np.arange(0, 100)
    test_idx = np.arange(100, 120)

    pipeline = LogisticRegression(max_iter=500)
    param_distributions = {"C": [1.0]}
    inner_cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=1, random_state=0)
    scorers = testing_module._create_scorers()

    # Test several fold_number values with outer_folds=5
    test_cases = [
        # (fold_number, outer_folds, expected_repeat, expected_fold)
        (0, 5, 0, 0),
        (4, 5, 0, 4),
        (5, 5, 1, 0),  # start of second repeat
        (9, 5, 1, 4),
        (10, 5, 2, 0), # start of third repeat
    ]

    for fold_num, outer_f, exp_repeat, exp_fold in test_cases:
        result = testing_module._process_outer_fold(
            pipeline=pipeline,
            param_distributions=param_distributions,
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            fold_number=fold_num,
            outer_folds=outer_f,
            inner_cv=inner_cv,
            scorer=scorers,
            random_state=0,
            n_iter=1,
            n_jobs=1,
        )

        assert result.repeat == exp_repeat, f"Failed for fold_number={fold_num}"
        assert result.fold == exp_fold, f"Failed for fold_number={fold_num}"

#________________________________________________________________________________________

# testing nested_cv_with_probabilities function

def test_nested_cv_with_probabilities_returns_expected_structure():
    # Small balanced dataset (needs >100 rows if your validator checks that)
    np.random.seed(42)
    X = pd.DataFrame({
        "x1": np.random.randn(120),
        "x2": np.random.randn(120),
    })
    y = pd.Series([0] * 60 + [1] * 60)

    pipeline = LogisticRegression(max_iter=500)
    param_distributions = {"C": [0.1, 1.0]}

    results = testing_module.nested_cv_with_probabilities(
        pipeline=pipeline,
        param_distributions=param_distributions,
        X=X,
        y=y,
        outer_folds=3,
        outer_repeats=1,
        inner_folds=2,
        inner_repeats=1,
        n_iter=2,
        random_state=0,
        n_jobs=1,
        verbose=False,
    )

    # Check keys
    assert set(results.keys()) == {
        "outer_scores",
        "predictions",
        "best_params_per_fold",
        "summary",
    }

    # outer_scores is a DataFrame with one row per outer fold
    outer_scores = results["outer_scores"]
    assert isinstance(outer_scores, pd.DataFrame)
    assert outer_scores.shape[0] == 3  # 3 folds * 1 repeat
    assert set(outer_scores.columns) >= {
        "repeat",
        "fold",
        "brier_score",
        "roc_auc",
        "log_loss",
    }

    # predictions is a DataFrame with one row per test sample across all folds
    predictions = results["predictions"]
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape[0] == 120  # entire dataset appears once
    assert set(predictions.columns) >= {
        "repeat",
        "fold",
        "y_true",
        "y_proba",
        "original_index",
    }

    # best_params_per_fold is a list of dicts
    best_params = results["best_params_per_fold"]
    assert isinstance(best_params, list)
    assert len(best_params) == 3
    assert all(isinstance(p, dict) for p in best_params)
    assert all("C" in p for p in best_params)

    # summary is a dict with mean and std for each metric
    summary = results["summary"]
    assert isinstance(summary, dict)
    expected_keys = {
        "brier_score_mean",
        "brier_score_std",
        "roc_auc_mean",
        "roc_auc_std",
        "log_loss_mean",
        "log_loss_std",
    }
    assert set(summary.keys()) == expected_keys

    # Sanity checks on metric ranges
    assert 0 <= summary["brier_score_mean"] <= 1
    assert 0 <= summary["roc_auc_mean"] <= 1
    assert summary["log_loss_mean"] >= 0


def test_nested_cv_predictions_cover_entire_dataset():
    np.random.seed(123)
    X = pd.DataFrame({"feature": np.random.randn(120)})
    y = pd.Series([0, 1] * 60)

    pipeline = LogisticRegression(max_iter=500)
    param_distributions = {"C": [1.0]}

    results = testing_module.nested_cv_with_probabilities(
        pipeline=pipeline,
        param_distributions=param_distributions,
        X=X,
        y=y,
        outer_folds=5,
        outer_repeats=1,
        inner_folds=2,
        inner_repeats=1,
        n_iter=1,
        random_state=42,
        n_jobs=1,
        verbose=False,
    )

    predictions = results["predictions"]
    original_indices = predictions["original_index"].values

    # Every row in the original dataset should appear exactly once
    assert len(original_indices) == 120
    assert set(original_indices) == set(range(120))

    # No duplicates
    assert len(original_indices) == len(set(original_indices))

    # y_true in predictions should match original y
    for idx in range(120):
        pred_row = predictions[predictions["original_index"] == idx].iloc[0]
        assert pred_row["y_true"] == y.iloc[idx]


def test_nested_cv_with_multiple_repeats():
    np.random.seed(0)
    X = pd.DataFrame({"x": np.random.randn(120)})
    y = pd.Series([0, 1] * 60)

    pipeline = LogisticRegression(max_iter=500)
    param_distributions = {"C": [1.0]}

    results = testing_module.nested_cv_with_probabilities(
        pipeline=pipeline,
        param_distributions=param_distributions,
        X=X,
        y=y,
        outer_folds=3,
        outer_repeats=2,  # 2 repeats
        inner_folds=2,
        inner_repeats=1,
        n_iter=1,
        random_state=99,
        n_jobs=1,
        verbose=False,
    )

    outer_scores = results["outer_scores"]
    
    # Total folds = outer_folds * outer_repeats = 3 * 2 = 6
    assert outer_scores.shape[0] == 6
    
    # Check repeat numbers are correct
    assert set(outer_scores["repeat"]) == {0, 1}
    
    # Each repeat should have all fold numbers 0, 1, 2
    for repeat in [0, 1]:
        repeat_folds = outer_scores[outer_scores["repeat"] == repeat]["fold"].values
        assert set(repeat_folds) == {0, 1, 2}
    
    # best_params_per_fold should also have 6 entries
    assert len(results["best_params_per_fold"]) == 6


def test_nested_cv_verbose_flag(capsys):
    np.random.seed(11)
    X = pd.DataFrame({"x": np.random.randn(120)})
    y = pd.Series([0, 1] * 60)

    pipeline = LogisticRegression(max_iter=500)
    param_distributions = {"C": [1.0]}

    # Run with verbose=True
    testing_module.nested_cv_with_probabilities(
        pipeline=pipeline,
        param_distributions=param_distributions,
        X=X,
        y=y,
        outer_folds=2,
        outer_repeats=1,
        inner_folds=2,
        inner_repeats=1,
        n_iter=1,
        random_state=0,
        n_jobs=1,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "Fold" in captured.out
    assert "Brier=" in captured.out

    # Run with verbose=False
    testing_module.nested_cv_with_probabilities(
        pipeline=pipeline,
        param_distributions=param_distributions,
        X=X,
        y=y,
        outer_folds=2,
        outer_repeats=1,
        inner_folds=2,
        inner_repeats=1,
        n_iter=1,
        random_state=0,
        n_jobs=1,
        verbose=False,
    )

    captured_silent = capsys.readouterr()
    assert captured_silent.out == ""



def test_nested_cv_summary_matches_scores():
    np.random.seed(42)
    X = pd.DataFrame({"x": np.random.randn(120)})
    y = pd.Series([0, 1] * 60)

    pipeline = LogisticRegression(max_iter=500)
    param_distributions = {"C": [1.0]}

    results = testing_module.nested_cv_with_probabilities(
        pipeline=pipeline,
        param_distributions=param_distributions,
        X=X,
        y=y,
        outer_folds=4,
        outer_repeats=1,
        inner_folds=2,
        inner_repeats=1,
        n_iter=1,
        random_state=77,
        n_jobs=1,
        verbose=False,
    )

    outer_scores = results["outer_scores"]
    summary = results["summary"]

    # Summary mean/std should match DataFrame calculations
    assert np.isclose(
        summary["brier_score_mean"], outer_scores["brier_score"].mean()
    )
    assert np.isclose(summary["brier_score_std"], outer_scores["brier_score"].std())

    assert np.isclose(summary["roc_auc_mean"], outer_scores["roc_auc"].mean())
    assert np.isclose(summary["roc_auc_std"], outer_scores["roc_auc"].std())

    assert np.isclose(summary["log_loss_mean"], outer_scores["log_loss"].mean())
    assert np.isclose(summary["log_loss_std"], outer_scores["log_loss"].std())



