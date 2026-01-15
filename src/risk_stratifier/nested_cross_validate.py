import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss, make_scorer
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class FoldResult:
    """Result from a single outer fold."""
    repeat: int
    fold: int
    y_true: np.ndarray
    y_proba: np.ndarray
    test_indices: np.ndarray
    best_params: Dict[str, Any]


def _create_scorers() -> Dict[str, Any]:
    """
    Create metric scorers for hyperparameter optimization.
    
    Returns
    -------
    dict
        Dictionary mapping metric names to sklearn scorer objects.
    """
    return {
        'brier': make_scorer(
            brier_score_loss,
            greater_is_better=False,
            response_method='predict_proba'
        )
    }


def _calculate_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics for a single fold.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_proba : np.ndarray
        Predicted probabilities for positive class.
    
    Returns
    -------
    dict
        Dictionary with metric names and values.
    """
    return {
        'brier_score': brier_score_loss(y_true, y_proba),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'log_loss': log_loss(y_true, y_proba)
    }


def _train_inner_loop(
    pipeline,
    param_distributions: Dict,
    X_train,
    y_train,
    inner_cv,
    scorer,
    random_state: int = None,
    n_iter: int = 50,
    n_jobs: int = -1
) -> Tuple[Any, Dict]:
    """
    Run hyperparameter optimization on inner CV loop.
    
    Parameters
    ----------
    pipeline : sklearn estimator
        The model to optimize.
    param_distributions : dict
        Hyperparameter distributions for RandomizedSearchCV.
    X_train, y_train : array-like
        Training data for inner loop.
    inner_cv : sklearn CV splitter
        Cross-validation strategy for inner loop.
    scorer : sklearn scorer
        Metric to optimize.
    random_state : int, optional
        Random state for reproducibility.
    n_iter : int, default=50
        Number of parameter combinations to sample.
    n_jobs : int, default=-1
        Parallelization setting.
    
    Returns
    -------
    tuple
        (best_estimator, best_params)
    """
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=inner_cv,
        scoring=scorer,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    
    return search.best_estimator_, search.best_params_


def _split_data(X: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray) -> Tuple:
    """
    Split data into train/test for a given fold.
    
    Parameters
    ----------
    X, y : pd.DataFrame/Series
        Full feature matrix and target.
    train_idx, test_idx : np.ndarray
        Indices for train/test split.
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    return X_train, X_test, y_train, y_test


def _process_outer_fold(
    pipeline,
    param_distributions: Dict,
    X, y,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_number: int,
    outer_folds: int,
    inner_cv,
    scorer: Dict,
    random_state: int = None,
    n_iter: int = 50,
    n_jobs: int = -1
) -> FoldResult:
    """
    Process a single outer fold: optimize hyperparameters on training data,
    predict and evaluate on test data.
    
    Parameters
    ----------
    pipeline : sklearn estimator
        The model to optimize.
    param_distributions : dict
        Hyperparameter distributions for inner loop.
    X, y : array-like
        Full feature matrix and target.
    train_idx, test_idx : np.ndarray
        Indices for train/test split of this fold.
    fold_number : int
        Current fold number (0-indexed across all repeats).
    outer_folds : int
        Total number of folds per repeat.
    inner_cv : sklearn CV splitter
        Cross-validation strategy for inner loop.
    scorer : dict
        Scorer object for optimization.
    random_state : int, optional
        Random state for reproducibility.
    n_iter : int, default=50
        Number of parameter combinations to sample in inner loop.
    n_jobs : int, default=-1
        Parallelization setting.
    
    Returns
    -------
    FoldResult
        Result object containing predictions and metadata for this fold.
    """
    # Split data
    X_train, X_test, y_train, y_test = _split_data(X, y, train_idx, test_idx)
    
    # Optimize hyperparameters on inner loop
    best_model, best_params = _train_inner_loop(
        pipeline,
        param_distributions,
        X_train,
        y_train,
        inner_cv,
        scorer['brier'],
        random_state=random_state,
        n_iter=n_iter,
        n_jobs=n_jobs
    )
    
    # Predict on outer test fold
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate fold metadata
    repeat_num = fold_number // outer_folds
    fold_in_repeat = fold_number % outer_folds
    
    return FoldResult(
        repeat=repeat_num,
        fold=fold_in_repeat,
        y_true=np.asarray(y_test),
        y_proba=y_proba,
        test_indices=test_idx,
        best_params=best_params
    )


def _aggregate_fold_results(
    fold_results: List[FoldResult]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate results from all outer folds into DataFrames.
    
    Parameters
    ----------
    fold_results : list of FoldResult
        Results from each outer fold.
    
    Returns
    -------
    tuple of pd.DataFrame
        (scores_df, predictions_df)
    """
    scores_list = []
    predictions_list = []
    
    for result in fold_results:
        # Calculate metrics for this fold
        metrics = _calculate_metrics(result.y_true, result.y_proba)
        
        # Add metadata to scores
        scores_dict = {
            'repeat': result.repeat,
            'fold': result.fold,
            **metrics
        }
        scores_list.append(scores_dict)
        
        # Add predictions
        for i, (true_label, pred_proba) in enumerate(
            zip(result.y_true, result.y_proba)
        ):
            predictions_list.append({
                'repeat': result.repeat,
                'fold': result.fold,
                'y_true': true_label,
                'y_proba': pred_proba,
                'original_index': result.test_indices[i]
            })
    
    scores_df = pd.DataFrame(scores_list)
    predictions_df = pd.DataFrame(predictions_list)
    
    return scores_df, predictions_df


def _summarize_scores(scores_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate summary statistics from fold-wise scores.
    
    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame with fold-wise metric scores.
    
    Returns
    -------
    dict
        Summary statistics (mean and std) for each metric.
    """
    summary = {}
    
    metrics = ['brier_score', 'roc_auc', 'log_loss']
    for metric in metrics:
        summary[f'{metric}_mean'] = scores_df[metric].mean()
        summary[f'{metric}_std'] = scores_df[metric].std()
    
    return summary


def nested_cv_with_probabilities(
    pipeline,
    param_distributions: Dict,
    X,
    y,
    outer_folds: int = 5,
    outer_repeats: int = 1,
    inner_folds: int = 3,
    inner_repeats: int = 1,
    n_iter: int = 50,
    random_state: int = None,
    n_jobs: int = -1,
    verbose: bool = True
) -> Dict:
    """
    Perform nested cross-validation optimizing Brier score.
    
    This function orchestrates nested CV by:
    1. Creating CV splitters for outer and inner loops
    2. Iterating over outer folds
    3. For each outer fold, optimizing hyperparameters on training data
    4. Evaluating on held-out test data with multiple metrics
    5. Aggregating results and computing summary statistics
    
    Parameters
    ----------
    pipeline : sklearn Pipeline or estimator
        The sklearn pipeline or estimator to optimize.
    param_distributions : dict
        Dictionary with parameter names as keys and distributions
        or lists of parameters to try.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Binary target variable (0s and 1s).
    outer_folds : int, default=5
        Number of folds for outer cross-validation loop.
    outer_repeats : int, default=1
        Number of repeats for outer cross-validation loop.
    inner_folds : int, default=3
        Number of folds for inner loop (hyperparameter tuning).
    inner_repeats : int, default=1
        Number of repeats for inner loop.
    n_iter : int, default=50
        Number of parameter settings sampled in RandomizedSearchCV.
    random_state : int or None, default=None
        Random state for reproducibility.
    n_jobs : int, default=-1
        Number of jobs to run in parallel. -1 uses all processors.
    verbose : bool, default=True
        Whether to print progress information.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'outer_scores': DataFrame with metrics per fold
        - 'predictions': DataFrame with true labels and predicted probabilities
        - 'best_params_per_fold': List of best parameters per fold
        - 'summary': Dict with mean and std of each metric
    """
    
    # Create CV splitters
    outer_cv = RepeatedStratifiedKFold(
        n_splits=outer_folds,
        n_repeats=outer_repeats,
        random_state=random_state
    )
    
    inner_cv = RepeatedStratifiedKFold(
        n_splits=inner_folds,
        n_repeats=inner_repeats,
        random_state=random_state
    )
    
    # Create scorers
    scorers = _create_scorers()
    
    # Process all outer folds
    fold_results = []
    total_folds = outer_folds * outer_repeats
    
    for fold_number, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        result = _process_outer_fold(
            pipeline=pipeline,
            param_distributions=param_distributions,
            X=X,
            y=y,
            train_idx=train_idx,
            test_idx=test_idx,
            fold_number=fold_number - 1,  # 0-indexed
            outer_folds=outer_folds,
            inner_cv=inner_cv,
            scorer=scorers,
            random_state=random_state,
            n_iter=n_iter,
            n_jobs=n_jobs
        )
        
        fold_results.append(result)
        
        if verbose:
            metrics = _calculate_metrics(result.y_true, result.y_proba)
            print(
                f"Fold {fold_number}/{total_folds} complete: "
                f"Brier={metrics['brier_score']:.4f}, "
                f"ROC-AUC={metrics['roc_auc']:.4f}, "
                f"LogLoss={metrics['log_loss']:.4f}"
            )
    
    # Aggregate results
    scores_df, predictions_df = _aggregate_fold_results(fold_results)
    
    # Compute summary
    summary = _summarize_scores(scores_df)
    
    # Extract best params in order
    best_params_list = [result.best_params for result in fold_results]
    
    return {
        'outer_scores': scores_df,
        'predictions': predictions_df,
        'best_params_per_fold': best_params_list,
        'summary': summary
    }