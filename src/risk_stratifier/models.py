from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression


#============================================
#           logistic ridge regression
#============================================

def get_logistic_ridge_pipeline_and_hyperparameters(
    categorical_features,
    numeric_features,
    solver = 'saga',
    max_iter = 5000
):
    """
    Provides an sklearn logistic ridge regression pipeline and an appropriate hyperparameter grid for the pipeline.
    
    The pipeline carries out the following transformations:

    - For numeric features (as indicated by the list provided by the user):
        - NAs are imputed to the median (it would be prudent to preprocess by creating flags of missings)
        - They are then standardized
        - Zero variance features are eliminated.

    - For categorical variables (as indicated by the list provided by the user):
        - NAs are treated as their own category
        - categories are one hot encoded dropping the first category to avoid perfect linear dependence.
        - binary features are standardized.
        - Zero variance features are eliminated.


    
    Parameters
    ----------
    categorical_features : list of str
        Column names of categorical features.
    numeric_features : list of str
        Column names of numeric features.
    cat_variance_threshold : float, default=0.0
        Variance threshold for filtering one-hot encoded categorical features.
    num_variance_threshold : float, default=0.0
        Variance threshold for filtering numeric features.
    solver: string, default=saga
        Optimal solver when samples are > 100,000 and samples >> features. for smaller datasets switch to 'lbfgs'.
    max_iter: int, default=5000
        Controls the number of times gradient descent is repeated to be below some level of tolerance. 5000 (slightly high) reflects the potentially large size of the dataset and the imbalance that is common in healthcare settings.
    
    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline ready for training.
    dict
        Parameter dictionary that is associated with this workflow.
    """
    
    # Categorical preprocessing: OHE with nulls as separate category + variance filter
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(
            drop='first',  # Note: 'most_frequent' not available in sklearn
            handle_unknown='ignore',
            sparse_output=False
        )),
        ('scaler', StandardScaler()),
        ('var_filter', VarianceThreshold(threshold=0))
    ])
    
    # Numeric preprocessing: median imputation + standardization + variance filter
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('var_filter', VarianceThreshold(threshold=0))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, categorical_features),
            ('num', num_transformer, numeric_features)
        ]
    )
    
    # Full pipeline with logistic ridge regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=max_iter, solver=solver, l1_ratio=0.0))
    ])

    #define parameter grid
    param_grid = {
        # Regularization strength (inverse of regularization, log-scale)
        'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    }

    
    return pipeline, param_grid

#============================================
#           logistic lasso regression
#============================================

def get_logistic_lasso_pipeline_and_hyperparameters(
    categorical_features,
    numeric_features,
    solver = 'saga',
    max_iter = 5000
):
    """
    Provides an sklearn logistic lasso regression pipeline and an appropriate hyperparameter grid for the pipeline.
    
    The pipeline carries out the following transformations:

    - For numeric features (as indicated by the list provided by the user):
        - NAs are imputed to the median (it would be prudent to preprocess by creating flags of missings)
        - They are then standardized
        - Zero variance features are eliminated.

    - For categorical variables (as indicated by the list provided by the user):
        - NAs are treated as their own category
        - categories are one hot encoded dropping the first category to avoid perfect linear dependence.
        - binary features are standardized.
        - Zero variance features are eliminated.


    
    Parameters
    ----------
    categorical_features : list of str
        Column names of categorical features.
    numeric_features : list of str
        Column names of numeric features.
    cat_variance_threshold : float, default=0.0
        Variance threshold for filtering one-hot encoded categorical features.
    num_variance_threshold : float, default=0.0
        Variance threshold for filtering numeric features.
    solver: string, default=saga
        Optimal solver when samples are > 100,000 and samples >> features. for smaller datasets switch to 'lbfgs'.
    max_iter: int, default=5000
        Controls the number of times gradient descent is repeated to be below some level of tolerance. 5000 (slightly high) reflects the potentially large size of the dataset and the imbalance that is common in healthcare settings.
    
    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline ready for training.
    dict
        Parameter dictionary that is associated with this workflow.
    """
    
    # Categorical preprocessing: OHE with nulls as separate category + variance filter
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(
            drop='first',  # Note: 'most_frequent' not available in sklearn
            handle_unknown='ignore',
            sparse_output=False
        )),
        ('scaler', StandardScaler()),
        ('var_filter', VarianceThreshold(threshold=0))
    ])
    
    # Numeric preprocessing: median imputation + standardization + variance filter
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('var_filter', VarianceThreshold(threshold=0))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, categorical_features),
            ('num', num_transformer, numeric_features)
        ]
    )
    
    # Full pipeline with logistic ridge regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=max_iter, solver=solver, l1_ratio=1.0))
    ])

    #define parameter grid
    param_grid = {
        # Regularization strength (inverse of regularization, log-scale)
        'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    }

    
    return pipeline, param_grid



#============================================
#           catboost
#============================================

def get_catboost_pipeline_and_hyperparameters(
    categorical_features,
    numeric_features,
    solver = 'saga',
    max_iter = 5000
):
    """
    Provides an sklearn catboost pipeline and an appropriate hyperparameter grid for the pipeline.
    
    The pipeline carries out the following transformations:

    - For numeric features (as indicated by the list provided by the user):
        - Zero variance features are eliminated.

    - For categorical variables (as indicated by the list provided by the user):
        - NAs are treated as their own category
        - categories are one hot encoded dropping the first category to avoid perfect linear dependence.
        - binary features are standardized.
        - Zero variance features are eliminated.


    
    Parameters
    ----------
    categorical_features : list of str
        Column names of categorical features.
    numeric_features : list of str
        Column names of numeric features.
    cat_variance_threshold : float, default=0.0
        Variance threshold for filtering one-hot encoded categorical features.
    num_variance_threshold : float, default=0.0
        Variance threshold for filtering numeric features.
    solver: string, default=saga
        Optimal solver when samples are > 100,000 and samples >> features. for smaller datasets switch to 'lbfgs'.
    max_iter: int, default=5000
        Controls the number of times gradient descent is repeated to be below some level of tolerance. 5000 (slightly high) reflects the potentially large size of the dataset and the imbalance that is common in healthcare settings.
    
    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline ready for training.
    dict
        Parameter dictionary that is associated with this workflow.
    """
    
    # Categorical preprocessing: OHE with nulls as separate category + variance filter
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(
            drop='first',  # Note: 'most_frequent' not available in sklearn
            handle_unknown='ignore',
            sparse_output=False
        )),
        ('scaler', StandardScaler()),
        ('var_filter', VarianceThreshold(threshold=0))
    ])
    
    # Numeric preprocessing: median imputation + standardization + variance filter
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('var_filter', VarianceThreshold(threshold=0))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, categorical_features),
            ('num', num_transformer, numeric_features)
        ]
    )
    
    # Full pipeline with logistic ridge regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=max_iter, solver=solver, l1_ratio=1.0))
    ])

    #define parameter grid
    param_grid = {
        # Regularization strength (inverse of regularization, log-scale)
        'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    }

    
    return pipeline, param_grid