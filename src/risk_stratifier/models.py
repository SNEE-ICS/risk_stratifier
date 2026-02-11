from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


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
            drop='first',
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
            drop='first',
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
#               XGBoost
#============================================

def get_xgboost_pipeline_and_hyperparameters(
    categorical_features,
    numeric_features,
    n_estimators = 800,
    positive_class_weighting = 1
):
    """
    Provides an sklearn catboost pipeline and an appropriate hyperparameter grid for the pipeline.
    
    The pipeline carries out the following transformations:

    - For numeric features (as indicated by the list provided by the user):
        - NAs are imputed to the median (it would be prudent to preprocess by creating flags of missings)
        - They are then standardized
        - Zero variance features are eliminated.

    - For categorical variables (as indicated by the list provided by the user):
        - NAs are treated as their own category.
        - Zero variance features are eliminated.


    
    Parameters
    ----------
    categorical_features : list of str
        Column names of categorical features.
    numeric_features : list of str
        Column names of numeric features.
    n_estimators: int, default=800
        Determines the number of boosting rounds. A larger value permits greater convergence to the optimal at the expense of compute time. 500 is somewhat high but permits more effective hyperparameterization. If computation is slow, consider lowering.
    positive_class_weighting: int, default=1
        Determines how errors in the positive class are weighted. E.g if set to 100, errors predicting the positive class are 100x more impactful as compared to errors on the negative class. For small imbalanced datasets consider manipulating this. defaul of 1 means equal weighting.
    
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
            drop='first',
            handle_unknown='ignore',
            sparse_output=False
        )),
        ('var_filter', VarianceThreshold(threshold=0))
    ])
    
    # Numeric preprocessing: median imputation + standardization + variance filter
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('var_filter', VarianceThreshold(threshold=0))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, categorical_features),
            ('num', num_transformer, numeric_features)
        ]
    )

    # define xgboost classifier algorithm
    xgb_estimator = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        scale_pos_weight = positive_class_weighting,
        random_state = 42,
        n_jobs = -1

    )
    
    # Full pipeline with logistic ridge regression
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb_estimator)
    ])

    #define parameter grid
    param_grid = {
        'classifier__max_depth': [3, 5, 6, 8],
        'classifier__max_depth': [0.01, 0.05, 0.1, 0.2],
        'classifier__min_child_weight': [1, 5, 10, 15, 20],
        'classifier__max_delta_step': [0, 1, 2, 4]
    }

    
    return pipeline, param_grid