import pandas as pd

def add_missingness_indicators(X: pd.DataFrame) -> pd.DataFrame:
    """
    For each numeric column with missing values, adds a '<col>_missingness'
    column containing 'missing' or 'present' as string dtype.
    Parameters
    ----------
    X : pd.DataFrame
    
    Returns
    -------
    pd.DataFrame
        Dataframe with missingness columns added
    """

    # Select numeric columns
    num_cols = X.select_dtypes(include=["number"]).columns

    # Keep only those with at least one missing value
    cols_with_na = [col for col in num_cols if X[col].isna().any()]

    # Create indicator columns
    for col in cols_with_na:
        new_col = f"{col}_missingness"
        X[new_col] = X[col].isna().map({True: "missing", False: "present"}).astype("string")

    return X
