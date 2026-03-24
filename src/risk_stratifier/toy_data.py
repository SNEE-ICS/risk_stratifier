import pandas as pd
import numpy as np


def make_toy_X_y(n_rows=1000, pos_rate=0.1, random_state=42):
    """
    Returns:
        X: pandas.DataFrame with 1000 rows and:
           - int_col: nullable integer with NAs
           - float_col: float with NAs, positively correlated with target
           - str_col: string with NAs

        y: pandas.Series of binary labels (0/1), ~10% positives (1)
    """
    rng = np.random.default_rng(random_state)

    # Target: mostly zeros with ~10% ones
    y = pd.Series((rng.random(n_rows) < pos_rate).astype(int), name="target")

    # Integer column with NAs (use nullable Int64 dtype)
    int_col = rng.integers(0, 100, size=n_rows).astype("float")
    int_na_idx = rng.choice(n_rows, size=int(n_rows * 0.1), replace=False)
    int_col[int_na_idx] = np.nan
    int_col = pd.Series(int_col, name="int_col").astype("Int64")

    # Float column with NAs — mean shifts by target (positives ~ N(2, 1))
    float_col = rng.normal(loc=0.0, scale=1.0, size=n_rows)
    float_col += y.values * 2.0  # add signal: positives shifted up by 2
    float_na_idx = rng.choice(n_rows, size=int(n_rows * 0.1), replace=False)
    float_col[float_na_idx] = np.nan
    float_col = pd.Series(float_col, name="float_col")

    # String column with NAs
    categories = np.array(["A", "B", "C", "D"])
    str_col = rng.choice(categories, size=n_rows)
    str_na_idx = rng.choice(n_rows, size=int(n_rows * 0.1), replace=False)
    str_col[str_na_idx] = None
    str_col = pd.Series(str_col, name="str_col", dtype="string")

    X = pd.DataFrame(
        {
            "int_col": int_col,
            "float_col": float_col,
            "str_col": str_col,
        }
    )

    return X, y
