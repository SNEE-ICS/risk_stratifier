import pandas as pd
import pytest

from risk_stratifier.data_validation import validate_binary_y_and_X

def test_validate_binary_y_and_X_happy_path():
    # y: binary integer series, > 100 rows
    y = pd.Series([0, 1] * 60)  # 120 rows

    # X: dataframe, > 100 rows, allowed dtypes
    X = pd.DataFrame(
        {
            "age": range(120),                  # int
            "income": [float(i) for i in range(120)],  # float
            "group": ["A"] * 60 + ["B"] * 60,   # string-like
        }
    )

    # Should not raise
    validate_binary_y_and_X(y, X)