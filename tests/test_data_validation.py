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

def test_validate_raises_if_y_not_series():
    y = [0, 1] * 60  # list instead of Series
    X = pd.DataFrame({"x": range(120)})

    with pytest.raises(TypeError, match="Expected a pandas Series for y"):
        validate_binary_y_and_X(y, X)


def test_validate_raises_if_y_not_integer_dtype():
    y = pd.Series([0.0, 1.0] * 60)  # float
    X = pd.DataFrame({"x": range(120)})

    with pytest.raises(ValueError, match="y must have an integer dtype."):
        validate_binary_y_and_X(y, X)


def test_validate_raises_if_y_not_binary():
    y = pd.Series([0, 1, 2] * 40)  # includes 2
    X = pd.DataFrame({"x": range(120)})

    with pytest.raises(ValueError, match="y must contain only 0s and 1s"):
        validate_binary_y_and_X(y, X)

def test_validate_raises_if_y_too_short():
    y = pd.Series([0, 1] * 40)  # 80 rows
    X = pd.DataFrame({"x": range(80)})

    with pytest.raises(ValueError, match="more than 100 rows"):
        validate_binary_y_and_X(y, X)


def test_validate_raises_if_X_too_short():
    y = pd.Series([0, 1] * 60)  # 120 rows
    X = pd.DataFrame({"x": range(80)})

    with pytest.raises(ValueError, match="more than 100 rows"):
        validate_binary_y_and_X(y, X)

def test_validate_raises_if_X_not_dataframe():
    y = pd.Series([0, 1] * 60)
    X = [0] * 120  # not a DataFrame

    with pytest.raises(TypeError, match="Expected a pandas DataFrame for X"):
        validate_binary_y_and_X(y, X)


def test_validate_raises_if_X_has_no_columns():
    y = pd.Series([0, 1] * 60)
    X = pd.DataFrame(index=range(120))  # 0 columns

    with pytest.raises(ValueError, match="must have at least one column"):
        validate_binary_y_and_X(y, X)

def test_validate_raises_if_X_has_unsupported_dtype():
    y = pd.Series([0, 1] * 60)
    X = pd.DataFrame(
        {
            "x": range(120),
            "when": pd.date_range("2024-01-01", periods=120, freq="D"),
        }
    )

    with pytest.raises(ValueError, match="unsupported dtype"):
        validate_binary_y_and_X(y, X)

def test_validate_raises_if_lengths_mismatch():
    y = pd.Series([0, 1] * 60)          # 120
    X = pd.DataFrame({"x": range(130)})  # 130

    with pytest.raises(ValueError, match="Objects must have the same length"):
        validate_binary_y_and_X(y, X)