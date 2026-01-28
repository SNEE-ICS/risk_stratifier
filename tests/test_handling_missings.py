import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from risk_stratifier.handling_missings import add_missingness_indicators
# ^ change this import if the function lives in a different module


class TestAddMissingnessIndicators:
    def test_no_numeric_missing_adds_no_indicator_columns(self):
        """No numeric column has missing values → no *_missingness columns created."""
        X = pd.DataFrame(
            {
                "age": [10, 20, 30],           # numeric, no NA
                "score": [0.1, 0.2, 0.3],      # numeric, no NA
                "group": ["a", "b", "c"],      # non‑numeric
            }
        )

        result = add_missingness_indicators(X.copy())

        # No new *_missingness columns
        assert not any(col.endswith("_missingness") for col in result.columns)
        # Original data unchanged
        pd.testing.assert_frame_equal(result[["age", "score", "group"]], X)

    def test_single_numeric_column_with_missing(self):
        """Adds one '<col>_missingness' column with correct 'missing'/'present' values."""
        X = pd.DataFrame(
            {
                "age": [10.0, np.nan, 30.0],
                "group": ["a", "b", "c"],
            }
        )

        result = add_missingness_indicators(X.copy())

        assert "age_missingness" in result.columns

        expected = pd.Series(
            ["present", "missing", "present"],
            name="age_missingness",
            dtype="string",
        )
        pd.testing.assert_series_equal(result["age_missingness"], expected)

    def test_multiple_numeric_columns_with_missing(self):
        """Creates indicators for each numeric column that has at least one missing."""
        X = pd.DataFrame(
            {
                "age": [10.0, np.nan, 30.0],
                "score": [0.1, 0.2, np.nan],
                "group": ["a", "b", "c"],
            }
        )

        result = add_missingness_indicators(X.copy())

        assert "age_missingness" in result.columns
        assert "score_missingness" in result.columns

        expected_age = pd.Series(
            ["present", "missing", "present"],
            name="age_missingness",
            dtype="string",
        )
        expected_score = pd.Series(
            ["present", "present", "missing"],
            name="score_missingness",
            dtype="string",
        )

        pd.testing.assert_series_equal(result["age_missingness"], expected_age)
        pd.testing.assert_series_equal(result["score_missingness"], expected_score)

    def test_non_numeric_columns_ignored_even_if_missing(self):
        """Missing values in non‑numeric columns should not generate indicators."""
        X = pd.DataFrame(
            {
                "age": [10.0, 20.0, 30.0],          # numeric, no NA
                "category": ["a", None, "c"],       # object with NA
            }
        )

        result = add_missingness_indicators(X.copy())

        # Only original columns present; no *_missingness columns
        assert list(result.columns) == ["age", "category"]
        pd.testing.assert_frame_equal(result, X)

    def test_indicator_columns_are_string_dtype(self):
        """Indicator columns must use pandas StringDtype, not plain object."""
        X = pd.DataFrame({"age": [10.0, np.nan]})

        result = add_missingness_indicators(X.copy())

        col = result["age_missingness"]
        assert str(col.dtype) == "string"
        assert set(col.unique()) == {"present", "missing"}
