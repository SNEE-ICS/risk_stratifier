import pandas as pd
import risk_stratifier.utils as utils
from pandas.api.types import is_integer_dtype, is_float_dtype, is_string_dtype
from typing import Any, Union
from colorama import Fore, Style, init
init()

PandasLike = Union[pd.Series, pd.DataFrame]

######### Validate y ##############

def _ensure_y_series_like(obj: Any) -> None:
    """
    Ensure obj is a pandas Series. Otherwise raises a TypeError.
    """
    is_series = isinstance(obj, pd.Series)

    if not (is_series):
        raise TypeError(
            "Expected a pandas Series for y"
        )
    
def _ensure_y_binary_integer_like(obj: pd.Series) -> None:
    """
    Ensure pandas series is an integer with 0s and 1s. Otherwise raises an error
    """

    if not is_integer_dtype(obj):
        raise ValueError("y must have an integer dtype.")
    
    uniques = obj.unique()
    if not set(uniques.tolist()) <= {0, 1}:
        raise ValueError("y must contain only 0s and 1s. 1 will be treated as the positive class.")
    
def _ensure_more_than_100_rows(obj) -> None:
    """
    Raise a ValueError unless `obj` (Series or DataFrame) has more than 100 rows. 
    """
    if isinstance(obj, pd.Series):
        n_rows = obj.shape[0]
    elif isinstance(obj, pd.DataFrame):
        n_rows = obj.shape[0]
    else:
        raise TypeError("Expected a pandas Series or DataFrame.")

    if n_rows <= 100:
        raise ValueError(f"Object must have more than 100 rows, got {n_rows}.")
    # otherwise do nothing, caller just continues

def _ensure_binary_y_is_permissable(obj: Any) -> None:
    """
    takes an object that is to be a binary dependent variable and confirms:
    1. it is a pandas series
    2. It is at least 100 rows long. 
    3. it is an integer type with 0s and 1s only

    If any of these conditions are not met, an error is raised.
    """
    _ensure_y_series_like(obj)
    _ensure_more_than_100_rows(obj)
    _ensure_y_binary_integer_like(obj)

######### Validate X ##############


def _ensure_X_is_df_with_columns(obj: Any) -> None:
    """
    Raise a TypeError unless `obj` is a pandas DataFrame,
    and a ValueError if it has no columns.
    """
    if not isinstance(obj, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame for X.")

    # DataFrame.shape is (n_rows, n_columns)
    if obj.shape[1] < 1:
        raise ValueError("X dataframe must have at least one column.")
    
def _ensure_X_columns_are_proper_type(obj: pd.DataFrame) -> None:
    """
    Raise a ValueError if any column is not int/float/string.
    """

    dtypes = obj.dtypes  # dtype of each column

    for col, dt in dtypes.items():
        # Allow only integer, float, or string-like dtypes
        if not (is_integer_dtype(dt) or is_float_dtype(dt) or is_string_dtype(dt)):
            raise ValueError(
                f"Column '{col}' has unsupported dtype {dt!r}; "
                "only integer, float, or string-like dtypes are allowed."
            )


def _ensure_X_dataframe_is_permissable(obj: Any) -> None:
    """
    Takes an object that is to be a feature matrix and confirms:
    1. It is a pandas dataframe with at least one column
    2. Is at least 100 rows long
    3. Contains only columns that that are of the types sting, integer, float.
    """
    _ensure_X_is_df_with_columns(obj)
    _ensure_more_than_100_rows(obj)
    _ensure_X_columns_are_proper_type(obj)

######## X and y dimensions match ##########

def _y_and_X_length_match(y:pd.Series, X: pd.DataFrame):
    """
    confirms that the y and X provided are the same length
    """
    matches, y_len, X_len = utils._dimensions_of_pandas_like_match(y, X)
    if not matches:
        raise ValueError(f"Objects must have the same length, y is of length {y_len} and X is of length {X_len}.")
    

######## Orchestrate validation of data ###########

def validate_binary_y_and_X(y: pd.Series, X:pd.DataFrame)->None:
    """
    This function confirms the data to be used in modelling is of a permissable format by checking that:

    1. y is a pandas series, is integer type, and consists of 0s and 1s, and exceeds 100 rows
    2. X is a pandas dataframe, col types are constricted to float, integer, and strings, and exceeds 100 rows
    3. X and y are the same length.

    If any checks fail, an error is raised.
    """
    _ensure_binary_y_is_permissable(y)
    _ensure_X_dataframe_is_permissable(X)
    _y_and_X_length_match(y, X)


    print(Fore.GREEN + "X and y data provided for modelling are permissable." + Style.RESET_ALL)