import pandas as pd
from typing import Union, Tuple

PandasLike = Union[pd.Series, pd.DataFrame]

######## dimensional match ##########

def _dimensions_of_pandas_like_match(obj_1: PandasLike, obj_2: PandasLike)-> Tuple[bool, int, int]:
    """
    Confirms whether two pandas like objects (either series or dataframe are of the same length)
    """

    if not isinstance(obj_1, (pd.Series, pd.DataFrame)):
        raise TypeError("obj_1 must be a pandas Series or DataFrame.")
    if not isinstance(obj_2, (pd.Series, pd.DataFrame)):
        raise TypeError("obj_2 must be a pandas Series or DataFrame.")

    len_1 = len(obj_1)

    len_2 = len(obj_2)

    return len_1 == len_2, len_1, len_2