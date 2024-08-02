"""Helper functions"""

import logging
import sys
from typing import List

from functools import reduce
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from src.utils.logging_helpers import set_external_module_log_levels

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def combine_series(dataframes: List[pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Merge dataframes from a list

    Args:
        dataframes List[pd.DataFrame]: List of dataframes to merge

    Kwargs:
        on List[str]: Column(s) to merge on
        how {str}: 'left', 'right', 'inner', 'outer'

    Returns:
        pd.DataFrame: Combined dataframe
    """
    return reduce(lambda left, right: pd.merge(left, right, **kwargs), dataframes)


def create_pearson_correlation_matrix(
    data: pd.DataFrame,
    p_values: List[float],
    include_stars: bool = True,
    display: bool = False,
    decimal_places: int = 2,
) -> pd.DataFrame:
    """Calculate pearson correlation and p-values and add asterisks
    to relevant values in table

    Args:
        data (pd.DataFrame): Data to correlate
        p_values (List[float]): p value thresholds for stars
        include_stars (bool, optional): Include star for p values. Defaults to True.
        display (bool, optional): Display p values for each correlation. Defaults to False.
        decimal_places (int, optional): Decimal places to include. Defaults to 2.

    Returns:
        pd.DataFrame: _description_
    """

    rho = data.corr()
    pval = data.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
    cols = data.columns.to_list()
    if display:
        print(f"P-values benchmarks: {p_values}")
        for c in cols:
            print(c)
            print(f"{c} p-values: \n{pval[c]}")
    if include_stars:
        p = pval.applymap(lambda x: "".join(["*" for t in p_values if x <= t]))
        return rho.round(decimal_places).astype(str) + p
    return rho.round(decimal_places)
