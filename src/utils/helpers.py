"""Helper functions"""

import logging
import sys
from typing import List

from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
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
