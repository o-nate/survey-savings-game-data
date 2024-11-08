"""Helper functions"""

import logging
from pathlib import Path
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


def export_plot(
    exported_file_path: Path, file_name: str, export_all_plots: bool
) -> None:
    """Export plot to results folder in png format

    Args:
        file_name (str): name of plot image file
    """
    file_path = exported_file_path / file_name
    if export_all_plots:
        plt.savefig(file_path, bbox_inches="tight")
    elif input(f"Export {file_name}? (y) ").lower() == "y":
        plt.savefig(file_path, bbox_inches="tight")
