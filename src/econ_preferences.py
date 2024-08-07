"""Module to process economic preferences data"""

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from src import calc_opp_costs, discontinuity, process_survey

from src.preprocess import final_df_dict
from src.utils import constants
from src.utils.logging_helpers import set_external_module_log_levels

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def count_preference_choices(data: pd.DataFrame, econ_preference: str) -> pd.Series:
    """Count choices for economic preference. For loss aversion, count the number
    of decisions to toss the coin. For risk aversion, count the number of safe
    choices. For time preferences, count the number of smaller-sooner choices.

    Args:
        data (pd.DataFrame): Data
        econ_preference (str): Economic preference measure ('lossAversion',
        'riskAversion', 'timePreference')

    Returns:
        pd.Series: Number of choices
    """
    column_selector = constants.CHOICES[econ_preference]
    cols = [c for c in data.columns if column_selector in c]
    return data[cols].sum(axis=1)


def count_switches(data: pd.DataFrame, econ_preference: str) -> pd.Series:
    """Count changes in economic preferences

    Args:
        data (pd.DataFrame): Data
        econ_preference (str): Economic preference measure ('lossAversion',
        'riskAversion', 'timePreference')

    Returns:
        pd.Series: Number of switches
    """
    column_selector = constants.CHOICES[econ_preference]
    cols = [c for c in data.columns if column_selector in c]
    return (data[cols] != data[cols].shift(axis=1)).sum(axis=1) - 1


def main() -> None:
    """Run script"""
    df = final_df_dict["lossAversion"].copy()
    df["n_switches"] = count_switches(df, "lossAversion")
    print(df.head())
    logger.debug(
        df[df["participant.code"] == "ub8goyln"][
            [c for c in df.columns if constants.CHOICES["lossAversion"] in c]
            + ["n_switches"]
        ]
    )


if __name__ == "__main__":
    main()
