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
        'riskAversion', 'timePreferences')

    Returns:
        pd.Series: Number of choices
    """
    column_selector = constants.CHOICES[econ_preference]
    cols = [c for c in data.columns if column_selector in c]
    if econ_preference == "timePreferences":
        return (data[cols] == 1).sum(axis=1)
    return data[cols].sum(axis=1)


def count_switches(data: pd.DataFrame, econ_preference: str) -> pd.Series:
    """Count changes in loss aversion and risk preferences (number greater than
    1 suggest inconsistent preferences)

    Args:
        data (pd.DataFrame): Data
        econ_preference (str): Economic preference measure ('lossAversion',
        'riskPreferences')

    Returns:
        pd.Series: Number of switches
    """
    column_selector = constants.CHOICES[econ_preference]
    cols = [c for c in data.columns if column_selector in c]
    return (data[cols] != data[cols].shift(axis=1)).sum(axis=1) - 1


def count_time_preference_switches(data: pd.DataFrame, rounds: int = 2) -> pd.Series:
    """Count changes in time preferences (number greater than the number or rounds
    suggest inconsistent preferences)

    Args:
        data (pd.DataFrame): Time preference data
        rounds (int, optional): Number of sets of smaller-sooner vs. larger-later choices.
        Defaults to 2.

    Returns:
        pd.Series: Total switches per participant over the course of all rounds
    """
    _data = data.copy()
    for r in range(1, 1 + rounds):
        cols = [c for c in _data.columns if f"timePreferences.{r}.player.q" in c]
        ## Convert to booleans
        _data[cols] = np.where(_data[cols] == 1, True, False)
        _data[f"count_{r}"] = (_data[cols] != _data[cols].shift(axis=1)).sum(axis=1) - 1
    return _data[[c for c in _data.columns if "count_" in c]].sum(axis=1)


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
