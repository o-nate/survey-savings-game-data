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


def count_wisconsin_correct(data: pd.DataFrame) -> pd.Series:
    """Produce series of total correct choices per subject

    Args:
        data (pd.DataFrame): Wisconsin Card Sorting Task data

    Returns:
        pd.Series: Number of total correct choices per subject
    """
    return data["wisconsin.1.player.num_correct"]


def count_wisconsin_errors(
    data: pd.DataFrame, error_type: str, num_trials: int = 30
) -> pd.Series:
    """Count number of perseverative or set-loss erros from Wisconsin Card Sorting Task.
    Perseverative errors are failures to adapt decisions to negative feedback. Set-loss
    errors are failures to maintain a decision, given positive feedback.

    Args:
        data (pd.DataFrame): Wisconsin Card Sorting Task data
        error_type (str): `perseverative` or `set-loss`
        num_trials (int, optional): Total number of decisions. Defaults to 30.

    Raises:
        ValueError: When incorrect `error_type` defined.

    Returns:
        pd.Series: Series of total error for defined `error_type` per subject
    """
    _data = data.copy()
    ## Start at trial 2 since we can only have errors after 1st trial
    for n in range(2, 1 + num_trials):
        if error_type == "perseverative":
            criteria = [
                _data[f"correct_{n-1}"].eq(False)
                & (_data[f"guess_{n-1}"] == _data[f"guess_{n}"])
            ]
        elif error_type == "set-loss":
            criteria = [
                _data[f"correct_{n-1}"].eq(True)
                & (_data[f"guess_{n-1}"] != _data[f"guess_{n}"])
            ]
        else:
            raise ValueError(
                "Please, choose either `perseverative` or `set-loss` error type."
            )
        choices = [1]
        if n == 2:
            _data["n_error"] = np.select(criteria, choices, default=0)
        else:
            _data["n_error"] += np.select(criteria, choices, default=0)
    return _data["n_error"]


def main() -> None:
    """Run script"""
    df = final_df_dict["wisconsin"].copy()
    logger.debug(df.shape)


if __name__ == "__main__":
    main()
