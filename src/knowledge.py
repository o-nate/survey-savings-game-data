"""Measures of financial literacy, numeracy, and abilities to calculate compound interest"""

import logging
import sys

import numpy as np
import pandas as pd

from src.preprocess import final_df_dict
from src.utils import helpers
from src.utils.logging_helpers import set_external_module_log_levels

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def count_correct_responses(data: pd.DataFrame, knowledge_measure: str) -> pd.Series:

    if knowledge_measure == "financial_literacy":
        _criteria = [
            data["Finance.1.player.finK_1"].eq(1)
            & data["Finance.1.player.finK_2"].eq(-1)
            & data["Finance.1.player.finK_9"].eq(1)
        ]
    if knowledge_measure == "numeracy":
        _criteria = [
            data["Numeracy.1.player.num_2b"].eq(20)
            | data["Numeracy.1.player.num_3"].eq(50)
        ]
    if knowledge_measure == "compound":
        _criteria = [
            data["Inflation.1.player.infCI_1"].eq(1100)
            & data["Inflation.1.player.infCI_2"].eq(2)
            & data["Inflation.1.player.infCI_3"].eq(2)
            & data["Inflation.1.player.infCI_4"].eq(32000)
        ]
    choices = [1]
    return np.select(_criteria, choices, default=0)


def create_knowledge_dataframe() -> pd.DataFrame:
    dataframes = []
    for i, j in zip(
        ["Finance", "Numeracy", "Inflation"],
        ["financial_literacy", "numeracy", "compound"],
    ):
        _df = final_df_dict[i].copy()
        _df[j] = count_correct_responses(_df, j)
        dataframes.append(_df[["participant.label", j]])
    return helpers.combine_series(dataframes, how="left", on="participant.label")


def main() -> None:
    """Run script"""
    df = create_knowledge_dataframe()
    print(df.head())
    logger.debug(df[df["participant.label"] == "9xHTKNJ"])


if __name__ == "__main__":
    main()
