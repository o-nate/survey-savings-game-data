"""Process survey responses of perceived and expected inflation"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from preprocess import final_df_dict
from src.helpers import INF_430, INF_1012

logging.basicConfig(level="INFO")


def create_survey_df() -> pd.DataFrame:
    """Creates a merged dataframe with the quantitative estimates (perceptions
    and expectations) and actual inflation in a single dataframe"""

    # * Combine quantitative estimates into one dataframe
    df1 = final_df_dict["inf_expectation"].copy()
    df2 = final_df_dict["inf_estimate"].copy()
    df3 = df1.merge(df2, how="left")
    logging.debug(df3.columns.to_list())
    df_survey = df3.melt(
        id_vars=[
            "participant.code",
            "participant.label",
            "participant.inflation",
            "participant.round",
        ],
        value_vars=[c for c in df3.columns if "inf_" in c],
        var_name="Measure",
        value_name="Estimate",
    )

    # * Extract month number
    df_survey["Month"] = df_survey["Measure"].str.extract("(\d+)")
    ## Convert to int
    df_survey = df_survey.apply(pd.to_numeric, errors="ignore")

    # * Rename measures
    df_survey["Measure"] = df_survey["Measure"].str.split("player.").str[1]
    df_survey["Measure"].replace(
        ["inf_estimate", "inf_expectation"],
        ["Perception", "Expectation"],
        inplace=True,
    )

    # * Add actual inflation
    inf_dict = {
        "participant.inflation": [
            430 for m in range(40)
        ],  # + [1012 for m in range(19)],
        "participant.round": [1 for m in range(10)]
        + [1 for m in range(10)]
        + [2 for m in range(10)]
        + [2 for m in range(10)],
        "Month": [(m + 1) * 12 for m in range(10)]
        + [m * 12 for m in range(10)]
        + [(m + 1) * 12 for m in range(10)]
        + [m * 12 for m in range(10)],
        "Measure": ["Actual" for m in range(10)]
        + ["Upcoming" for m in range(10)]
        + ["Actual" for m in range(10)]
        + ["Upcoming" for m in range(10)],
        "Estimate": INF_430 + INF_430 + INF_430 + INF_430,  # + INF_1012 + INF_1012[1:],
    }
    ## Convert to dataframe
    df_inf = pd.DataFrame(inf_dict)
    ## Merge with survey responses
    df_survey = pd.concat([df_survey, df_inf], ignore_index=True)
    df_survey["participant.inflation"].replace(
        [430, 1012],
        ["4x30", "10x12"],
        inplace=True,
    )

    return df_survey


def main() -> None:
    """Run script"""
    data = create_survey_df()

    # * Plot
    g = sns.relplot(
        data=data,
        x="Month",
        y="Estimate",
        errorbar=None,
        hue="Measure",
        style="Measure",
        kind="line",
        col="participant.round",
    )

    ## Adjust titles
    (
        g.set_axis_labels("Month", "Inflation rate (%)")
        .set_titles("Savings Game round: {col_name}")
        .tight_layout(w_pad=0.5)
    )

    plt.show()


if __name__ == "__main__":
    main()
