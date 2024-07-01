"""Process survey responses of perceived and expected inflation"""

import logging

from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from preprocess import final_df_dict
from src.helpers import disable_module_debug_log, INF_430

# * Logging settings
logger = logging.getLogger(__name__)
disable_module_debug_log("warning")
logger.setLevel(logging.DEBUG)


def create_survey_df() -> pd.DataFrame:
    """Creates a merged dataframe with the estimates (perceptions
    and expectations) and actual inflation in a single dataframe"""

    # * Combine estimates into one dataframe
    df1 = final_df_dict["inf_expectation"].copy()
    df2 = final_df_dict["inf_estimate"].copy()
    df3 = final_df_dict["qualitative_expectation"].copy()
    df4 = final_df_dict["qualitative_estimate"].copy()
    dfs = [df1, df2, df3, df4]
    df5 = reduce(
        lambda df_left, df_right: pd.merge(df_left, df_right, how="left"),
        dfs,
    )
    logger.debug(df5.columns.to_list())
    df_survey = df5.melt(
        id_vars=[
            "participant.code",
            "participant.label",
            "participant.time_started_utc",
            "participant.inflation",
            "participant.round",
        ],
        value_vars=[c for c in df5.columns if ("inf_" in c) or ("qualitative_" in c)],
        var_name="Measure",
        value_name="Estimate",
    )

    # * Extract month number
    df_survey["Month"] = df_survey["Measure"].str.extract("(\d+)")

    # * Convert columns to int, except participant.time_started_utc
    cols_to_convert = [
        c for c in df_survey.columns if c != "participant.time_started_utc"
    ]
    df_survey[cols_to_convert] = df_survey[cols_to_convert].apply(
        pd.to_numeric, errors="ignore"
    )

    # * Rename measures
    df_survey["Measure"] = df_survey["Measure"].str.split("player.").str[1]
    df_survey["Measure"].replace(
        [
            "inf_estimate",
            "inf_expectation",
            "qualitative_estimate",
            "qualitative_expectation",
        ],
        [
            "Quant Perception",
            "Quant Expectation",
            "Qual Perception",
            "Qual Expectation",
        ],
        inplace=True,
    )
    logger.debug(df_survey.info())
    print(df_survey.head())

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

    # * Convert datetime to date
    data["date"] = data["participant.time_started_utc"].dt.normalize()

    # # ! Filter for just 20-06-2024
    # data = data[data["date"] >= "2024-06-20"]
    print(data["participant.label"].nunique())

    # # * Plot qualitative responses
    qual_responses = ["Qual Perception", "Qual Expectation"]
    print(data[data["Measure"].isin(qual_responses)].head())
    # hue = (
    #     data[data["Measure"].isin(qual_responses)]["participant.round"].astype(str)
    #     + ", "
    #     + data[data["Measure"].isin(qual_responses)]["Measure"].astype(str)
    # )
    # logger.debug("%s vs %s", len(hue), len(data[data["Measure"].isin(qual_responses)]))
    h = sns.FacetGrid(
        data=data[data["Measure"].isin(qual_responses)],
        col="Month",
        height=2.5,
        col_wrap=3,
        hue="Measure",
    )
    h.map(
        sns.histplot,
        "Estimate",
        multiple="dodge",
        shrink=0.8,
    )

    # * Plot estimates over time
    estimates = ["Quant Perception", "Quant Expectation", "Actual", "Upcoming"]
    g = sns.relplot(
        data=data[data["Measure"].isin(estimates)],
        x="Month",
        y="Estimate",
        errorbar=None,
        hue="Measure",
        style="Measure",
        kind="line",
        row="participant.round",
        col="date",
    )

    ## Adjust titles
    (
        g.set_axis_labels("Month", "Inflation rate (%)")
        .set_titles("Savings Game round: {col_name}")
        .tight_layout(w_pad=0.5)
    )

    # sns.kdeplot(data=data[data["Measure"].isin(estimates)], x="Estimate", hue="Measure")
    # plt.yscale("log")
    # plt.xscale("log")

    plt.show()


if __name__ == "__main__":
    main()
