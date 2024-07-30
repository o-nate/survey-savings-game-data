"""Process survey responses of perceived and expected inflation"""

import logging
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocess import final_df_dict
from src.utils.constants import INFLATION_DICT
from src.utils.logging_helpers import set_external_module_log_levels
from src.utils.helpers import combine_series

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def calculate_estimate_bias(
    df: pd.DataFrame, estimate_col: str, real_inflation_col: str
) -> pd.Series:
    """Takes the difference bewteen real and estimate inflation (real minus estimate)

    Args:
        df (pd.DataFrame): DataFrame with estimate and real inflation columns
        estimate_col (str): column name of estimates
        real_inflation_col (str): column name of real inflation

    Returns:
        pd.Series: returns with column for bias
    """
    return df[real_inflation_col] - df[estimate_col]


def pivot_inflation_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Pivots inflation and inflation estimate dataframes to calculate inflation measures

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_inf = pd.DataFrame(INFLATION_DICT)
    df_inf = pd.pivot_table(
        df_inf,
        index=["participant.inflation", "participant.round", "Month"],
        columns="Measure",
    )
    df_inf.columns = [
        "_".join(str(i) for i in a) for a in df_inf.columns.to_flat_index()
    ]
    df_inf.reset_index(inplace=True)
    df_pivot = pd.pivot_table(
        data=df,
        index=[
            "participant.code",
            "participant.label",
            "date",
            "participant.inflation",
            "participant.round",
            "Month",
        ],
        columns="Measure",
        fill_value=0,
    )
    # Remove multiindex
    df_pivot.columns = [
        "_".join(str(i) for i in a) for a in df_pivot.columns.to_flat_index()
    ]
    df_pivot.reset_index(inplace=True)
    df_pivot = df_pivot.merge(
        df_inf[[c for c in df_inf.columns if "inflation" not in c]],
        how="left",
        on=["Month", "participant.round"],
    )
    df_pivot.rename(
        columns={
            col: col.removeprefix("Estimate_")
            for col in df_pivot.columns
            if "Estimate_" in col
        },
        inplace=True,
    )
    return df_pivot


def create_survey_df() -> pd.DataFrame:
    """Creates a merged dataframe with the estimates (perceptions
    and expectations) and actual inflation in a single dataframe"""

    # * Combine estimates into one dataframe
    df1 = final_df_dict["inf_expectation"].copy()
    df2 = final_df_dict["inf_estimate"].copy()
    df3 = final_df_dict["qualitative_expectation"].copy()
    df4 = final_df_dict["qualitative_estimate"].copy()
    dfs = [df1, df2, df3, df4]
    df5 = combine_series(
        dfs,
        on=[
            "participant.code",
            "participant.label",
            "participant.inflation",
            "treatment",
            "date",
            "participant.round",
        ],
        how="left",
    )
    logger.debug(df5.columns.to_list())
    df_survey = df5.melt(
        id_vars=[
            "participant.code",
            "participant.label",
            "date",
            "participant.inflation",
            "participant.round",
        ],
        value_vars=[c for c in df5.columns if ("inf_" in c) or ("qualitative_" in c)],
        var_name="Measure",
        value_name="Estimate",
    )

    # * Extract month number
    df_survey["Month"] = df_survey["Measure"].str.extract(r"(\d+)")

    # * Convert columns to int, except participant.time_started_utc
    cols_to_convert = [c for c in df_survey.columns if c != "date"]
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
    logger.debug("INFLATION_DICT %s", INFLATION_DICT)
    ## Convert to dataframe
    df_inf = pd.DataFrame(INFLATION_DICT)
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
    measures = pivot_inflation_measures(data)
    logger.debug(measures.shape)

    print("participants included: ", data["participant.label"].nunique())

    # # * Plot qualitative responses
    qual_responses = ["Qual Perception", "Qual Expectation"]
    print(data[data["Measure"].isin(qual_responses)].head())

    data2 = data.copy()
    data2["Month"] = data2["Month"].astype(str)

    sns.catplot(
        data=data2[data2["Measure"].isin(qual_responses)],
        x="Estimate",
        y="Month",
        col="Measure",
        hue="participant.round",
        kind="violin",
        split=True,
        palette="bright",
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
        col="participant.round",
    )

    ## Adjust titles
    (
        g.set_axis_labels("Month", "Inflation rate (%)")
        # .set_titles("Savings Game round: {col_name}")
        .tight_layout(w_pad=0.5)
    )

    # sns.kdeplot(data=data[data["Measure"].isin(estimates)], x="Estimate", hue="Measure")
    # plt.yscale("log")
    # plt.xscale("log")

    plt.show()


if __name__ == "__main__":
    main()
