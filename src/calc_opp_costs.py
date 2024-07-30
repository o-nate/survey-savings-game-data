"""
Calculate the opportunities costs incurred by each participant for each type 
of error (early, late, and excess purchases). When run directly as a script 
(`python calc_opp_costs.py`), generate a csv in `data/preprocessed`
"""

from decimal import Decimal, ROUND_UP
import logging
import math
from pathlib import Path
import sys
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

from src.preprocess import final_df_dict
from src.utils.logging_helpers import set_external_module_log_levels

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# * Declare inflation csv file
inf_file = Path(__file__).parents[1] / "data" / "animal_spirits.csv"

# * Declare optimal decisions file
optimal_file = Path(__file__).parents[1] / "data" / "optimal_purchases.csv"

# ! For exporting (if script run directly)
# * Declare name of output file
FINAL_FILE_PREFIX = "opp_cost"

# * Declare directory of output file
final_dir = Path(__file__).parents[1] / "data" / "preprocessed"


def categorize_opp_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate opportunity costs for each possible mistake
    (early, late and excesss purchases)
    """
    df["excess"] = df["s1"] - df["sreal"]
    df["early"] = df["s2"] - df["s1"]
    df["late"] = df["soptimal"] - df["s2"]

    return df


def savings_calc(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Function to calculate savings after month t=1
    After 1st month, Si_t = Si_(t-1) * (1 + r) + y - p_t * qi_t"""
    q_col = f"q{strategy}" if strategy != "real" else "decision"
    for i in tqdm(range(len(df)), desc=f"S{strategy}"):
        if df.month.iat[i] > 1:
            ## Must round Si_(t-1) and p_t to nearest 0.01 because of rounding in oTree
            si_t = (
                round_price(df[f"s{strategy}"].iat[i - 1]) * (1 + INTEREST)
                + WAGE
                - df.newPrice.iat[i] * df[q_col].iat[i]
            )
            ## Must subsequently round si_t to nearest 0.01 too
            df[f"s{strategy}"].iat[i] = round_price(si_t)
        else:
            pass
    return df[f"s{strategy}"]


def strategy_savings_calc(strategy_list: List[str], df: pd.DataFrame) -> pd.DataFrame:
    """Apply savings_calc for each strategy"""
    for strat in strategy_list:
        q_col = f"q{strat}" if strat != "real" else "decision"
        df[f"s{strat}"] = np.nan
        ## For 1st month, Si_1 = Si_(t-1) * (1+r) + y - p_1*qi_1
        df[f"s{strat}"][df["month"] == 1] = (
            ENDOWMENT + WAGE - df["newPrice"] * df[q_col]
        )
        df[f"s{strat}"] = savings_calc(df, strat)
    return df


def truncate_decimals(target_allocation, two_decimal_places) -> float:
    """Functions for rounding values"""
    decimal_exponent = 10.0**two_decimal_places
    return math.trunc(decimal_exponent * target_allocation) / decimal_exponent


def round_price(number, decimal_precision=2):
    """Round up or down depending on trailing decimal value"""
    if number > math.floor(number) + 0.5:
        number = round(number, decimal_precision)
    elif number < math.ceil(number) - 0.5:
        number = truncate_decimals(number, decimal_precision)
    else:
        pass
    return number


df_inf = pd.read_csv(inf_file, delimiter=",", header=0)
df_inf.rename(columns={"period": "month"}, inplace=True)

# Import optimal purchase
opt = pd.read_csv(optimal_file, delimiter=";")

# # Set rounding
pd.set_option("display.float_format", lambda x: "%.7f" % x)

# # CONSTANTS
## Initial endowment
ENDOWMENT = 863.81
## Interest rate
INTEREST = 0.2277300 / 12
## Wage
WAGE = 4.32
logging.info(
    f"Unrounded constants:\n\
      Initial endowment: {ENDOWMENT}, Interest rate: {INTEREST}, Wage: {WAGE}"
)

## Correct for Python rounding issues versus interest rate used in oTree
INTEREST = Decimal(INTEREST).quantize(Decimal(".0000001"), rounding=ROUND_UP)
INTEREST = float(INTEREST)
logging.info(
    f"Rounded constants:\n\
      Initial endowment: {ENDOWMENT}, Interest rate: {INTEREST}, Wage: {WAGE}"
)

## Columns to pull from original dataframes
MELT_COLS = [
    "participant.code",
    "participant.label",
    "date",
    "treatment",
    "participant.inflation",
    "participant.round",
]

# # Create time series for purchase decisions and stock
## Decision quantity
df1 = final_df_dict["decision"].copy()
df2 = df1.melt(
    id_vars=MELT_COLS,
    value_vars=[c for c in df1.columns if "decision" in c],
    var_name="month",
    value_name="decision",
)

## For stock
df1 = final_df_dict["finalStock"].copy()
df3 = df1.melt(
    id_vars=MELT_COLS,
    value_vars=[c for c in df1.columns if "finalStock" in c],
    var_name="month",
    value_name="finalStock",
)

## Add price column
df_prices = final_df_dict["newPrice"].copy()
df_prices2 = df_prices.melt(
    id_vars=MELT_COLS,
    value_vars=[c for c in df_prices.columns if "newPrice" in c],
    var_name="month",
    value_name="newPrice",
)
## Round new price after inflation rate applied
df_prices2["newPrice"] = df_prices2["newPrice"].apply(round_price)

## Combine dataframes
df_combine = pd.concat([df2, df3, df_prices2], axis=1, join="inner")
# df = df.merge(df_prices2, how="left")
## Remove duplicate rows and columns
df_combine.drop_duplicates(inplace=True)  ## Rows
df_combine = df_combine.loc[:, ~df_combine.columns.duplicated()].copy()
## Extract month number
df_combine["month"] = df_combine["month"].str.extract("(\d+)")
## Convert columns to int, except date
cols_to_convert = [c for c in df_combine.columns if c != "date"]
df_combine[cols_to_convert] = df_combine[cols_to_convert].apply(
    pd.to_numeric, errors="ignore"
)
df_combine.sort_values(
    [
        "participant.round",
        "participant.code",
        "month",
    ],
    ascending=True,
    inplace=True,
)

## Categorize pre- and post-intervention
criteria = [
    df_combine["participant.round"].lt(2),
]
choices = ["pre"]
df_combine["phase"] = np.select(criteria, choices, default="post")
## Categorize inflation phases (1=high, 0=low)
## Months of low inflation
inf_phases_1012 = [i for i in range(0, 121) if ((i - 1) / 12 % 2) < 1]
inf_phases_430 = [i for i in range(0, 121) if ((i - 1) / 30 % 2) < 1]

criteria = [
    df_combine["month"].isin(inf_phases_1012)
    & df_combine["participant.inflation"].eq(1012),
    df_combine["month"].isin(inf_phases_430)
    & df_combine["participant.inflation"].eq(430),
]

choices = [0, 0]
df_combine["inf_phase"] = np.select(criteria, choices, default=1)

## Cumulative purchases
df_combine = df_combine.merge(
    df_combine.groupby("participant.code")["decision"].cumsum(),
    left_index=True,
    right_index=True,
)
df_combine.rename(
    columns={"decision_x": "decision", "decision_y": "cum_decision"}, inplace=True
)
# ## Stock phase 1
# Stock at end of of month t, removing excess purchases: SG1_t = Min(SGt, 120-t)
# Stock at end of of month t, removing excess purchases: SG1_t = Min(SGt, 120-t)
df_opp_cost = df_combine.copy()

## Add naive and optimal strategy stocks
df_opp_cost["sgnaive"] = 0
## Convert opt table to time series
opt2 = opt.melt(
    id_vars="month",
    value_vars=["optimal_430", "optimal_1012"],
    var_name="participant.inflation",
    value_name="sgoptimal",
)
## Remove additional text for inflation sequences
opt2["participant.inflation"] = opt2["participant.inflation"].str.extract("(\d+)")
## Convert data types
opt2 = opt2.apply(pd.to_numeric, errors="ignore")
## Combine with participants
df_opp_cost = df_opp_cost.merge(opt2, how="left")

## SG1_t = min(SG_t, 120-t)
df_opp_cost["mos_remaining"] = 120 - df_opp_cost["month"]
df_opp_cost["sg1"] = df_opp_cost[["finalStock", "mos_remaining"]].min(axis=1)
## Remove mos_remaining column
df_opp_cost = df_opp_cost[[c for c in df_opp_cost.columns if "mos_remaining" not in c]]

# ## Stock phase 2
# Final stock at month t after removing excess and early purchases:
# - for months t of first low-inflation phase: `SG2_t = 0`
# - for all following low-inflation phases, `SG2_t=Max(0, SG1_[t-1] - 1)` in first month,
# then `SG2_t+1=Max(0, SG2_t)`
# - for all months with inflation, `SG2_t = SG1_t`

criteria = [
    ## 1st phase, SG2_t = 0
    df_opp_cost["participant.inflation"].eq(1012) & df_opp_cost["month"].le(12),
    df_opp_cost["participant.inflation"].eq(430) & df_opp_cost["month"].le(30),
    ## For high-inflation phases, SG2_t = SG1_t
    df_opp_cost["inf_phase"].eq(1),
    ## 1st month in low-inflation phase, SG2_t = SG1_(t-1) - 1
    df_opp_cost["participant.inflation"].eq(1012)
    & df_opp_cost["month"].gt(12)
    & df_opp_cost["inf_phase"].shift(1).eq(1)
    & df_opp_cost["inf_phase"].eq(0),
    df_opp_cost["participant.inflation"].eq(430)
    & df_opp_cost["month"].gt(30)
    & df_opp_cost["inf_phase"].shift(1).eq(1)
    & df_opp_cost["inf_phase"].eq(0),
]

choices = [
    0,
    0,
    df_opp_cost["sg1"],
    df_opp_cost["sg1"].shift(1) - 1,
    df_opp_cost["sg1"].shift(1) - 1,
]

df_opp_cost["sg2"] = np.select(criteria, choices, default=np.nan)

## After 1st month in low-inflation phases, SG2_t = max(0, (SG2_t) - 1)
df_opp_cost["sg2"] = (
    df_opp_cost["sg2"].ffill()
    - df_opp_cost.groupby(df_opp_cost["sg2"].notnull().cumsum()).cumcount()
)
df_opp_cost["sg2"] = np.maximum(df_opp_cost["sg2"], 0)

# ## Non-foresighted strategy
# Non-farsighted strategy: SGNF_t = SG2_t during the first low-inflation phase
# and the first month of high inflation, then `SGNF_(t+1) = Max(0, SGNF_t)`

## 1st phase and 1st month of first high-inflation phase, SGNF_t = SG2_t
criteria = [
    df_opp_cost["participant.inflation"].eq(1012) & df_opp_cost["month"].le(13),
    df_opp_cost["participant.inflation"].eq(430) & df_opp_cost["month"].le(31),
]
choices = [
    df_opp_cost["sg2"],
    df_opp_cost["sg2"],
]
df_opp_cost["sgnf"] = np.select(criteria, choices, default=np.nan)

## Afterwards, SGNF_(t+1) = max(0, (SGNF_t) - 1)
df_opp_cost["sgnf"] = (
    df_opp_cost["sgnf"].ffill()
    - df_opp_cost.groupby(df_opp_cost["sgnf"].notnull().cumsum()).cumcount()
)
df_opp_cost["sgnf"] = np.maximum(df_opp_cost["sgnf"], 0)

# ## Consumption
# From the modified stocks, it is possible to calculate the corresponding
# consumption decisions for strategy `i`: `qi_t = SGi_t - SGi_(t-1) + 1`

## Calculate the purchase decision based on qi_t = SGi_t - SGi_(t-1) + 1,
## where qi_1 = SGi_1
strategies = ["naive", "1", "2", "nf", "optimal"]
for s in strategies:
    df_opp_cost[f"q{s}"] = np.nan
    df_opp_cost[f"q{s}"][df_opp_cost["month"] == 1] = df_opp_cost[f"sg{s}"] + 1
    df_opp_cost[f"q{s}"][df_opp_cost["month"] > 1] = (
        df_opp_cost[f"sg{s}"] - df_opp_cost[f"sg{s}"].shift(1) + 1
    )

# ## Savings
# Calculate the savings for strategy `i`: `Si_t=Si_t-1(1+r)+y-ptqi_t`

## Calculate savings balance for each strategy, `i`
strategies = ["naive", "1", "2", "nf", "optimal", "real"]
logging.info("Calculating savings balances for strategies: %s", ", ".join(strategies))
df_opp_cost = strategy_savings_calc(strategies, df_opp_cost)
print(df_opp_cost[(df_opp_cost["month"] == 13) | (df_opp_cost["month"] == 31)].head())

# ## Actual savings

## Add actual savings balance
df_save = final_df_dict["finalSavings"].copy()
df_save2 = df_save.melt(
    id_vars=[
        "participant.code",
        "participant.label",
        "participant.inflation",
        "participant.round",
    ],
    value_vars=[c for c in df_save.columns if "finalSavings" in c],
    var_name="month",
    value_name="finalSavings",
)

## Extract month number
df_save2["month"] = df_save2["month"].str.extract("(\d+)")
## Convert to int
df_save2 = df_save2.apply(pd.to_numeric, errors="ignore")

## Reorder
df_save2.sort_values(
    [
        "participant.round",
        "participant.code",
        "month",
    ],
    ascending=True,
    inplace=True,
)

df_opp_cost = df_opp_cost.merge(df_save2, how="left")

logging.info("Done, df_opp_cost shape: %s", df_opp_cost.shape)

# * Calculate opportunity costs for each category: early, late, and excess
df_opp_cost = categorize_opp_cost(df_opp_cost)
logging.info("Done, df_opp_cost shape: %s", df_opp_cost.shape)
logging.info("Done, df_opp_cost columns: %s", df_opp_cost.columns.to_list())


def plot_savings_and_stock(data: pd.DataFrame, **kwargs) -> None:
    """Plot average performance versus optimal and naive strategies

    Args:
        data (pd.DataFrame): _description_
    """
    ## Convert to time series-esque dataframe for multi-bar plot
    df_stock = data.melt(
        id_vars=[
            "participant.code",
            "participant.label",
            "treatment",
            "participant.inflation",
            "phase",
            "month",
        ],
        var_name="Strategy",
        value_vars=["participant.inflation", "finalStock", "sgoptimal", "sgnaive"],
        value_name="Stock",
    )

    df_savings = data.melt(
        id_vars=[
            "participant.code",
            "participant.label",
            "treatment",
            "participant.inflation",
            "phase",
            "month",
        ],
        var_name="Strategy",
        value_vars=["participant.inflation", "sreal", "soptimal", "snaive"],
        value_name="Savings",
    )

    dfts = pd.concat([df_stock, df_savings], axis=1, join="inner")

    dfts.drop_duplicates(inplace=True)

    ## Remove duplicate columns
    dfts = dfts.loc[:, ~dfts.columns.duplicated()].copy()

    ## Rename strategies
    dfts.Strategy.replace(
        ["finalStock", "sgnaive", "sgoptimal"],
        ["Average", "Naive", "Optimal"],
        inplace=True,
    )

    fig = sns.catplot(
        data=dfts,
        x="month",
        y="Stock",
        kind="bar",
        hue="Strategy",
        legend_out=False,
        estimator="mean",
        errorbar=None,
        height=5,
        aspect=1.75,
        **kwargs,
    )
    logger.debug("items %s", fig.axes_dict.items())
    for phase, ax in fig.axes_dict.items():
        logger.debug(phase)
        if type(phase) == tuple:
            data_line_plot = dfts[
                (dfts["phase"] == phase[1]) & (dfts["treatment"] == phase[0])
            ]
        elif type(phase) == str:
            data_line_plot = dfts[dfts["phase"] == phase]
        ax2 = ax.twinx()
        sns.lineplot(
            data=data_line_plot,
            legend=None,
            x="month",
            y="Savings",
            hue="Strategy",
            ci=None,
            ax=ax2,
            palette=kwargs["palette"],
        )
        ax2.set_ylim(0, dfts["Savings"].max() + 500)

    ax2.set_xticks(ax2.get_xticks()[0:120:12])
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Export results to csv file & graph stuffs"""
    export_data = input("Export data? (y/n):")
    if export_data != "y" and export_data != "n":
        export_data = input("Please respond with 'y' or 'n':")
    if export_data == "y":
        timestr = time.strftime("%Y%m%d-%H%M%S")
        logging.info(timestr)
        file_name = f"{final_dir}/{FINAL_FILE_PREFIX}_{timestr}.csv"
        df_opp_cost.to_csv(file_name, sep=";")
        logging.info("Created %s", file_name)

    graph_data = input("Plot data? (y/n):")
    if graph_data != "y" and graph_data != "n":
        graph_data = input("Please respond with 'y' or 'n':")
    if graph_data == "y":
        plot_savings_and_stock(df_opp_cost, col="phase", palette="tab10")


if __name__ == "__main__":
    main()
