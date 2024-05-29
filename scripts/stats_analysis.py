"""Statistical analysis of data"""

# %%

import logging

from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

from preprocess import final_df_dict
from calc_opp_costs import df_str
from discontinuity import purchase_discontinuity
from process_survey import create_survey_df
from src.helpers import disable_module_debug_log

# * Logging settings
logger = logging.getLogger(__name__)
disable_module_debug_log("warning")
logger.setLevel(logging.DEBUG)

# * Output settings
pd.set_option("display.max_rows", None, "display.max_columns", None)

# ! For exporting (if script run directly)
# * Declare name of output file
FINAL_FILE_PREFIX = "stats_analysis"

# * Declare directory of output file
final_dir = Path(__file__).parents[1] / "results" / "results_csv"

# * Define `decision quantity` measure
DECISION_QUANTITY = "cum_decision"

# * Define purchase window, i.e. how many months before and after inflation phase change to count
WINDOW = 3


# %%
def main() -> None:
    """Run script"""
    df_results = df_str.copy()
    logger.debug(df_results.shape)

    # %%
    df_results = purchase_discontinuity(
        df_results, decision_quantity=DECISION_QUANTITY, window=WINDOW
    )

    # %%
    print(df_results[(df_results["month"] == 120) & (df_results["finalSavings"] == 0)])
    print(
        df_results[df_results["month"] == 120]
        .groupby("participant.round")[["finalSavings", "early", "excess"]]
        .describe()
        .T
    )

    # * Correlation between expectation at month 1 and stock at months 1 and 12
    df1 = final_df_dict["inf_expectation"].copy()
    logger.debug(df1.columns.to_list())
    df_inf = df1.melt(
        id_vars=[
            "participant.code",
            "participant.label",
            "participant.inflation",
            "participant.round",
        ],
        value_vars=[c for c in df1.columns if "inf_" in c],
        var_name="Measure",
        value_name="estimate",
    )

    # * Extract month number
    df_inf["month"] = df_inf["Measure"].str.extract("(\d+)")
    ## Convert to int
    df_inf = df_inf.apply(pd.to_numeric, errors="ignore")
    logger.debug(df_inf.dtypes)

    df_stock = df_str.copy()
    df_stock = df_stock.merge(
        df_inf[["participant.code", "participant.label", "month", "estimate"]],
        how="left",
    )
    print(df_stock.head())
    logger.debug(df_stock.info())

    export_data = input("Export data? (y/n):")
    if export_data not in ("y", "n"):
        export_data = input("Please respond with 'y' or 'n':")
    if export_data == "y":
        timestr = time.strftime("%Y%m%d-%H%M%S")
        logger.info(timestr)
        file_name = f"{final_dir}/{FINAL_FILE_PREFIX}_{timestr}.csv"
        df_results[df_results["month"] == 120].groupby("participant.round")[
            ["finalSavings", "early", "excess"]
        ].describe().T.to_csv(file_name, sep=";")
        logger.info("Created %s", file_name)

    cols = [
        "finalSavings",
        "early",
        "excess",
    ]

    data = df_results[df_results["month"] == 120].copy()

    for c in cols:
        before = data[data["phase"] == "pre"][c]
        after = data[data["phase"] == "post"][c]
        p_value = stats.wilcoxon(
            before, after, zero_method="zsplit", nan_policy="raise"
        )[1]
        print(f"p value for {c}: {p_value}")

    graph_data = input("Plot data? (y/n):")
    if graph_data not in ("y", "n"):
        graph_data = input("Please respond with 'y' or 'n':")
    if graph_data == "y":
        data2 = df_results.melt(
            id_vars=[
                "participant.code",
                "participant.label",
                "participant.inflation",
                "participant.round",
            ],
            value_vars=cols,
            var_name="Measure",
            value_name="Result",
        )
        logger.debug(data2.columns.to_list())
        sns.violinplot(
            data=data2,
            x="Measure",
            y="Result",
            hue="participant.round",
            split=True,
            gap=0.1,
            density_norm="width",
            linewidth=1,
            linecolor="k",
            palette="tab10",
        )

        plt.show()


if __name__ == "__main__":
    main()
