"""Statistical analysis of data"""

# %%

import logging
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

# from preprocess import final_df_dict
from calc_opp_costs import df_str
from discontinuity import purchase_discontinuity

# * Output settings
logging.basicConfig(level="INFO")
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
    logging.debug(df_results.shape)

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
    export_data = input("Export data? (y/n):")
    if export_data != "y" and export_data != "n":
        export_data = input("Please respond with 'y' or 'n':")
    if export_data == "y":
        timestr = time.strftime("%Y%m%d-%H%M%S")
        logging.info(timestr)
        file_name = f"{final_dir}/{FINAL_FILE_PREFIX}_{timestr}.csv"
        df_results[df_results["month"] == 120].groupby("participant.round")[
            ["finalSavings", "early", "excess"]
        ].describe().T.to_csv(file_name, sep=";")
        logging.info("Created %s", file_name)

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
    if graph_data != "y" and graph_data != "n":
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
        logging.debug(data2.columns.to_list())
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
