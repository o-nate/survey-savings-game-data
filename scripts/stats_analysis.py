"""Statistical analysis of data"""

# %%

import logging

from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import statsmodels.api as sm
import statsmodels.formula.api as smf

from preprocess import final_df_dict
from calc_opp_costs import df_str
from discontinuity import purchase_discontinuity
from process_survey import create_survey_df
from src.utils.logging_helpers import set_external_module_log_levels

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("warning")
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
    df_stock["stock_after"] = df_stock.groupby("participant.code")["finalStock"].shift(
        -11
    )
    print(df_stock.head())
    print(
        df_stock[df_stock["month"] == 1]
        .groupby("participant.round")[["estimate", "finalStock", "stock_after"]]
        .describe()
        .T
    )

    # * Imput 0% for expectation when qualitative expectation is 0
    df_stock[df_stock["month"] == 1]["estimate"] = df_stock[df_stock["month"] == 1][
        "estimate"
    ].fillna(0)

    # * Measure Pearson correlation
    print("\nPearson Correlations")
    print(
        "Month 1 x Month 1\n",
        df_stock[df_stock["month"] == 1]["finalStock"].corr(
            df_stock[df_stock["month"] == 1]["estimate"]
        ),
    )
    print(
        "Month 1 x Month 12\n",
        df_stock[df_stock["month"] == 1]["stock_after"].corr(
            df_stock[df_stock["month"] == 1]["estimate"]
        ),
    )

    # * Linear regression
    X = df_stock[(df_stock["month"] == 1) & (df_stock["estimate"].notnull())][
        ["estimate"]
    ].to_numpy()
    y = df_stock[(df_stock["month"] == 1) & (df_stock["estimate"].notnull())][
        ["stock_after"]
    ].to_numpy()

    ## sklearn
    reg = LinearRegression().fit(X, y)
    print("sklearn")
    print("Regression score:", reg.score(X, y))
    print("Regression coefficients:", reg.coef_)

    ## statsmodels
    # x = sm.add_constant(X)  # Add constant
    data = df_stock[(df_stock["month"] == 1) & (df_stock["estimate"].notnull())][
        ["estimate", "stock_after"]
    ]
    model = smf.ols(formula="stock_after ~ estimate", data=data)
    results = model.fit()
    print("statsmodel")
    print(results.summary())

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

        sns.jointplot(
            data=df_stock[(df_stock["month"] == 1)],
            x="estimate",
            y="stock_after",
            hue="participant.round",
        )

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        ax1 = sm.graphics.influence_plot(results)
        ax2 = sm.graphics.plot_regress_exog(results, "estimate")

        fig.tight_layout(pad=1.0)

        plt.show()


if __name__ == "__main__":
    main()
