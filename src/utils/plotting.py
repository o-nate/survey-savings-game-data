"""Plotting functions"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import calc_opp_costs, process_survey

from src.utils.constants import INTEREST_RATE
from src.utils.logging_config import get_logger

# * Set logger
logger = get_logger(__name__)

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# * Decimal rounding
pd.set_option("display.float_format", lambda x: "%.2f" % x)


def visualize_persona_results(
    data: pd.DataFrame, persona_horizon: str, measures: list[str]
) -> plt.Figure:
    grouped_df = (
        data.dropna()
        .groupby([persona_horizon, "treatment", "participant.round"])[measures]
        .describe()[[(m, me) for m in measures for me in ["count", "mean"]]]
    ).reset_index()

    # Flatten column names
    grouped_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] != "" else col[0] for col in grouped_df.columns
    ]

    # Pivot to get participant.round as columns for easier difference calculation
    values = [f"{m}_mean" for m in measures] + [f"{measures[-1]}_count"]
    pivot_df = grouped_df.pivot_table(
        index=[persona_horizon, "treatment"],
        columns="participant.round",
        values=values,
    ).reset_index()

    # Calculate the difference between Round 2 and Round 1
    for measure in measures:
        pivot_df[f"{measure}_diff"] = (
            pivot_df[(f"{measure}_mean", 2)] - pivot_df[(f"{measure}_mean", 1)]
        )
        if measure == measures[-1]:
            pivot_df["count_diff"] = (
                pivot_df[(f"{measure}_count", 2)] - pivot_df[(f"{measure}_count", 1)]
            )

    # Plot configuration for three columns (Round 1, Round 2, Difference)
    fig, axs = plt.subplots(len(measures) + 1, 3, figsize=(18, 5 * len(measures)))
    fig.suptitle(
        "Mean and Count Results by Participant Round and Difference", fontsize=16
    )

    round1_df = grouped_df[grouped_df["participant.round"] == 1]
    round2_df = grouped_df[grouped_df["participant.round"] == 2]

    for idx, measure in enumerate(measures):

        # Round 1 plot
        sns.barplot(
            data=round1_df,
            x=persona_horizon,
            y=f"{measure}_mean",
            hue="treatment",
            ax=axs[idx, 0],
            dodge=True,
            legend=False,
        )
        axs[idx, 0].set_title(f"{measure}: Round 1")
        axs[idx, 0].set_ylabel("Mean")

        # Round 2 plot
        sns.barplot(
            data=round2_df,
            x=persona_horizon,
            y=f"{measure}_mean",
            hue="treatment",
            ax=axs[idx, 1],
            dodge=True,
            legend=False,
        )
        axs[idx, 1].set_title(f"{measure}: Round 2")
        axs[idx, 1].set_ylabel("Mean")

        # Difference plot
        sns.barplot(
            data=pivot_df,
            x=persona_horizon,
            y=f"{measure}_diff",
            hue="treatment",
            ax=axs[idx, 2],
            dodge=True,
            legend=False,
        )
        axs[idx, 2].set_title(f"{measure}: Difference (Round 2 - Round 1)")
        axs[idx, 2].set_ylabel("Difference")

        if idx == len(measures) - 1:
            # Round 1 plot
            sns.barplot(
                data=round1_df,
                x=persona_horizon,
                y=f"{measure}_count",
                hue="treatment",
                ax=axs[idx + 1, 0],
                dodge=True,
                legend=False,
            )
            axs[idx + 1, 0].set_title("Count: Round 1")
            axs[idx + 1, 0].set_ylabel("Count")

            # Round 2 plot
            sns.barplot(
                data=round2_df,
                x=persona_horizon,
                y=f"{measure}_count",
                hue="treatment",
                ax=axs[idx + 1, 1],
                dodge=True,
                legend=False,
            )
            axs[idx + 1, 1].set_title("Count: Round 2")
            axs[idx + 1, 1].set_ylabel("Count")

            # Difference plot
            sns.barplot(
                data=pivot_df,
                x=persona_horizon,
                y="count_diff",
                hue="treatment",
                ax=axs[idx + 1, 2],
                dodge=True,
                legend=False,
            )
            axs[idx + 1, 2].set_title("Count: Difference (Round 2 - Round 1)")
            axs[idx + 1, 2].set_ylabel("Difference")

    # Extract legend handles and labels from one of the plots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles and labels:
        # Create a common legend
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(labels),
            title="Treatment",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> None:

    df_opp_cost = calc_opp_costs.calculate_opportunity_costs()

    df_opp_cost = df_opp_cost.rename(columns={"month": "Month"})
    df_opp_cost.head()

    df_survey = process_survey.create_survey_df(include_inflation=True)
    df_inf_measures = process_survey.pivot_inflation_measures(df_survey)
    df_inf_measures = process_survey.include_inflation_measures(df_inf_measures)
    df_inf_measures["participant.inflation"] = np.where(
        df_inf_measures["participant.inflation"] == "4x30", 430, 1012
    )

    df_decisions = df_inf_measures.merge(df_opp_cost, how="left")

    # * Store final savings at month t = 120
    df_decisions["finalSavings_120"] = (
        df_decisions[df_decisions["Month"] == 120]
        .groupby("participant.code")["finalSavings"]
        .transform("mean")
    )
    df_decisions["finalSavings_120"] = df_decisions.groupby("participant.code")[
        "finalSavings_120"
    ].bfill()

    df_decisions.head()

    # * Classify subjects as Rational-Accurate, Rational-Pessimitic,
    # * Irrational-MoneyIllusion, Irrational-DeathAverse

    MAX_RATIONAL_STOCK = 15
    PERSONAS = [
        "Rational & Accurate",
        "Rational & Pessimistic",
        "Irrational & Money Illusioned",
        "Irrational & Death Averse",
    ]  # ["RA", "RP", "IM", "ID"]
    ANNUAL_INTEREST_RATE = ((1 + INTEREST_RATE) ** 12 - 1) * 100

    df_personas = df_decisions[df_decisions["Month"].isin([1, 12])]
    df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
        "Quant Expectation"
    ].shift(1)

    _, axs = plt.subplots(3, 5, figsize=(30, 20))
    axs = axs.flatten()

    # _, axs2 = plt.subplots(3, 5, figsize=(30, 20))
    # axs2 = axs2.flatten()

    for max_stock in list(range(MAX_RATIONAL_STOCK)):
        data = df_personas.copy()

        CONDITIONS = [
            # Rational and accurate
            (data["finalStock"] <= max_stock)
            & (data["previous_expectation"] <= ANNUAL_INTEREST_RATE),
            # Rational and pessimistic
            (data["finalStock"] > max_stock)
            & (data["previous_expectation"] > ANNUAL_INTEREST_RATE),
            # Irrational and money illusioned
            (data["finalStock"] <= max_stock)
            & (data["previous_expectation"] > ANNUAL_INTEREST_RATE),
            # Irrational and death averse
            (data["finalStock"] > max_stock)
            & (data["previous_expectation"] <= ANNUAL_INTEREST_RATE),
        ]

        data[f"persona_horizon_{max_stock}"] = np.select(
            condlist=CONDITIONS, choicelist=PERSONAS, default=np.nan
        )

        # * Add column for persona based on max_stock to track how distribution changes
        df_personas = df_personas.merge(data, how="left")

        data = data[data["Month"].isin([12])]

        print(data.value_counts(f"persona_horizon_{max_stock}"))

    MEASURES = ["previous_expectation", "finalStock", "finalSavings_120"]

    print(
        df_personas.dropna()
        .groupby(["persona_horizon_0", "treatment", "participant.round"])[MEASURES]
        .describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]
    )

    _ = visualize_persona_results(df_personas, "persona_horizon_0", MEASURES)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
