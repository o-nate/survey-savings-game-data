"""Script to analyze intervention's effect"""

import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

from calc_opp_costs import df_str
from discontinuity import purchase_discontinuity
from preprocess import final_df_dict
from src.helpers import disable_module_debug_log

# * Logging settings
logger = logging.getLogger(__name__)
disable_module_debug_log("warning")
logger.setLevel(logging.DEBUG)


# * Define `decision quantity` measure
DECISION_QUANTITY = "cum_decision"

# * Define purchase window, i.e. how many months before and after inflation phase change to count
WINDOW = 3


def measure_feedback_impact(
    data: pd.DataFrame, measures_impacted: List[str], error_feedback: List[str]
) -> None:
    """Print comparison between those who are convinced by intervention feedback and not across measures"""
    for m in measures_impacted:
        for error in error_feedback:
            convinced_response = 9 if error == "convinced" else 3
            field = (
                error if error == "convinced" else f"task_int.1.player.confirm_{error}"
            )
            before = data[
                (data["phase"] == "pre") & (data[field] == convinced_response)
            ][m]
            after = data[
                (data["phase"] == "post") & (data[field] == convinced_response)
            ][m]
            p_value = stats.wilcoxon(
                before, after, zero_method="zsplit", nan_policy="raise"
            )[1]
            print(f"p value for convinced of {error}, {m}:\t{p_value}")
            print(
                "Change in measure:",
                data[(data["phase"] == "post") & (data[field] == convinced_response)][
                    m
                ].mean()
                - data[(data["phase"] == "pre") & (data[field] == convinced_response)][
                    m
                ].mean(),
            )
            ## Not very convinced
            before = data[
                (data["phase"] == "pre") & (data[field] < convinced_response)
            ][m]
            after = data[
                (data["phase"] == "post") & (data[field] < convinced_response)
            ][m]
            p_value = stats.wilcoxon(
                before, after, zero_method="zsplit", nan_policy="raise"
            )[1]
            print(f"p value for not convinced of {error}, {m}:\t{p_value}")
            print(
                "Change in measure:",
                data[(data["phase"] == "post") & (data[field] < 3)][m].mean()
                - data[(data["phase"] == "pre") & (data[field] < 3)][m].mean(),
            )


def main() -> None:
    """Run script"""
    df_int = final_df_dict["task_int"].copy()
    df_results = df_str.copy()
    logging.debug(df_results.shape)

    df_results = purchase_discontinuity(
        df_results, decision_quantity=DECISION_QUANTITY, window=WINDOW
    )

    questions = ["intro_1", "q", "confirm"]
    cols = [c for c in df_int.columns if any(q in c for q in questions)]
    logging.debug(cols)

    # TODO Link mistakes participants made to their questions and responses
    # * Compare impact of intervention
    measures = [
        "finalSavings",
        "early",
        "late",
        "excess",
    ]

    data_df = df_results[df_results["month"] == 120].copy()
    logging.debug(data_df.shape)
    data_df = data_df.merge(df_int[["participant.label"] + cols], how="left")
    logging.debug(data_df.shape)
    data_df["convinced"] = data_df[[c for c in data_df.columns if "confirm" in c]].sum(
        axis=1
    )
    logging.debug([c for c in data_df.columns if "confirm" in c])
    print(data_df[["participant.label"] + cols + ["convinced"]].head())

    ## Fully convinced
    measure_feedback_impact(data_df, measures, ["convinced"])

    ## Mostly convinced
    errors = ["early", "excess"]
    measure_feedback_impact(data_df, measures, errors)

    graph_data = input("Plot intervention data? (y/n):")
    if graph_data != "y" and graph_data != "n":
        graph_data = input("Please respond with 'y' or 'n':")
    if graph_data == "y":
        df_melted = data_df.melt(
            id_vars=[
                "participant.code",
                "participant.label",
                "participant.round",
                "convinced",
                "task_int.1.player.confirm_early",
                "task_int.1.player.confirm_late",
                "task_int.1.player.confirm_excess",
            ],
            value_vars=measures,
            var_name="Measure",
            value_name="Result",
        )

        ## Convert to dummy variables for plotting
        df_melted["convinced"] = [
            True if answer == 9 else False for answer in df_melted["convinced"]
        ]
        df_melted["task_int.1.player.confirm_early"] = [
            True if answer == 3 else False
            for answer in df_melted["task_int.1.player.confirm_early"]
        ]
        df_melted["task_int.1.player.confirm_late"] = [
            True if answer == 3 else False
            for answer in df_melted["task_int.1.player.confirm_late"]
        ]
        df_melted["task_int.1.player.confirm_excess"] = [
            True if answer == 3 else False
            for answer in df_melted["task_int.1.player.confirm_excess"]
        ]

        sns.catplot(
            data=df_melted,
            x="Measure",
            y="Result",
            col="convinced",
            hue="participant.round",
            kind="violin",
            split=True,
        )

        for m in ["early", "late", "excess"]:
            sns.catplot(
                data=df_melted,
                x="Measure",
                y="Result",
                col=f"task_int.1.player.confirm_{m}",
                hue="participant.round",
                kind="violin",
                split=True,
            )

        plt.show()

    graph_data = input("Plot general response data? (y/n):")
    if graph_data != "y" and graph_data != "n":
        graph_data = input("Please respond with 'y' or 'n':")
    if graph_data == "y":
        df_melted = df_int.melt(
            id_vars=[
                "participant.code",
                "participant.label",
            ],
            value_vars=cols,
            var_name="Measure",
            value_name="Result",
        )
        g = sns.FacetGrid(df_melted, row="Measure")
        g.map(plt.hist, "Result")
        plt.show()


if __name__ == "__main__":
    main()
