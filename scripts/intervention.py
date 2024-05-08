"""Script to analyze intervention's effect"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from calc_opp_costs import df_str
from discontinuity import purchase_discontinuity
from preprocess import final_df_dict

logging.basicConfig(level="DEBUG")


def main() -> None:
    """Run script"""
    df_int = final_df_dict["task_int"].copy()

    questions = ["intro_1", "q", "confirm"]
    cols = [c for c in df_int.columns if any(q in c for q in questions)]
    logging.debug(cols)

    # TODO Link mistakes participants made to their questions and resposes

    df_melted = df_int.melt(
        id_vars=[
            "participant.code",
            "participant.label",
        ],
        value_vars=cols,
        var_name="Measure",
        value_name="Result",
    )

    graph_data = input("Plot data? (y/n):")
    if graph_data != "y" and graph_data != "n":
        graph_data = input("Please respond with 'y' or 'n':")
    if graph_data == "y":
        g = sns.FacetGrid(df_melted, row="Measure")
        g.map(plt.hist, "Result")
        plt.show()


if __name__ == "__main__":
    main()
