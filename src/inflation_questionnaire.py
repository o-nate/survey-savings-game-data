"""Process responses from inflation questionnaire"""

# %%
import logging
import sys

from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.calc_opp_costs import df_opp_cost
from src.preprocess import final_df_dict
from src.utils.logging_helpers import set_external_module_log_levels

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def calculate_inflation_score(data: pd.DataFrame) -> pd.Series:
    """Generate inflation behavior score based on answers to how individuals changed
    their behavior when faced with inflation in real life

    Args:
        data (pd.DataFrame): DataFrame with real-life behavioral change responses

    Returns:
        pd.Series: Series with score for each individual
    """
    return (
        data["Inflation.1.player.inf_checking"] * -1
        + data["Inflation.1.player.inf_stock"]
        + data["Inflation.1.player.inf_quantity"]
    )


# %%
df = final_df_dict["Inflation"].copy()
df.describe().T

# %%
# sns.pairplot(df[[c for c in df.columns if any(bc in c for bc inf_behavior_cols)]])


# %%
df["inflation_score"] = calculate_inflation_score(df)

sns.histplot(data=df, x="inflation_score")

inf_behavior_cols = [
    "Inflation.1.player.inf_food",
    "Inflation.1.player.inf_housing",
    "Inflation.1.player.inf_other",
    "Inflation.1.player.inf_quantity",
    "Inflation.1.player.inf_stock",
    "Inflation.1.player.inf_checking",
    "Inflation.1.player.inf_savings",
    "inflation_score",
]
# %%
df2 = df_opp_cost.copy()
df2 = df2.merge(df[["participant.label"] + inf_behavior_cols], how="left")

for c in inf_behavior_cols:
    sns.jointplot(
        data=df2[(df2["month"] == 120) & (df2["participant.round"] == 1)],
        x=c,
        y="sreal_%",
        # hue="participant.round",
        kind="reg",
    )
    plt.show()


# %%
def main() -> None:
    """Run script"""


if __name__ == "__main__":
    main()
