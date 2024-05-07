"""Statistical analysis of data"""

# %%

import logging

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# from preprocess import final_df_dict
from calc_opp_costs import df_str
from discontinuity import purchase_discontinuity

# * Output settings
logging.basicConfig(level="INFO")
pd.set_option("display.max_rows", None, "display.max_columns", None)

# * Define `decision quantity` measure
DECISION_QUANTITY = "cum_decision"

# * Define purchase window, i.e. how many months before and after inflation phase change to count
WINDOW = 3

# %%
df_results = df_str.copy()
print(df_results.head())
logging.debug(df_results.shape)

# %%
df_results = purchase_discontinuity(
    df_results, decision_quantity=DECISION_QUANTITY, window=WINDOW
)
df_results.head()

# %%
print(
    df_results[(df_results["month"] == 120) & (df_results["finalSavings"] > 0)]
    .groupby("participant.round")[["finalSavings", "early", "late", "excess"]]
    .describe()
    .T
)
