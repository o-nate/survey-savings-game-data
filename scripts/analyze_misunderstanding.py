"""Analyze wasteful-stocking and errors in instructions questions"""

# %%
import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pingouin import mediation_analysis
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

from scripts.utils import constants

from src import (
    calc_opp_costs,
    discontinuity,
    intervention,
    econ_preferences,
    knowledge,
    process_survey,
)

from src.preprocess import final_df_dict
from src.stats_analysis import (
    create_bonferroni_correlation_table,
    create_pearson_correlation_matrix,
    run_forward_selection,
    run_treatment_forward_selection,
)
from src.utils.helpers import combine_series, export_plot
from src.utils.logging_config import get_logger

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

## Decimal rounding
pd.set_option("display.float_format", lambda x: "%.2f" % x)


# %%
df_opp_cost = calc_opp_costs.calculate_opportunity_costs()

# %%
df = final_df_dict["task_instructions"].copy()
df2 = df_opp_cost[
    (df_opp_cost["month"] == 120) & (df_opp_cost["participant.round"] == 1)
]
df3 = df2[["participant.label", "excess_%", "early_%", "finalSavings_%"]].merge(
    df[["participant.label"] + [c for c in df.columns if "error" in c]], how="left"
)

# %% [markdown]
## Plot correlation matrix
# Compute the correlation matrix
corr = df3[[c for c in df3.columns if "label" not in c]].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(20, 220, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=0.3,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
)
