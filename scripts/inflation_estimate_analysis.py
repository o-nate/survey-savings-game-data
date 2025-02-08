# %%
import logging
import sys

from pathlib import Path

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
from src.process_survey import create_survey_df, pivot_inflation_measures
from src.stats_analysis import (
    create_bonferroni_correlation_table,
    create_pearson_correlation_matrix,
    run_forward_selection,
    run_treatment_forward_selection,
)
from src.utils.constants import INFLATION_DICT
from src.utils.helpers import combine_series, export_plot
from src.utils.logging_helpers import set_external_module_log_levels

inf_file = Path(__file__).parents[1] / "data" / "animal_spirits.csv"

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# %% [markdown]
## Estimate expectation and perception of inflation
# P_[12n] = P_[12(n-1)] + \beta*\pi_[12(n-1)-12n] + \epsilon
# E_[12n] = \sigma * E_[12(n-1)] + \beta*P_[12n] + \gamma\pi_[12n-12(n-1)] + \epsilon

df_survey = create_survey_df(include_inflation=True)
df_survey["participant.round"] = df_survey["participant.round"].astype(int)

inf = pd.DataFrame(INFLATION_DICT)
inf["participant.inflation"] = np.where(
    inf["participant.inflation"] == 430, "4x30", "10x12"
)

# %%
survey_pivot = pivot_inflation_measures(df_survey)
survey_pivot.head()

# %%
survey_pivot["Quant_Perception_before"] = survey_pivot.groupby("participant.code")[
    "Quant Perception"
].shift(-1)
survey_pivot["Quant_Expectation_before"] = survey_pivot.groupby("participant.code")[
    "Quant Expectation"
].shift(-1)

survey_pivot.head()
