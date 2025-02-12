"""Sentiment analysis of post-experiment survey responses"""

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
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

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

logger = get_logger(__name__)

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# * Decimal rounding
pd.set_option("display.float_format", lambda x: "%.2f" % x)

MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

# %%
# Use a pipeline as a high-level helper
pipe = pipeline("text-classification", model=MODEL, from_tf=True)

# %%
