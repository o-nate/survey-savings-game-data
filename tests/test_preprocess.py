"""Tests for preprocess module"""

import logging
import sys

from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.preprocess import final_df_dict
from src.helpers import disable_module_debug_log

# * Logging settings
logger = logging.getLogger(__name__)
disable_module_debug_log("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Constants for testing
MEAN_TRIAL_NUMBER = 30
NCORR_TRIAL_NUMBER_5 = 107

logger.info("Testing preprocess module")

logger.info(
    "Testing conversion of nested dicts/lists to columns for Wisconsin Card Sorting Task"
)
assert final_df_dict["wisconsin"]["trial_number_30"].mean() == MEAN_TRIAL_NUMBER
logger.info("Trial numbers correct")

assert (
    final_df_dict["wisconsin"].value_counts("correct_5")[True] == NCORR_TRIAL_NUMBER_5
)
logger.info("Number of correct guesses correct")
