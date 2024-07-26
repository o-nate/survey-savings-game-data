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

logger.info("Testing preprocess module")

logger.info(
    "Testing conversion of nested dicts/lists to columns for Wisconsin Card Sorting Task"
)

MEAN_TRIAL_NUMBER = 30  # Test that trial number column is correctly assigned
assert final_df_dict["wisconsin"]["trial_number_30"].mean() == 30
