"""Tests for preprocess module"""

import logging
import sys

from src.preprocess import final_df_dict
from src.utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

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
