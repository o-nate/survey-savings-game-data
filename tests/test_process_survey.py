"""Tests for process_survey module"""

import logging
import sys

from src.process_survey import create_survey_df, pivot_inflation_measures
from src.utils.logging_helpers import set_external_module_log_levels


# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("Error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

TEST_PARTICIPANT_CODE = "17c9d4zc"
TEST_MONTH = 36
TEST_VALUES = {
    "Quant Expectation": 10.0,
    "Quant Perception": 3.0,
    "Qual Expectation": 3.0,
    "Qual Perception": 2.0,
}
TEST_DF_LEN = 3454

logger.info("Testing process_survey module")
data = create_survey_df()
data = data[
    (data["participant.code"] == TEST_PARTICIPANT_CODE) & (data["Month"] == TEST_MONTH)
]
for k, v in TEST_VALUES.items():
    assert data[data["Measure"] == k]["Estimate"].iat[0] == v

logger.info("Testing pivot function")
data = create_survey_df()
df = pivot_inflation_measures(data)
assert df.shape[0] == TEST_DF_LEN


logger.info("Test complete")
