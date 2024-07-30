"""Tests for opportunity costs calculation module"""

import logging
import sys

from src.calc_opp_costs import df_opp_cost
from src.utils.logging_helpers import set_external_module_log_levels

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

TEST_PARTICIPANT_CODE = "17c9d4zc"

logger.info("Initiating tests")

logger.info("Testing late cost makes sense at month 1")
assert (
    df_opp_cost[df_opp_cost["participant.code"] == TEST_PARTICIPANT_CODE]["late"].iat[0]
    == 0
)

logger.info("Testing costs and savings add to maximum")
total_costs = (
    df_opp_cost[
        (df_opp_cost["participant.code"] == TEST_PARTICIPANT_CODE)
        & (df_opp_cost["month"] == 120)
    ][["early", "late", "excess"]]
    .sum(axis=1)
    .values[0]
)
savings = df_opp_cost[
    (df_opp_cost["participant.code"] == TEST_PARTICIPANT_CODE)
    & (df_opp_cost["month"] == 120)
]["sreal"].iat[0]
max_savings = df_opp_cost[
    (df_opp_cost["participant.code"] == TEST_PARTICIPANT_CODE)
    & (df_opp_cost["month"] == 120)
]["soptimal"].iat[0]

logger.debug(
    "total costs (%s) + savings (%s) %s = max savings (%s)",
    total_costs,
    savings,
    total_costs + savings,
    max_savings,
)

assert total_costs + savings == max_savings


logger.info("Test complete")
