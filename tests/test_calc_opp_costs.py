"""Tests for opportunity costs calculation module"""

import logging
import sys

from src.calc_opp_costs import calculate_opportunity_costs
from src.utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

TEST_PARTICIPANT_CODE = "17c9d4zc"

logger.info("Initiating tests")

logger.info("Testing late cost makes sense at month 1")
df_opp_cost = calculate_opportunity_costs()
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
