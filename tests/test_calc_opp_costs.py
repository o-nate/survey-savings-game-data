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

logger.info("Initiating tests")

logger.info("Testing calculation")
assert (
    df_opp_cost[
        (df_opp_cost["participant.label"] == "xVRXlxl")
        & (df_opp_cost["participant.round"] == 2)
    ]["late"].iat[0]
    == 0
)

logger.info("Test complete")
