"""Test economic preferences module"""

import logging
import sys

from src.econ_preferences import (
    count_preference_choices,
    count_switches,
    count_time_preference_switches,
)
from src.preprocess import final_df_dict
from src.utils.logging_helpers import set_external_module_log_levels
from utils import constants

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.info("Testing economic preferences module")

logger.info("Testing safe choices: Risk preferences")
df = final_df_dict["riskPreferences"].copy()
df["n_safe"] = count_preference_choices(df, "riskPreferences")
df["n_switches"] = count_switches(df, "riskPreferences")
result = df[df["participant.code"] == constants.RISK_SAFE_PARTICIPANT_CODE][
    "n_safe"
].values[0]
assert result == constants.RISK_SAFE_NUMBER
result = df[df["participant.code"] == constants.RISK_SAFE_PARTICIPANT_CODE][
    "n_switches"
].values[0]
assert result == constants.RISK_SWITCHES_NUMBER


logger.info("Testing switch count: Loss aversion")
df = final_df_dict["lossAversion"].copy()
df["n_switches"] = count_switches(df, "lossAversion")
result = df[df["participant.code"] == constants.LOSS_SWITCHES_PARTICIPANT_CODE][
    "n_switches"
].values[0]
assert result == constants.LOSS_SWITCHES_NUMBER


logger.info("Testing switch count: Time preferences")
df = final_df_dict["timePreferences"].copy()
df["n_present"] = count_preference_choices(df, "timePreferences")
result = df[df["participant.code"] == constants.TIME_PRESENT_PARTICIPANT_CODE][
    "n_present"
].values[0]
logger.debug("tpref result %s", result)
assert result == constants.TIME_PRESENT_NUMBER
df["n_switches"] = count_time_preference_switches(df)
result = df[df["participant.code"] == constants.TIME_PRESENT_PARTICIPANT_CODE][
    "n_switches"
].values[0]
logger.debug("tpref result switches %s", result)
assert result == constants.TIME_PRESENT_SWITCHES

logger.info("Tests complete")
