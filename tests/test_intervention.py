"""Test of intervention module"""

import logging
import math
import sys
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

from src.calc_opp_costs import calculate_opportunity_costs
from src.discontinuity import purchase_discontinuity
from src.intervention import calculate_change_in_measure
from src.preprocess import final_df_dict
from src.utils.logging_helpers import set_external_module_log_levels

from tests.utils.constants import CHANGE_IN_PERFORMANCE

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# * Define `decision quantity` measure
DECISION_QUANTITY = "cum_decision"

# * Define purchase window, i.e. how many months before and after inflation phase change to count
WINDOW = 3

logger.info("Testing intervention module")

df_int = final_df_dict["task_int"].copy()

df_results = calculate_opportunity_costs()

df_results = purchase_discontinuity(
    df_results, decision_quantity=DECISION_QUANTITY, window=WINDOW
)

questions = ["intro_1", "q", "confirm"]
cols = [c for c in df_int.columns if any(q in c for q in questions)]

# * Compare impact of intervention
measures = [
    "finalSavings",
    "early",
    "late",
    "excess",
]

data_df = df_results[df_results["month"] == 120].copy()
logging.debug(data_df.shape)
data_df = data_df.merge(df_int[["participant.label", "date"] + cols], how="left")

# * Rename measures
data_df.rename(
    columns={
        "finalSavings": "Total savings",
        "early": "Over-stocking",
        "late": "Under-stocking",
        "excess": "Wasteful-stocking",
    },
    inplace=True,
)

measures = ["Total savings", "Over-stocking", "Under-stocking", "Wasteful-stocking"]

# * Measure intervention impact
for treat in ["Intervention 1", "Intervention 2", "Control"]:
    for m in measures:
        before, after, _ = calculate_change_in_measure(
            data_df[data_df["treatment"] == treat], m
        )
        assert math.isclose(
            before, CHANGE_IN_PERFORMANCE[treat][f"Initial {m}"], rel_tol=0.05
        )
        assert math.isclose(
            after, CHANGE_IN_PERFORMANCE[treat][f"Final {m}"], rel_tol=0.05
        )

logger.info("Test complete")
