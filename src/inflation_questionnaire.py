"""Process responses from inflation questionnaire"""

# %%
import logging
import sys

from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocess import final_df_dict
from src.utils.logging_helpers import set_external_module_log_levels

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# %%
df = final_df_dict["Inflation"].copy()
df.describe().T

# %%
inf_behavior_cols = [c for c in df.columns if "inf_" in c]
sns.pairplot(df[inf_behavior_cols])


# %%


def main() -> None:
    """Run script"""


if __name__ == "__main__":
    main()
