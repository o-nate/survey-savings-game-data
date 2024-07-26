"""Process responses from inflation questionnaire"""

# %%
import logging
import sys

from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocess import final_df_dict
from src.helpers import disable_module_debug_log

# * Logging settings
logger = logging.getLogger(__name__)
disable_module_debug_log("warning")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# %%
df = final_df_dict["Inflation"].copy()
df.describe().T

# %%


# %%


def main() -> None:
    """Run script"""


if __name__ == "__main__":
    main()
