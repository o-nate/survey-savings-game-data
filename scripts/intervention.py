"""Script to analyze intervention's effect"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from preprocess import final_df_dict

logging.basicConfig(level="DEBUG")


def main() -> None:
    """Run script"""
    df_int = final_df_dict["task_int"].copy()
    print(df_int)


if __name__ == "__main__":
    main()
