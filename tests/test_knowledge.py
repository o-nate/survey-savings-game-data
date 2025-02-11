"""Test knowledge measures module"""

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from src import calc_opp_costs, discontinuity, process_survey

from src.knowledge import count_correct_responses, create_knowledge_dataframe
from src.preprocess import final_df_dict
from src.utils.logging_config import get_logger

from tests.utils import constants

logger = get_logger(__name__)

logger.info("Testing knowledge measures")

df = final_df_dict["Finance"].copy()
measure = "financial_literacy"
df[measure] = count_correct_responses(df, measure)
result = df[df["participant.code"] == constants.FIN_LIT_PARTICIPANT_CODE][
    measure
].values[0]
assert result == constants.SCORE


df = final_df_dict["Numeracy"].copy()
measure = "numeracy"
df[measure] = count_correct_responses(df, measure)
result = df[df["participant.code"] == constants.NUMERACY_PARTICIPANT_CODE_2B][
    measure
].values[0]
assert result == constants.SCORE

result = df[df["participant.code"] == constants.NUMERACY_PARTICIPANT_CODE_3][
    measure
].values[0]
assert result == constants.SCORE

result = df[df["participant.code"] == constants.NUMERACY_PARTICIPANT_CODE_3_NOT][
    measure
].values[0]
assert result != constants.SCORE


df = final_df_dict["Inflation"].copy()
measure = "compound"
df[measure] = count_correct_responses(df, measure)
result = df[df["participant.code"] == constants.COMPOUND_PARTICIPANT_CODE][
    measure
].values[0]
assert result == constants.SCORE

logger.info("Testing knowledge dataframe creation")
df = create_knowledge_dataframe()
result_fin = df[
    df["participant.label"] == constants.KNOWLEDGE_DATAFRAME_PARTICIPANT_LABEL
]["financial_literacy"].values[0]
assert result_fin == constants.KNOWLEDGE_DATAFRAME_FIN_LIT_SCORE
df = create_knowledge_dataframe()
result_fin = df[
    df["participant.label"] == constants.KNOWLEDGE_DATAFRAME_PARTICIPANT_LABEL
]["numeracy"].values[0]
assert result_fin == constants.KNOWLEDGE_DATAFRAME_NUM_SCORE
df = create_knowledge_dataframe()
result_fin = df[
    df["participant.label"] == constants.KNOWLEDGE_DATAFRAME_PARTICIPANT_LABEL
]["compound"].values[0]
assert result_fin == constants.KNOWLEDGE_DATAFRAME_COMPOUND_SCORE

logger.info("Testing complete")
