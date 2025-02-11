# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import calc_opp_costs, process_survey

from src.utils.constants import INTEREST_RATE
from src.utils.logging_config import get_logger

# * Set logger
logger = get_logger(__name__)

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# %%
df_opp_cost = calc_opp_costs.calculate_opportunity_costs()

# %%
df_opp_cost = df_opp_cost.rename(columns={"month": "Month"})
df_opp_cost.head()

# %%
df_survey = process_survey.create_survey_df(include_inflation=True)
df_inf_measures = process_survey.pivot_inflation_measures(df_survey)
df_inf_measures = process_survey.include_inflation_measures(df_inf_measures)
df_inf_measures["participant.inflation"] = np.where(
    df_inf_measures["participant.inflation"] == "4x30", 430, 1012
)

# %%
df_decisions = df_inf_measures.merge(df_opp_cost, how="left")
df_decisions.head()

# %%
# * Classify subjects as Rational-Accurate, Rational-Pessimitic, Irrational-MoneyIllusion, Irrational-DeathAverse

MAX_RATIONAL_STOCK = 15
PERSONAS = ["RA", "RP", "IM", "ID"]

df_personas = df_decisions[
    (df_decisions["Month"].isin([1, 12])) & (df_decisions["participant.round"] == 1)
]
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    "Quant Expectation"
].shift(1)

_, axs = plt.subplots(3, 5, figsize=(30, 20))
axs = axs.flatten()

_, axs2 = plt.subplots(3, 5, figsize=(30, 20))
axs2 = axs2.flatten()

for max_stock in list(range(MAX_RATIONAL_STOCK)):
    print(max_stock)
    data = df_personas.copy()

    CONDITIONS = [
        # Rational and accurate
        (data["finalStock"] <= max_stock)
        & (data["previous_expectation"] <= INTEREST_RATE),
        # Rational and pessimistic
        (data["finalStock"] > max_stock)
        & (data["previous_expectation"] > INTEREST_RATE),
        # Irrational and money illusioned
        (data["finalStock"] <= max_stock)
        & (data["previous_expectation"] > INTEREST_RATE),
        # Irrational and death averse
        (data["finalStock"] > max_stock)
        & (data["previous_expectation"] <= INTEREST_RATE),
    ]

    data[f"persona_horizon_{max_stock}"] = np.select(
        condlist=CONDITIONS, choicelist=PERSONAS
    )
    data = data[data["Month"].isin([12])]

    print(data.value_counts(f"persona_horizon_{max_stock}"))

    sns.histplot(
        data, x=f"persona_horizon_{max_stock}", ax=axs[max_stock], stat="percent"
    )
    sns.scatterplot(
        data,
        x="finalStock",
        y="previous_expectation",
        hue=f"persona_horizon_{max_stock}",
        ax=axs2[max_stock],
    )

# %%
# * Repeat with qualitative expectations

QUALITATIVE_EXPECTATION_THRESHOLD = 1

df_personas = df_decisions[
    (df_decisions["Month"].isin([1, 12])) & (df_decisions["participant.round"] == 1)
]
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    "Qual Expectation"
].shift(1)

_, axs = plt.subplots(3, 5, figsize=(30, 20))
axs = axs.flatten()

_, axs2 = plt.subplots(3, 5, figsize=(30, 20))
axs2 = axs2.flatten()

for max_stock in list(range(MAX_RATIONAL_STOCK)):
    print(max_stock)
    data = df_personas.copy()

    CONDITIONS = [
        # Rational and accurate
        (data["finalStock"] <= max_stock)
        & (data["previous_expectation"] <= QUALITATIVE_EXPECTATION_THRESHOLD),
        # Rational and pessimistic
        (data["finalStock"] > max_stock)
        & (data["previous_expectation"] > QUALITATIVE_EXPECTATION_THRESHOLD),
        # Irrational and money illusioned
        (data["finalStock"] <= max_stock)
        & (data["previous_expectation"] > QUALITATIVE_EXPECTATION_THRESHOLD),
        # Irrational and death averse
        (data["finalStock"] > max_stock)
        & (data["previous_expectation"] <= QUALITATIVE_EXPECTATION_THRESHOLD),
    ]

    data[f"persona_horizon_{max_stock}"] = np.select(
        condlist=CONDITIONS, choicelist=PERSONAS
    )
    data = data[data["Month"].isin([12])]

    print(data.value_counts(f"persona_horizon_{max_stock}"))

    sns.histplot(
        data, x=f"persona_horizon_{max_stock}", ax=axs[max_stock], stat="percent"
    )
    sns.scatterplot(
        data,
        x="finalStock",
        y="previous_expectation",
        hue=f"persona_horizon_{max_stock}",
        ax=axs2[max_stock],
    )
