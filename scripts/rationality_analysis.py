# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import calc_opp_costs, process_survey

from src.utils.constants import INTEREST_RATE
from src.utils.logging_config import get_logger
from src.utils.plotting import visualize_persona_results

# * Set logger
logger = get_logger(__name__)

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# * Decimal rounding
pd.set_option("display.float_format", lambda x: "%.2f" % x)

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

# * Add uncertainty measure
df_inf_measures["Uncertain Expectation"] = process_survey.include_uncertainty_measure(
    df_inf_measures, "Quant Expectation", 1, 0
)
df_inf_measures["Average Uncertain Expectation"] = df_inf_measures.groupby(
    "participant.code"
)["Uncertain Expectation"].transform("mean")

# %%
df_decisions = df_opp_cost.merge(df_inf_measures, how="left")

# * Store final savings at month t = 120
df_decisions["finalSavings_120"] = (
    df_decisions[df_decisions["Month"] == 120]
    .groupby("participant.code")["finalSavings"]
    .transform("mean")
)
df_decisions["finalSavings_120"] = df_decisions.groupby("participant.code")[
    "finalSavings_120"
].bfill()

df_decisions.head()

# %%
# * Classify subjects as Rational-Accurate, Rational-Pessimitic, Irrational-MoneyIllusion, Irrational-DeathAverse

MAX_RATIONAL_STOCK = 0
MONTH = 12
PERSONAS = ["GECD", "GEID", "BECD", "BEIC"]
ANNUAL_INTEREST_RATE = ((1 + INTEREST_RATE) ** 12 - 1) * 100
ESTIMATE = "Quant Expectation"

df_personas = df_decisions[df_decisions["Month"].isin([1, MONTH])]
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    ESTIMATE
].shift(1)

data = df_personas.copy()

CONDITIONS = [
    # Good expectation & coherent decision
    (data["previous_expectation"] <= ANNUAL_INTEREST_RATE)
    & (data["finalStock"] <= MAX_RATIONAL_STOCK),
    # Good expectation & Incoherent decision
    (data["previous_expectation"] <= ANNUAL_INTEREST_RATE)
    & (data["finalStock"] > MAX_RATIONAL_STOCK),
    # Bad expectation & coherent decision"
    (data["previous_expectation"] > ANNUAL_INTEREST_RATE)
    & (data["finalStock"] > MAX_RATIONAL_STOCK),
    # Bad expectation & Bad decision
    (data["previous_expectation"] > ANNUAL_INTEREST_RATE)
    & (data["finalStock"] <= MAX_RATIONAL_STOCK),
]

data[f"persona_start"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default=np.nan
)

# * Add column for persona based on MAX_RATIONAL_STOCK to track how distribution changes
df_personas = df_personas.merge(data, how="left")

data = data[data["Month"].isin([MONTH])]

print(data[data["participant.round"] == 1].value_counts(f"persona_start"))

# %%
MEASURES = ["previous_expectation", "finalStock", "finalSavings_120"]
df_personas.dropna().groupby(["persona_start", "treatment", "participant.round"])[
    MEASURES
].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(df_personas, "persona_start", MEASURES)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
#### Repeat with qualitative expectations

MAX_RATIONAL_STOCK = 0
MONTH = 12
PERSONAS = ["GECD", "GEID", "BECD", "BEIC"]
QUALITATIVE_EXPECTATION_THRESHOLD = 1
ESTIMATE = "Qual Expectation"

df_personas = df_decisions[df_decisions["Month"].isin([1, MONTH])]
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    ESTIMATE
].shift(1)

data = df_personas.copy()

CONDITIONS = [
    # Good expectation & coherent decision
    (data["previous_expectation"] <= QUALITATIVE_EXPECTATION_THRESHOLD)
    & (data["finalStock"] <= MAX_RATIONAL_STOCK),
    # Good expectation & Incoherent decision
    (data["previous_expectation"] <= QUALITATIVE_EXPECTATION_THRESHOLD)
    & (data["finalStock"] > MAX_RATIONAL_STOCK),
    # Bad expectation & coherent decision"
    (data["previous_expectation"] > QUALITATIVE_EXPECTATION_THRESHOLD)
    & (data["finalStock"] > MAX_RATIONAL_STOCK),
    # Bad expectation & Bad decision
    (data["previous_expectation"] > QUALITATIVE_EXPECTATION_THRESHOLD)
    & (data["finalStock"] <= MAX_RATIONAL_STOCK),
]

data[f"persona_start"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default=np.nan
)

# * Add column for persona based on MAX_RATIONAL_STOCK to track how distribution changes
df_personas = df_personas.merge(data, how="left")

data = data[data["Month"].isin([MONTH])]

print(data[data["participant.round"] == 1].value_counts(f"persona_start"))

# %%
MEASURES = ["previous_expectation", "finalStock", "finalSavings_120"]
df_personas.dropna().groupby(["persona_start", "treatment", "participant.round"])[
    MEASURES
].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(df_personas, "persona_start", MEASURES)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
#### Repeat with 12 months after first shock
# $(\Delta{\text{S}_{36}} > 0, E_{36} > 25)$ => Good expectations, coherent decision
# $(\Delta{\text{S}_{36}} <= 0, E_{36} > 25)$ => Good expectations, incoherent decision
# $(\Delta{\text{S}_{36}} <= 0, E_{36} <= 25)$ => Bad expectations, coherent decision
# $(\Delta{\text{S}_{36}} > 0, E_{36} <= 25)$ => Bad expectations, incoherent decision

# %%
END_MONTH = 36
START_MONTH = END_MONTH - 12
CHANGE_IN_STOCK = 0
PERSONAS = ["GECD", "GEID", "BECD", "BEIC"]
ANNUAL_INTEREST_RATE = ((1 + INTEREST_RATE) ** 12 - 1) * 100
ESTIMATE = "Quant Expectation"

df_personas = df_decisions[
    df_decisions["Month"].isin([START_MONTH, START_MONTH + 6, END_MONTH])
]
df_personas["previous_stock"] = df_personas.groupby("participant.code")[
    "finalStock"
].shift(2)
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    ESTIMATE
].shift(2)
df_personas = df_personas[df_personas["Month"] == END_MONTH]
df_personas["change_in_stock"] = (
    df_personas["finalStock"] - df_personas["previous_stock"]
)


CONDITIONS = [
    # Good expectations, coherent decision
    (df_personas[ESTIMATE] > ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] > CHANGE_IN_STOCK),
    # Good expectations, incoherent decision
    (df_personas[ESTIMATE] > ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] <= CHANGE_IN_STOCK),
    # Bad expectations, coherent decision
    (df_personas[ESTIMATE] <= ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] <= CHANGE_IN_STOCK),
    # Bad expectations, incoherent decision
    (df_personas[ESTIMATE] <= ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] > CHANGE_IN_STOCK),
]

df_personas["persona_post_shock"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default="N/A"
)

# * Add column for persona based on stock to track how distribution changes
df_personas = df_personas.merge(df_personas, how="left")

# * Drop too N/A subjects
df_personas = df_personas[df_personas["persona_post_shock"] != "N/A"]

print(
    df_personas[df_personas["participant.round"] == 1].value_counts(
        "persona_post_shock"
    )
)

# %%
MEASURES = [ESTIMATE, "finalStock", "finalSavings_120"]
df_personas[df_personas["participant.round"] == 1].dropna().groupby(
    ["persona_post_shock"]
)[MEASURES].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(df_personas, "persona_post_shock", MEASURES)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
#### Repeat with qualitative expectations
# $(\Delta{\text{S}_{36}} > 0, E_{36} > 1)$ => Good expectations, coherent decision
# $(\Delta{\text{S}_{36}} <= 0, E_{36} > 1)$ => Good expectations, incoherent decision
# $(\Delta{\text{S}_{36}} <= 0, E_{36} <= 1)$ => Bad expectations, coherent decision
# $(\Delta{\text{S}_{36}} > 0, E_{36} <= 1)$ => Bad expectations, incoherent decision

# %%
END_MONTH = 36
START_MONTH = END_MONTH - 12
CHANGE_IN_STOCK = 0
PERSONAS = ["GECD", "GEID", "BECD", "BEIC"]
QUALITATIVE_EXPECTATION_THRESHOLD = 1
ESTIMATE = "Qual Expectation"

df_personas = df_decisions[
    df_decisions["Month"].isin([START_MONTH, START_MONTH + 6, END_MONTH])
]
df_personas["previous_stock"] = df_personas.groupby("participant.code")[
    "finalStock"
].shift(2)
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    ESTIMATE
].shift(2)
df_personas = df_personas[df_personas["Month"] == END_MONTH]
df_personas["change_in_stock"] = (
    df_personas["finalStock"] - df_personas["previous_stock"]
)


CONDITIONS = [
    # Good expectations, coherent decision
    (df_personas[ESTIMATE] > QUALITATIVE_EXPECTATION_THRESHOLD)
    & (df_personas["change_in_stock"] > CHANGE_IN_STOCK),
    # Good expectations, incoherent decision
    (df_personas[ESTIMATE] > QUALITATIVE_EXPECTATION_THRESHOLD)
    & (df_personas["change_in_stock"] <= CHANGE_IN_STOCK),
    # Bad expectations, coherent decision
    (df_personas[ESTIMATE] <= QUALITATIVE_EXPECTATION_THRESHOLD)
    & (df_personas["change_in_stock"] <= CHANGE_IN_STOCK),
    # Bad expectations, incoherent decision
    (df_personas[ESTIMATE] <= QUALITATIVE_EXPECTATION_THRESHOLD)
    & (df_personas["change_in_stock"] > CHANGE_IN_STOCK),
]

df_personas["persona_post_shock"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default="N/A"
)

# * Add column for persona based on stock to track how distribution changes
df_personas = df_personas.merge(df_personas, how="left")

# * Drop too N/A subjects
df_personas = df_personas[df_personas["persona_post_shock"] != "N/A"]

print(
    df_personas[df_personas["participant.round"] == 1].value_counts(
        "persona_post_shock"
    )
)

# %%
MEASURES = [ESTIMATE, "finalStock", "finalSavings_120"]
df_personas[df_personas["participant.round"] == 1].dropna().groupby(
    ["persona_post_shock"]
)[MEASURES].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(df_personas, "persona_post_shock", MEASURES)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
### Repeat with perceptions

MAX_RATIONAL_STOCK = 0
MONTH = 12
PERSONAS = ["GPCD", "BPCD", "GPID", "BPID"]
ANNUAL_INTEREST_RATE = ((1 + INTEREST_RATE) ** 12 - 1) * 100

df_personas = df_decisions[df_decisions["Month"] == MONTH]

data = df_personas.copy()

CONDITIONS = [
    # Good expectation & Good decision
    (data["finalStock"] <= MAX_RATIONAL_STOCK)
    & (data["Quant Perception"] <= ANNUAL_INTEREST_RATE),
    # ad expectation & Good decision"
    (data["finalStock"] > MAX_RATIONAL_STOCK)
    & (data["Quant Perception"] > ANNUAL_INTEREST_RATE),
    # Good expectation & Bad decision
    (data["finalStock"] > MAX_RATIONAL_STOCK)
    & (data["Quant Perception"] <= ANNUAL_INTEREST_RATE),
    # Bad expectation & Bad decision
    (data["finalStock"] <= MAX_RATIONAL_STOCK)
    & (data["Quant Perception"] > ANNUAL_INTEREST_RATE),
]

data["persona_start"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default="N/A"
)

# * Add column for persona based on MAX_RATIONAL_STOCK to track how distribution changes
df_personas = df_personas.merge(data, how="left")

print(data[data["participant.round"] == 1].value_counts("persona_start"))

# %%
MEASURES = ["Quant Perception", "finalStock", "finalSavings_120"]
df_personas.dropna().groupby(["persona_start", "treatment", "participant.round"])[
    MEASURES
].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(df_personas, "persona_start", MEASURES)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
## Regressions
MONTHS = [1] + [m * 12 for m in range(1, 4)]
df_regression = df_decisions[df_decisions["Month"].isin(MONTHS)]
df_regression["previous_stock"] = df_regression.groupby("participant.code")[
    "finalStock"
].shift(1)

_, axs = plt.subplots(3, 5, figsize=(30, 20))
axs = axs.flatten()

# _, axs2 = plt.subplots(3, 5, figsize=(30, 20))
# axs2 = axs2.flatten()

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
    # sns.scatterplot(
    #     data,
    #     x="finalStock",
    #     y="previous_expectation",
    #     hue=f"persona_horizon_{max_stock}",
    #     ax=axs2[max_stock],
    # )
