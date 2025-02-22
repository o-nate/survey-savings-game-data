# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

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

PERSONAS = ["AC", "AI", "IC", "II"]
ANNUAL_INTEREST_RATE = ((1 + INTEREST_RATE) ** 12 - 1) * 100
MEASURE = "finalStock"

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

print(df_decisions.shape)
df_decisions.head()

# %% [markdown]
## Classify subjects per accurate vs. inaccurate estimate and coherent vs. incoherent decision

# %% [markdown]
### Perceptions

MAX_RATIONAL_STOCK = 0
MONTH = 12
ESTIMATE = "Quant Perception"

df_personas = df_decisions[df_decisions["Month"] == MONTH]

data = df_personas.copy()

CONDITIONS = [
    # Accurate estimate & Coherent decision
    (data[ESTIMATE] <= ANNUAL_INTEREST_RATE)
    & (data["finalStock"] <= MAX_RATIONAL_STOCK),
    # Accurate estimate & Incoherent decision
    (data[ESTIMATE] <= ANNUAL_INTEREST_RATE)
    & (data["finalStock"] > MAX_RATIONAL_STOCK),
    # Inaccurate estimate & Coherent decision
    (data[ESTIMATE] > ANNUAL_INTEREST_RATE) & (data["finalStock"] > MAX_RATIONAL_STOCK),
    # Inaccurate estimate & Incoherent decision
    (data[ESTIMATE] > ANNUAL_INTEREST_RATE)
    & (data["finalStock"] <= MAX_RATIONAL_STOCK),
]

data["perception_pattern_12"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default="N/A"
)

# * Add column for persona based on MAX_RATIONAL_STOCK to track how distribution changes
df_personas = df_personas.merge(data, how="left")

print(df_decisions.shape)

print(data[data["participant.round"] == 1].value_counts("perception_pattern_12"))

# %%
MEASURES = [ESTIMATE] + [MEASURE]
df_personas[df_personas["participant.round"] == 1].dropna().groupby(
    ["perception_pattern_12"]
)[MEASURES].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(
    df_personas, "perception_pattern_12", MEASURES, figsize=(20, 10)
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
### Expectations

MAX_RATIONAL_STOCK = 0
MONTH = 12
ESTIMATE = "Quant Expectation"

df_personas = df_decisions[df_decisions["Month"].isin([1, MONTH])]
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    ESTIMATE
].shift(1)

CONDITIONS = [
    # Accurate estimate & Coherent decision
    (df_personas["previous_expectation"] <= ANNUAL_INTEREST_RATE)
    & (df_personas["finalStock"] <= MAX_RATIONAL_STOCK),
    # Accurate estimate & Incoherent decision
    (df_personas["previous_expectation"] <= ANNUAL_INTEREST_RATE)
    & (df_personas["finalStock"] > MAX_RATIONAL_STOCK),
    # Inaccurate estimate & Coherent decision"
    (df_personas["previous_expectation"] > ANNUAL_INTEREST_RATE)
    & (df_personas["finalStock"] > MAX_RATIONAL_STOCK),
    # Inaccurate estimate & Incoherent decision
    (df_personas["previous_expectation"] > ANNUAL_INTEREST_RATE)
    & (df_personas["finalStock"] <= MAX_RATIONAL_STOCK),
]

df_personas[f"quant_expectation_pattern_12"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default=np.nan
)

print(df_decisions.shape)

# %%
MEASURES = ["previous_expectation"] + [MEASURE]
df_personas[df_personas["participant.round"] == 1].dropna().groupby(
    ["quant_expectation_pattern_12"]
)[MEASURES].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(
    df_personas, "quant_expectation_pattern_12", MEASURES
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
#### Repeat with qualitative expectations

MAX_RATIONAL_STOCK = 0
MONTH = 12
QUALITATIVE_EXPECTATION_THRESHOLD = 1
ESTIMATE = "Qual Expectation"

df_personas = df_decisions[df_decisions["Month"].isin([1, MONTH])]
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    ESTIMATE
].shift(1)

data = df_personas.copy()

CONDITIONS = [
    # Accurate estimate & Coherent decision
    (data["previous_expectation"] <= QUALITATIVE_EXPECTATION_THRESHOLD)
    & (data["finalStock"] <= MAX_RATIONAL_STOCK),
    # Accurate estimate & Incoherent decision
    (data["previous_expectation"] <= QUALITATIVE_EXPECTATION_THRESHOLD)
    & (data["finalStock"] > MAX_RATIONAL_STOCK),
    # Inaccurate estimate & Coherent decision"
    (data["previous_expectation"] > QUALITATIVE_EXPECTATION_THRESHOLD)
    & (data["finalStock"] > MAX_RATIONAL_STOCK),
    # Inaccurate estimate & Incoherent decision
    (data["previous_expectation"] > QUALITATIVE_EXPECTATION_THRESHOLD)
    & (data["finalStock"] <= MAX_RATIONAL_STOCK),
]

data[f"qual_expectation_pattern_12"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default=np.nan
)

# * Add column for persona based on MAX_RATIONAL_STOCK to track how distribution changes
df_personas = df_personas.merge(data, how="left")

print(df_decisions.shape)

# %%
MEASURES = ["previous_expectation"] + [MEASURE]
df_personas[df_personas["participant.round"] == 1].dropna().groupby(
    ["qual_expectation_pattern_12"]
)[MEASURES].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(df_personas, "qual_expectation_pattern_12", MEASURES)
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
CHANGE_IN_STOCK = 1
ESTIMATE = "Quant Expectation"

df_personas = df_decisions[df_decisions["Month"].isin([START_MONTH, END_MONTH])]
df_personas["previous_stock"] = df_personas.groupby("participant.code")[
    "finalStock"
].shift(1)
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    ESTIMATE
].shift(1)
df_personas = df_personas[df_personas["Month"] == END_MONTH]
df_personas["change_in_stock"] = (
    df_personas["finalStock"] - df_personas["previous_stock"]
)


CONDITIONS = [
    # Accurate estimate & Coherent decision
    (df_personas[ESTIMATE] > ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] > CHANGE_IN_STOCK),
    # Accurate estimate & Incoherent decision
    (df_personas[ESTIMATE] > ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] <= CHANGE_IN_STOCK),
    # Inaccurate estimate & Coherent decision
    (df_personas[ESTIMATE] <= ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] <= CHANGE_IN_STOCK),
    # Inaccurate estimate & Incoherent decision
    (df_personas[ESTIMATE] <= ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] > CHANGE_IN_STOCK),
]

df_personas["quant_expectation_pattern_36"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default="N/A"
)

print(df_decisions.shape)

# * Drop too N/A subjects
df_personas = df_personas[df_personas["quant_expectation_pattern_36"] != "N/A"]

print(
    df_personas[df_personas["participant.round"] == 1].value_counts(
        "quant_expectation_pattern_36"
    )
)

# %%
MEASURES = [ESTIMATE] + [MEASURE]
df_personas.groupby(["quant_expectation_pattern_36", "treatment", "participant.round"])[
    MEASURES
].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(
    df_personas, "quant_expectation_pattern_36", MEASURES
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
### Perceptions
END_MONTH = 48
START_MONTH = 30
CHANGE_IN_STOCK = 1
ESTIMATE = "Quant Perception"

df_personas = df_decisions[df_decisions["Month"].isin([START_MONTH, END_MONTH])]
df_personas["previous_stock"] = df_personas.groupby("participant.code")[
    "finalStock"
].shift(1)
df_personas = df_personas[df_personas["Month"] == END_MONTH]
df_personas["change_in_stock"] = (
    df_personas["finalStock"] - df_personas["previous_stock"]
)


CONDITIONS = [
    # Accurate estimate & Coherent decision
    (df_personas[ESTIMATE] > ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] > CHANGE_IN_STOCK),
    # Accurate estimate & Incoherent decision
    (df_personas[ESTIMATE] > ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] <= CHANGE_IN_STOCK),
    # Inaccurate estimate & Coherent decision
    (df_personas[ESTIMATE] <= ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] <= CHANGE_IN_STOCK),
    # Inaccurate estimate & Incoherent decision
    (df_personas[ESTIMATE] <= ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] > CHANGE_IN_STOCK),
]

df_personas["quant_perception_pattern_48"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default="N/A"
)

# * Drop too N/A subjects
df_personas = df_personas[df_personas["quant_perception_pattern_48"] != "N/A"]

print(
    df_personas[df_personas["participant.round"] == 1].value_counts(
        "quant_perception_pattern_48"
    )
)

# %%
MEASURES = [ESTIMATE] + [MEASURE]
df_personas.groupby(["quant_perception_pattern_48", "treatment", "participant.round"])[
    MEASURES
].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(df_personas, "quant_perception_pattern_48", MEASURES)
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
QUALITATIVE_EXPECTATION_THRESHOLD = 1
ESTIMATE = "Qual Expectation"

df_personas = df_decisions[df_decisions["Month"].isin([START_MONTH, END_MONTH])]
df_personas["previous_stock"] = df_personas.groupby("participant.code")[
    "finalStock"
].shift(1)
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    ESTIMATE
].shift(1)
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

df_personas["qual_expectation_pattern_36"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default="N/A"
)

# * Add column for persona based on stock to track how distribution change

# * Drop too N/A subjects
df_personas = df_personas[df_personas["qual_expectation_pattern_36"] != "N/A"]

print(
    df_personas[df_personas["participant.round"] == 1].value_counts(
        "qual_expectation_pattern_36"
    )
)

# %%
MEASURES = [ESTIMATE] + [MEASURE]
df_personas[df_personas["participant.round"] == 1].dropna().groupby(
    ["qual_expectation_pattern_36"]
)[MEASURES].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(df_personas, "qual_expectation_pattern_36", MEASURES)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
## Regressions
MONTHS = [1] + [m * 12 for m in range(1, 4)]
df_regression = df_decisions[df_decisions["Month"].isin(MONTHS)]
df_regression["previous_stock"] = df_regression.groupby("participant.code")[
    "finalStock"
].shift(1)

models = {
    "Quantitative Perception": "finalStock ~ Actual + Quant_Perception + previous_stock + Month",
    "Quantitative Expectation": "finalStock ~ Actual + Quant_Expectation + previous_stock + Month",
    "Qualitative Perception": "finalStock ~ Actual + Qual_Perception + previous_stock + Month",
    "Qualitative Expectation": "finalStock ~ Actual + Qual_Expectation + previous_stock + Month",
}

regressions = {}

data = df_regression[df_regression["participant.round"] == 1]
data = data.rename(
    columns={
        "Quant Perception": "Quant_Perception",
        "Quant Expectation": "Quant_Expectation",
        "Qual Perception": "Qual_Perception",
        "Qual Expectation": "Qual_Expectation",
    },
)

for estimate, formula in models.items():
    model = smf.ols(formula=formula, data=data)
    regressions[estimate] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results
