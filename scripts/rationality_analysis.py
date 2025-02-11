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

MAX_RATIONAL_STOCK = 15
MONTH = 12
PERSONAS = [
    "Rational & Accurate",
    "Rational & Pessimistic",
    "Irrational & Money Illusioned",
    "Irrational & Death Averse",
]  # ["RA", "RP", "IM", "ID"]
ANNUAL_INTEREST_RATE = ((1 + INTEREST_RATE) ** 12 - 1) * 100

df_personas = df_decisions[df_decisions["Month"].isin([1, MONTH])]
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    "Quant Expectation"
].shift(1)

_, axs = plt.subplots(3, 5, figsize=(30, 20))
axs = axs.flatten()

# _, axs2 = plt.subplots(3, 5, figsize=(30, 20))
# axs2 = axs2.flatten()

for max_stock in list(range(MAX_RATIONAL_STOCK)):
    data = df_personas.copy()

    CONDITIONS = [
        # Rational and accurate
        (data["finalStock"] <= max_stock)
        & (data["previous_expectation"] <= ANNUAL_INTEREST_RATE),
        # Rational and pessimistic
        (data["finalStock"] > max_stock)
        & (data["previous_expectation"] > ANNUAL_INTEREST_RATE),
        # Irrational and money illusioned
        (data["finalStock"] <= max_stock)
        & (data["previous_expectation"] > ANNUAL_INTEREST_RATE),
        # Irrational and death averse
        (data["finalStock"] > max_stock)
        & (data["previous_expectation"] <= ANNUAL_INTEREST_RATE),
    ]

    data[f"persona_horizon_{max_stock}"] = np.select(
        condlist=CONDITIONS, choicelist=PERSONAS, default=np.nan
    )

    # * Add column for persona based on max_stock to track how distribution changes
    df_personas = df_personas.merge(data, how="left")

    data = data[data["Month"].isin([MONTH])]

    print(data.value_counts(f"persona_horizon_{max_stock}"))

    sns.histplot(
        data,
        x=f"persona_horizon_{max_stock}",
        ax=axs[max_stock],
        stat="percent",
        hue="participant.round",
    )
    # sns.scatterplot(
    #     data,
    #     x="finalStock",
    #     y="previous_expectation",
    #     hue=f"persona_horizon_{max_stock}",
    #     ax=axs2[max_stock],
    #     style="participant.round",
    # )

# %%
MEASURES = ["previous_expectation", "finalStock", "finalSavings_120"]
df_personas.dropna().groupby(["persona_horizon_0", "treatment", "participant.round"])[
    MEASURES
].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(df_personas, "persona_horizon_0", MEASURES)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
#### Repeat with 12 months after first shock
MONTH = 36
OPTIMAL_STOCK = 84
MIN_OPTIMAL_STOCK = 74
MAX_OPTIMISTIC_STOCK = 12  # Rational stock for an accurate estimator who doesn't anticipate long inflation phase
MAX_OPTIMAL_STOCK_BEFORE = (
    12  # Margin for error in amount of stock accumlate by month t = 24
)
PERSONAS = [
    "Good decision, good anticipation",
    "Bad decision, good anticipation",
    "Good decision, bad anticipation",
    "Bad decision (high), bad anticipation (low)",
    "Bad decision (low), bad anticipation (low)",
    "Death averse",
]  # ["RA", "RP", "IM", "ID"]
ANNUAL_INTEREST_RATE = ((1 + INTEREST_RATE) ** 12 - 1) * 100

df_personas = df_decisions[
    df_decisions["Month"].isin([MONTH - 12, 30, MONTH])
]  # ! Include t=30 for stock right before shock
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    "Quant Expectation"
].shift(2)
df_personas["previous_stock"] = df_personas.groupby("participant.code")[
    "finalStock"
].shift(1)

_, axs = plt.subplots(1, 1, figsize=(50, 20))


CONDITIONS = [
    # Good decision, good anticipation
    (
        (MIN_OPTIMAL_STOCK <= df_personas["finalStock"])
        & (df_personas["finalStock"] <= OPTIMAL_STOCK)
    )
    & (df_personas["previous_stock"] < MAX_OPTIMAL_STOCK_BEFORE)
    & (df_personas["previous_expectation"] > ANNUAL_INTEREST_RATE),
    # Bad decision, good anticipation
    (
        (df_personas["finalStock"] < MAX_OPTIMISTIC_STOCK)
        | (df_personas["finalStock"] < MIN_OPTIMAL_STOCK)
    )
    & (df_personas["previous_expectation"] > ANNUAL_INTEREST_RATE),
    # Good decision, bad anticipation
    (df_personas["finalStock"] <= MAX_OPTIMISTIC_STOCK)
    & (df_personas["previous_expectation"] <= ANNUAL_INTEREST_RATE),
    # Bad decision (high), bad anticipation (low)
    (
        (MAX_OPTIMISTIC_STOCK < df_personas["finalStock"])
        & (df_personas["finalStock"] >= MIN_OPTIMAL_STOCK)
    )
    & (df_personas["previous_expectation"] <= ANNUAL_INTEREST_RATE),
    # Bad decision (low), bad anticipation (low)
    (
        (MAX_OPTIMISTIC_STOCK < df_personas["finalStock"])
        & (df_personas["finalStock"] < MIN_OPTIMAL_STOCK)
    )
    & (df_personas["previous_expectation"] <= ANNUAL_INTEREST_RATE),
    # Death averse
    df_personas["finalStock"] > OPTIMAL_STOCK,
]

df_personas["persona_post_shock"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default="N/A"
)

# * Add column for persona based on stock to track how distribution changes
df_personas = df_personas.merge(df_personas, how="left")

df_personas = df_personas[df_personas["Month"].isin([MONTH])]

# * Drop too N/A subjects
df_personas = df_personas[df_personas["persona_post_shock"] != "N/A"]

print(
    df_personas[df_personas["participant.round"] == 1].value_counts(
        "persona_post_shock"
    )
)

sns.histplot(
    df_personas,
    x="persona_post_shock",
    # ax=axs[stock],
    stat="percent",
    hue="participant.round",
    legend=True,
)

# %%
MEASURES = ["previous_expectation", "finalStock", "finalSavings_120"]
df_personas.dropna().groupby(["persona_post_shock", "treatment", "participant.round"])[
    MEASURES
].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(df_personas, "persona_post_shock", MEASURES)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
### Repeat with perceptions

MAX_RATIONAL_STOCK = 15
MONTH = 12
PERSONAS = [
    "Rational & Accurate",
    "Rational & Pessimistic",
    "Irrational & Money Illusioned",
    "Irrational & Death Averse",
]  # ["RA", "RP", "IM", "ID"]
ANNUAL_INTEREST_RATE = ((1 + INTEREST_RATE) ** 12 - 1) * 100

df_personas = df_decisions[df_decisions["Month"] == MONTH]

_, axs = plt.subplots(3, 5, figsize=(30, 20))
axs = axs.flatten()

# _, axs2 = plt.subplots(3, 5, figsize=(30, 20))
# axs2 = axs2.flatten()

for max_stock in list(range(MAX_RATIONAL_STOCK)):
    data = df_personas.copy()

    CONDITIONS = [
        # Rational and accurate
        (data["finalStock"] <= max_stock)
        & (data["Quant Perception"] <= ANNUAL_INTEREST_RATE),
        # Rational and pessimistic
        (data["finalStock"] > max_stock)
        & (data["Quant Perception"] > ANNUAL_INTEREST_RATE),
        # Irrational and money illusioned
        (data["finalStock"] <= max_stock)
        & (data["Quant Perception"] > ANNUAL_INTEREST_RATE),
        # Irrational and death averse
        (data["finalStock"] > max_stock)
        & (data["Quant Perception"] <= ANNUAL_INTEREST_RATE),
    ]

    data[f"persona_horizon_{max_stock}"] = np.select(
        condlist=CONDITIONS, choicelist=PERSONAS, default=np.nan
    )

    # * Add column for persona based on max_stock to track how distribution changes
    df_personas = df_personas.merge(data, how="left")

    print(data.value_counts(f"persona_horizon_{max_stock}"))

    sns.histplot(
        data,
        x=f"persona_horizon_{max_stock}",
        ax=axs[max_stock],
        stat="percent",
        hue="participant.round",
    )
    # sns.scatterplot(
    #     data,
    #     x="finalStock",
    #     y="Quant Perception",
    #     hue=f"persona_horizon_{max_stock}",
    #     ax=axs2[max_stock],
    #     style="participant.round",
    # )
# %%
MEASURES = ["Quant Perception", "finalStock", "finalSavings_120"]
df_personas.dropna().groupby(["persona_horizon_0", "treatment", "participant.round"])[
    MEASURES
].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(df_personas, "persona_horizon_0", MEASURES)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
#### Repeat with 12 months after first shock
MONTH = 36
OPTIMAL_STOCK = 84
MIN_OPTIMAL_STOCK = 74
MAX_OPTIMISTIC_STOCK = 12  # Rational stock for an accurate estimator who doesn't anticipate long inflation phase
MAX_OPTIMAL_STOCK_BEFORE = (
    12  # Margin for error in amount of stock accumlate by month t = 24
)
PERSONAS = [
    "GD GP",  # "Good decision, good perception",
    "BD GP",  # "Bad decision, good perception",
    "GD BP",  # "Good decision, bad perception",
    "BDH, BPL",  # "Bad decision (high), bad perception (low)",
    "BDL BPL",  # "Bad decision (low), bad perception (low)",
    "DA",  # "Death averse",
]  # ["RA", "RP", "IM", "ID"]
ANNUAL_INTEREST_RATE = ((1 + INTEREST_RATE) ** 12 - 1) * 100

df_personas = df_decisions[
    df_decisions["Month"].isin([30, MONTH])
]  # ! Include t=30 for stock right before shock
df_personas["previous_stock"] = df_personas.groupby("participant.code")[
    "finalStock"
].shift(1)

_, axs = plt.subplots(1, 1, figsize=(50, 20))


CONDITIONS = [
    # Good decision, good anticipation
    (
        (MIN_OPTIMAL_STOCK <= df_personas["finalStock"])
        & (df_personas["finalStock"] <= OPTIMAL_STOCK)
    )
    & (df_personas["previous_stock"] < MAX_OPTIMAL_STOCK_BEFORE)
    & (df_personas["Quant Perception"] > ANNUAL_INTEREST_RATE),
    # Bad decision, good perception
    (
        (df_personas["finalStock"] < MAX_OPTIMISTIC_STOCK)
        | (df_personas["finalStock"] < MIN_OPTIMAL_STOCK)
    )
    & (df_personas["Quant Perception"] > ANNUAL_INTEREST_RATE),
    # Good decision, bad perception
    (df_personas["finalStock"] <= MAX_OPTIMISTIC_STOCK)
    & (df_personas["Quant Perception"] <= ANNUAL_INTEREST_RATE),
    # Bad decision (high), bad perception (low)
    (
        (MAX_OPTIMISTIC_STOCK < df_personas["finalStock"])
        & (df_personas["finalStock"] >= MIN_OPTIMAL_STOCK)
    )
    & (df_personas["previous_stock"] < MAX_OPTIMAL_STOCK_BEFORE)
    & (df_personas["Quant Perception"] <= ANNUAL_INTEREST_RATE),
    # Bad decision (low), bad perception (low)
    (
        (MAX_OPTIMISTIC_STOCK < df_personas["finalStock"])
        & (df_personas["finalStock"] < MIN_OPTIMAL_STOCK)
    )
    & (df_personas["previous_stock"] < MAX_OPTIMAL_STOCK_BEFORE)
    & (df_personas["Quant Perception"] <= ANNUAL_INTEREST_RATE),
    # Death averse
    df_personas["finalStock"] > OPTIMAL_STOCK,
]

df_personas["persona_post_shock"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default="N/A"
)

# * Add column for persona based on stock to track how distribution changes
df_personas = df_personas.merge(df_personas, how="left")

df_personas = df_personas[df_personas["Month"].isin([MONTH])]

# # * Drop too N/A subjects
# df_personas = df_personas[df_personas["persona_post_shock"] != "N/A"]

print(
    df_personas[df_personas["participant.round"] == 1].value_counts(
        "persona_post_shock"
    )
)

sns.histplot(
    df_personas,
    x="persona_post_shock",
    # ax=axs[stock],
    stat="percent",
    hue="participant.round",
    legend=True,
)

# %%
MEASURES = ["Quant Perception", "finalStock", "finalSavings_120"]
df_personas.dropna().groupby(["persona_post_shock", "treatment", "participant.round"])[
    MEASURES
].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(df_personas, "persona_post_shock", MEASURES)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# # %%
# # * Repeat with qualitative expectations

# QUALITATIVE_EXPECTATION_THRESHOLD = 1

# df_personas = df_decisions[
#     (df_decisions["Month"].isin([1, 12])) & (df_decisions["participant.round"] == 1)
# ]
# df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
#     "Qual Expectation"
# ].shift(1)

# _, axs = plt.subplots(3, 5, figsize=(30, 20))
# axs = axs.flatten()

# _, axs2 = plt.subplots(3, 5, figsize=(30, 20))
# axs2 = axs2.flatten()

# for max_stock in list(range(MAX_RATIONAL_STOCK)):
#     print(max_stock)
#     data = df_personas.copy()

#     CONDITIONS = [
#         # Rational and accurate
#         (data["finalStock"] <= max_stock)
#         & (data["previous_expectation"] <= QUALITATIVE_EXPECTATION_THRESHOLD),
#         # Rational and pessimistic
#         (data["finalStock"] > max_stock)
#         & (data["previous_expectation"] > QUALITATIVE_EXPECTATION_THRESHOLD),
#         # Irrational and money illusioned
#         (data["finalStock"] <= max_stock)
#         & (data["previous_expectation"] > QUALITATIVE_EXPECTATION_THRESHOLD),
#         # Irrational and death averse
#         (data["finalStock"] > max_stock)
#         & (data["previous_expectation"] <= QUALITATIVE_EXPECTATION_THRESHOLD),
#     ]

#     data[f"persona_horizon_{max_stock}"] = np.select(
#         condlist=CONDITIONS, choicelist=PERSONAS
#     )
#     data = data[data["Month"].isin([12])]

#     print(data.value_counts(f"persona_horizon_{max_stock}"))

#     sns.histplot(
#         data, x=f"persona_horizon_{max_stock}", ax=axs[max_stock], stat="percent"
#     )
#     sns.scatterplot(
#         data,
#         x="finalStock",
#         y="previous_expectation",
#         hue=f"persona_horizon_{max_stock}",
#         ax=axs2[max_stock],
#     )
