"""Present results from experiment"""

# %%
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from scripts.utils import constants

from src import (
    calc_opp_costs,
    discontinuity,
    econ_preferences,
    knowledge,
    process_survey,
)

from src.preprocess import final_df_dict
from src.stats_analysis import (
    create_bonferroni_correlation_table,
    create_pearson_correlation_matrix,
)
from src.utils.helpers import combine_series
from src.utils.logging_helpers import set_external_module_log_levels

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# %% [markdown]
## Descriptive statistics: Subjects

# %%
df_questionnaire = final_df_dict["Questionnaire"].copy()
df_questionnaire.head()

measures = [
    "age",
    "gender",
    "educationLevel",
    "employmentStatus",
    "financialStatusIncome",
    "financialStatusSavings_1",
    "financialStatusSavings_2",
    "financialStatusDebt_1",
    "financialStatusDebt_2",
]
## Investment holdings
holdings = [
    "stocks",
    "mutualFunds",
    "bonds",
    "savingsAccounts",
    "lifeInsurance",
    "retirementAccounts",
    "crypto",
]

df_questionnaire[
    [c for c in df_questionnaire if any(m in c for m in measures + holdings)]
].describe().T


# %% [markdown]
## Overall performance
df_opp_cost = calc_opp_costs.calculate_opportunity_costs()
calc_opp_costs.plot_savings_and_stock(df_opp_cost, col="phase", palette="tab10")


# %% [markdown]
## Behavior in the Savings Game

# %% [markdown]
## Performance measures: Over-, under-, and wasteful-stocking and purchase adaptation
df_measures = discontinuity.purchase_discontinuity(
    df_opp_cost, constants.DECISION_QUANTITY, constants.WINDOW
)

## Set avg_q and avg_q_% as month=33 value
df_pivot_measures = pd.pivot_table(
    df_measures[df_measures["month"] == 33][["participant.code", "avg_q", "avg_q_%"]],
    index="participant.code",
)
df_pivot_measures.reset_index(inplace=True)
df_measures = df_measures[[c for c in df_measures.columns if "avg_q" not in c]].merge(
    df_pivot_measures, how="left"
)

# %%
df_pivot_measures = df_measures[
    (df_measures["month"] == 120) & (df_measures["participant.round"] == 1)
].melt(
    id_vars="participant.label",
    value_vars=["sreal_%", "early_%", "excess_%"],
    var_name="Performance measure",
    value_name="Percent of maximum",
)
old_names = ["sreal_%", "early_%", "excess_%"]
new_names = ["Total savings", "Over-stocking", "Wasteful-stocking"]
df_pivot_measures["Performance measure"].replace(old_names, new_names, inplace=True)
df_pivot_measures["Percent of maximum"] = df_pivot_measures["Percent of maximum"] * 100
fig = sns.boxplot(
    data=df_pivot_measures,
    x="Performance measure",
    y="Percent of maximum",
)

# %%
df_pivot_measures = df_measures[
    (df_measures["month"] == 33) & (df_measures["participant.round"] == 1)
].melt(
    id_vars=["participant.label", "participant.round"],
    value_vars=["avg_q_%"],
    var_name="Performance measure",
    value_name="Percent",
)
old_names = ["avg_q_%"]
new_names = ["Purchase adaptation"]
df_pivot_measures["Performance measure"].replace(old_names, new_names, inplace=True)
fig = sns.boxplot(
    data=df_pivot_measures,
    x="Performance measure",
    y="Percent",
    hue="participant.round",
)


# %% [markdown]
## Expectation and perception of inflation

# %% [markdown]
### Quality of inflation expectations and perceptions and performance
df_survey = process_survey.create_survey_df(include_inflation=True)

# * Plot estimates over time
estimates = ["Quant Perception", "Quant Expectation", "Actual", "Upcoming"]
g = sns.relplot(
    data=df_survey[df_survey["Measure"].isin(estimates)],
    x="Month",
    y="Estimate",
    errorbar=None,
    hue="Measure",
    style="Measure",
    kind="line",
    col="participant.round",
)

## Adjust titles
(g.set_axis_labels("Month", "Inflation rate (%)").tight_layout(w_pad=0.5))
plt.show()

# %%
df_inf_measures = process_survey.pivot_inflation_measures(df_survey)
df_inf_measures = process_survey.include_inflation_measures(df_inf_measures)

df_inf_measures[df_inf_measures["participant.round"] == 1].describe().T

# %% [markdown]
#### Relationship between expectations, perceptions, and decisions
# (Difference between quantitative and qualitative estimates)
df_inf_measures.rename(columns={"Month": "month"}, inplace=True)
df_inf_measures = df_inf_measures.merge(
    df_measures[[c for c in df_measures.columns if "participant.inflation" not in c]],
    how="left",
)

# %%
## Separate inflation measures by high- and low-inflation
df_inf_measures["inf_phase"] = np.where(
    df_inf_measures["inf_phase"] == 1,
    "high",
    "low",
)

df_bias = pd.pivot_table(
    data=df_inf_measures[
        [
            "participant.code",
            "inf_phase",
            "Perception_bias",
            "Expectation_bias",
        ]
    ],
    index=["participant.code"],
    columns="inf_phase",
)
df_bias = df_bias.reset_index()
df_bias.columns = df_bias.columns.map("_".join)
df_bias.reset_index(inplace=True)
df_bias.head()

df_bias.rename(columns={"participant.code_": "participant.code"}, inplace=True)

df_inf_measures = df_inf_measures.merge(df_bias, how="left")

# %%
## Determine qualitative estimates accuracy
conditions_list = [
    (df_inf_measures["inf_phase"] == "high") & (df_inf_measures["Qual Perception"] > 1),
    (df_inf_measures["inf_phase"] == "low")
    & (df_inf_measures["Qual Perception"] >= 0)
    & (df_inf_measures["Qual Perception"] <= 1),
    df_inf_measures["Qual Perception"].isna(),
]
df_inf_measures["Qual Perception Accuracy"] = np.select(
    conditions_list, [1, 1, np.NaN], default=0
)

conditions_list = [
    (df_inf_measures.groupby("participant.code")["inf_phase"].shift(-1) == "high")
    & (df_inf_measures["Qual Expectation"] > 1),
    (df_inf_measures.groupby("participant.code")["inf_phase"].shift(-1) == "low")
    & (df_inf_measures["Qual Expectation"] >= 0)
    & (df_inf_measures["Qual Expectation"] <= 1),
    df_inf_measures["Qual Expectation"].isna(),
]

df_inf_measures["Qual Expectation Accuracy"] = np.select(
    conditions_list, [1, 1, np.NaN], default=0
)

df_accuracy = pd.pivot_table(
    df_inf_measures[
        ["participant.code", "Qual Perception Accuracy", "Qual Expectation Accuracy"]
    ],
    index="participant.code",
    aggfunc="mean",
)

df_accuracy.rename(
    columns={col: f"Avg {col}" for col in df_accuracy.columns}, inplace=True
)

df_inf_measures = df_inf_measures.merge(df_accuracy.reset_index(), how="left")

# %%
create_pearson_correlation_matrix(
    df_inf_measures[
        (df_inf_measures["participant.round"] == 1) & (df_inf_measures["month"] == 120)
    ][
        [
            "Perception_sensitivity",
            "Perception_bias_low",
            "Perception_bias_high",
            "Expectation_sensitivity",
            "Expectation_bias_low",
            "Expectation_bias_high",
            "Avg Qual Perception Accuracy",
            "Avg Qual Expectation Accuracy",
            "avg_q",
            "avg_q_%",
            "sreal",
        ]
    ],
    p_values=constants.P_VALUES_THRESHOLDS,
)


# %% [markdown]
## Real life vs. savings game

### Comparison to trends from CAMME in real life

# %% [markdown]
#### Figure I – Correlation between perceived and expected inflation (%) <br><br>
sns.lmplot(
    df_inf_measures[df_inf_measures["participant.round"] == 1],
    x="Quant Perception",
    y="Quant Expectation",
    hue="participant.round",
)

# %% [markdown]
#### Tableau 3 – Réponses à la question qualitative sur l’anticipation à un an <br><br>
df_inf_measures.groupby("inf_phase")[["Qual Expectation"]].value_counts(normalize=True)


# %% [markdown]
#### Figure III – Distribution of perceived and expected inflaiton (% of respondents)
df = df_inf_measures[df_inf_measures["participant.round"] == 1][
    ["participant.code", "inf_phase", "month", "Quant Perception", "Quant Expectation"]
].melt(
    id_vars=["participant.code", "inf_phase", "month"],
    value_vars=["Quant Perception", "Quant Expectation"],
    var_name="Estimate Type",
    value_name="Estimate",
)

# %%
sns.displot(
    df, x="Estimate", hue="inf_phase", col="Estimate Type", kde=True, common_norm=False
)

# %% [markdown]
# Figure IV – Inflation IPCH et anticipations d’inflation 2020-2021
# * Plot estimates over time
estimates = ["Quant Expectation", "Actual", "Upcoming"]
g = sns.relplot(
    data=df_survey[df_survey["participant.round"] == 1][
        df_survey["Measure"].isin(estimates)
    ],
    x="Month",
    y="Estimate",
    errorbar=None,
    hue="Measure",
    style="Measure",
    kind="line",
)

## Adjust titles
(
    g.set_axis_labels("Month", "Inflation rate (%)")
    # .set_titles("Savings Game round: {col_name}")
    .tight_layout(w_pad=0.5)
)

# %%
# Figure V – Change in estimation uncertainty (% of responses)
df_inf_measures["Uncertain Expectation"] = process_survey.include_uncertainty_measure(
    df_inf_measures, "Quant Expectation", 1, 0
)
df_inf_measures["Average Uncertain Expectation"] = df_inf_measures.groupby(
    "participant.code"
)["Uncertain Expectation"].transform("mean")
df_uncertain = (
    pd.pivot_table(
        df_inf_measures[
            [
                "month",
                "Quant Expectation",
                "Uncertain Expectation",
                "Actual",
            ]
        ],
        index="month",
        aggfunc="mean",
    )
    .reset_index()
    .dropna()
)

df_uncertain["Uncertain Expectation"] = df_uncertain["Uncertain Expectation"] * 100

sns.lineplot(
    df_uncertain.melt(
        id_vars="month",
        value_vars=["Quant Expectation", "Uncertain Expectation", "Actual"],
        var_name="Measure",
        value_name="Value",
    ),
    x="month",
    y="Value",
    hue="Measure",
)

# %% [markdown]
## The role of individual characteristics and behavior

df_knowledge = knowledge.create_knowledge_dataframe()
df_econ_preferences = econ_preferences.create_econ_preferences_dataframe()
df_individual_char = combine_series(
    [df_inf_measures, df_knowledge, df_econ_preferences],
    how="left",
    on="participant.label",
)

# %% [markdown]
### Results of knowledge tasks
df_knowledge.describe().T

# %% [markdown]
### Results of economic preference tasks
df_econ_preferences.describe().T

# %% [markdown]
### Correlations between knowledge and performance measures
create_pearson_correlation_matrix(
    df_individual_char[
        (df_individual_char["participant.round"] == 1)
        & (df_individual_char["month"] == 120)
    ][constants.KNOWLEDGE_MEASURES + constants.PERFORMANCE_MEASURES],
    p_values=constants.P_VALUES_THRESHOLDS,
)

# %%
# * Bonferroni correction
create_bonferroni_correlation_table(
    df_individual_char,
    constants.KNOWLEDGE_MEASURES,
    constants.PERFORMANCE_MEASURES,
    "pointbiserial",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between inconsistencies in economic preferences and performance measures
create_pearson_correlation_matrix(
    df_individual_char[
        (df_individual_char["participant.round"] == 1)
        & (df_individual_char["month"] == 120)
    ][constants.ECON_PREFERENCE_MEASURES + constants.PERFORMANCE_MEASURES],
    p_values=constants.P_VALUES_THRESHOLDS,
)

# %%
# * Bonferroni correction
create_bonferroni_correlation_table(
    df_individual_char,
    constants.ECON_PREFERENCE_MEASURES,
    constants.PERFORMANCE_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between knowledge and inflation bias and sensitivity measures

## Set mean perception and expectation biases
df_individual_char["avg_perception_bias"] = df_individual_char.groupby(
    "participant.code"
)["Perception_bias"].transform("mean")
df_individual_char["avg_expectation_bias"] = df_individual_char.groupby(
    "participant.code"
)["Expectation_bias"].transform("mean")

create_pearson_correlation_matrix(
    df_individual_char[
        (df_individual_char["participant.round"] == 1)
        & (df_individual_char["month"] == 120)
    ][constants.KNOWLEDGE_MEASURES + constants.QUANT_INFLATION_MEASURES],
    p_values=constants.P_VALUES_THRESHOLDS,
)

# %%
# * Bonferroni correction
create_bonferroni_correlation_table(
    df_individual_char,
    constants.KNOWLEDGE_MEASURES,
    constants.QUANT_INFLATION_MEASURES,
    "pointbiserial",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between inconsistency and inflation bias and sensitivity measures
create_pearson_correlation_matrix(
    df_individual_char[
        (df_individual_char["participant.round"] == 1)
        & (df_individual_char["month"] == 120)
    ][constants.ECON_PREFERENCE_MEASURES + constants.QUANT_INFLATION_MEASURES],
    p_values=constants.P_VALUES_THRESHOLDS,
)

# %%
# * Bonferroni correction
create_bonferroni_correlation_table(
    df_individual_char,
    constants.ECON_PREFERENCE_MEASURES,
    constants.QUANT_INFLATION_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between knowledge and inflation qualitative inflation measures
create_pearson_correlation_matrix(
    df_individual_char[
        (df_individual_char["participant.round"] == 1)
        & (df_individual_char["month"] == 120)
    ][constants.KNOWLEDGE_MEASURES + constants.QUAL_INFLATION_MEASURES],
    p_values=constants.P_VALUES_THRESHOLDS,
)

# %%
# * Bonferroni correction
create_bonferroni_correlation_table(
    df_individual_char,
    constants.KNOWLEDGE_MEASURES,
    constants.QUAL_INFLATION_MEASURES,
    "pointbiserial",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between knowledge and inflation qualitative inflation measures
create_pearson_correlation_matrix(
    df_individual_char[
        (df_individual_char["participant.round"] == 1)
        & (df_individual_char["month"] == 120)
    ][constants.ECON_PREFERENCE_MEASURES + constants.QUAL_INFLATION_MEASURES],
    p_values=constants.P_VALUES_THRESHOLDS,
)

# %%
# * Bonferroni correction
create_bonferroni_correlation_table(
    df_individual_char,
    constants.ECON_PREFERENCE_MEASURES,
    constants.QUAL_INFLATION_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
    filtered_results=False,
)


# %% [markdown]
## Efficacy of interventions

### Régression de performance
# `totale, over-stocking, excess, biais_perc, biais_antic, sens_perc ~
# avant/après * intervention_1|intervention_2|contrôle`
#
#### Analyse de médiation
#
# `Intervention → perception et anticipation → performance`
#
# `Intervention → performance`
#
### Les participants prennent-ils plus en compte l’inflation pour s’adapter ?
#
# `Régression d’adaptation (1-12, 12-24, 24-36, 36-48) ~ inflation réelle +
# perception_finaleDePeriode + anticipation_débutDePeriode + avant|après +
# intervention_1|intervention_2|contrôle`
#
### Généralité de l’efficacité
#
# `Régression diff_performance ~ intervention_1|intervention_2|contrôle *
# (toutes les caractéristiques)`
#
# Croiser une à une
