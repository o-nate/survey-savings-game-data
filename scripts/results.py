"""Present results from experiment"""

# %%
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pingouin import mediation_analysis
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.mediation import Mediation
from statsmodels.iolib.summary2 import summary_col

from scripts.utils import constants

from src import (
    calc_opp_costs,
    discontinuity,
    intervention,
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
df_questionnaire = final_df_dict["Questionnaire"].copy()

df_questionnaire[
    [
        m
        for m in df_questionnaire
        if any(qm in m for qm in constants.QUESTIONNAIRE_MEASURES)
    ]
].describe().T


# %% [markdown]
## Overall performance
# As can be seen in the graph comparing to the maximum and naïve
# strategies, the average performance is well below the maximum. Overall, the
# average performance also does not improve drastically between rounds of the
# Savings Game when taken across all treatment group together.
df_opp_cost = calc_opp_costs.calculate_opportunity_costs()
calc_opp_costs.plot_savings_and_stock(df_opp_cost, col="phase", palette="tab10")


# %% [markdown]
## Behavior in the Savings Game
### Performance measures: Over- and wasteful-stocking and purchase adaptation
# In the first round across all subjects, the average total savings as a percent of
# the maximum is 53.6%. Over- and wasteful-stocking account for 19.3% and 8.8% of
# the maximum. As can be seen in the boxplot, however, the mean wasteful-stocking
# measure is greatly skewed by outliers. Finally, average purchase adaptation (as a
# percentage of the cumulative quantity purchased leading up to the inflation
# phase-change) is 9.2%, however also with significant outliers; the median is 3.1%.

df_measures = discontinuity.purchase_discontinuity(
    df_opp_cost, constants.DECISION_QUANTITY, constants.WINDOW
)

## Set avg_q and avg_q_% as month=33 value
df_pivot_measures = pd.pivot_table(
    df_measures[df_measures["month"] == 33][["participant.code", "avg_q", "avg_q_%"]],
    index="participant.code",
)
df_pivot_measures.reset_index(inplace=True)
df_measures = df_measures[[m for m in df_measures.columns if "avg_q" not in m]].merge(
    df_pivot_measures, how="left"
)

## Rename columns for results table
df_measures.rename(
    columns={
        k: v
        for k, v in zip(
            constants.PERFORMANCE_MEASURES_OLD_NAMES
            + constants.PURCHASE_ADAPTATION_OLD_NAME,
            constants.PERFORMANCE_MEASURES_NEW_NAMES
            + constants.PURCHASE_ADAPTATION_NEW_NAME,
        )
    },
    inplace=True,
)
df_measures[(df_measures["month"] == 120) & (df_measures["phase"] == "pre")].describe()[
    constants.PERFORMANCE_MEASURES_NEW_NAMES + constants.PURCHASE_ADAPTATION_NEW_NAME
].T

# %%
df_pivot_measures = df_measures[
    (df_measures["month"] == 120) & (df_measures["participant.round"] == 1)
].melt(
    id_vars="participant.label",
    value_vars=constants.PERFORMANCE_MEASURES_NEW_NAMES
    + constants.PURCHASE_ADAPTATION_NEW_NAME,
    var_name="Performance measure",
    value_name="Percent of maximum",
)
df_pivot_measures["Percent of maximum"] = df_pivot_measures["Percent of maximum"] * 100
fig = sns.boxplot(
    data=df_pivot_measures[
        df_pivot_measures["Performance measure"].isin(
            constants.PERFORMANCE_MEASURES_NEW_NAMES
        )
    ],
    x="Performance measure",
    y="Percent of maximum",
)
# %%
df_pivot_measures["Percent of maximum"] = df_pivot_measures["Percent of maximum"] / 100
df_pivot_measures.rename(columns={"Percent of maximum": "Percent"}, inplace=True)
fig = sns.boxplot(
    data=df_pivot_measures[
        df_pivot_measures["Performance measure"].isin(
            constants.PURCHASE_ADAPTATION_NEW_NAME
        )
    ],
    x="Performance measure",
    y="Percent",
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
    df_measures[[m for m in df_measures.columns if "participant.inflation" not in m]],
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
    p_values=constants.P_VALUE_THRESHOLDS,
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
## Set mean perception and expectation biases
df_individual_char["avg_perception_bias"] = df_individual_char.groupby(
    "participant.code"
)["Perception_bias"].transform("mean")
df_individual_char["avg_expectation_bias"] = df_individual_char.groupby(
    "participant.code"
)["Expectation_bias"].transform("mean")

# %% [markdown]
### Results of knowledge tasks
df_knowledge.describe().T

# %% [markdown]
### Results of economic preference tasks
df_econ_preferences.describe().T

data = df_individual_char[
    (df_individual_char["participant.round"] == 1)
    & (df_individual_char["month"] == 120)
]

# %% [markdown]
### Correlations between knowledge and performance measures
create_pearson_correlation_matrix(
    data[constants.KNOWLEDGE_MEASURES + constants.PERFORMANCE_MEASURES],
    p_values=constants.P_VALUE_THRESHOLDS,
)

# %%
# * Bonferroni correction
create_bonferroni_correlation_table(
    data,
    constants.KNOWLEDGE_MEASURES,
    constants.PERFORMANCE_MEASURES,
    "pointbiserial",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between inconsistencies in economic preferences and performance measures
create_pearson_correlation_matrix(
    data[constants.ECON_PREFERENCE_MEASURES + constants.PERFORMANCE_MEASURES],
    p_values=constants.P_VALUE_THRESHOLDS,
)

# %%
# * Bonferroni correction
create_bonferroni_correlation_table(
    data,
    constants.ECON_PREFERENCE_MEASURES,
    constants.PERFORMANCE_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between knowledge and inflation bias and sensitivity measures

create_pearson_correlation_matrix(
    data[constants.KNOWLEDGE_MEASURES + constants.QUANT_INFLATION_MEASURES],
    p_values=constants.P_VALUE_THRESHOLDS,
)

# %%
# * Bonferroni correction
create_bonferroni_correlation_table(
    data,
    constants.KNOWLEDGE_MEASURES,
    constants.QUANT_INFLATION_MEASURES,
    "pointbiserial",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between inconsistency and inflation bias and sensitivity measures
create_pearson_correlation_matrix(
    data[constants.ECON_PREFERENCE_MEASURES + constants.QUANT_INFLATION_MEASURES],
    p_values=constants.P_VALUE_THRESHOLDS,
)

# %%
# * Bonferroni correction
create_bonferroni_correlation_table(
    data,
    constants.ECON_PREFERENCE_MEASURES,
    constants.QUANT_INFLATION_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between knowledge and inflation qualitative inflation measures
create_pearson_correlation_matrix(
    data[constants.KNOWLEDGE_MEASURES + constants.QUAL_INFLATION_MEASURES],
    p_values=constants.P_VALUE_THRESHOLDS,
)

# %%
# * Bonferroni correction
create_bonferroni_correlation_table(
    data,
    constants.KNOWLEDGE_MEASURES,
    constants.QUAL_INFLATION_MEASURES,
    "pointbiserial",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between knowledge and inflation qualitative inflation measures
create_pearson_correlation_matrix(
    data[constants.ECON_PREFERENCE_MEASURES + constants.QUAL_INFLATION_MEASURES],
    p_values=constants.P_VALUE_THRESHOLDS,
)

# %%
# * Bonferroni correction
create_bonferroni_correlation_table(
    data,
    constants.ECON_PREFERENCE_MEASURES,
    constants.QUAL_INFLATION_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
    filtered_results=False,
)


# %% [markdown]
## Efficacy of interventions
data = df_individual_char[df_individual_char["month"] == 120]

# %% [markdown]
### Change in performance between first and second session (Learning effect)
df_learning_effect = intervention.create_learning_effect_table(
    data, constants.PERFORMANCE_MEASURES, constants.P_VALUE_THRESHOLDS
)
df_learning_effect

# %% [markdown]
### Diff-in-diff of treatments
df_treatments = intervention.create_diff_in_diff_table(
    data,
    constants.PERFORMANCE_MEASURES,
    constants.TREATMENTS,
    constants.P_VALUE_THRESHOLDS,
)
df_treatments

# %% [markdown]
### Regression of performance
# `totale, over-stocking, excess, biais_perc, biais_antic, sens_perc ~
# avant/après * intervention_1|intervention_2|contrôle`
data = df_individual_char[df_individual_char["month"] == 120]

regressions = {}

for m in constants.PERFORMANCE_MEASURES[:-1] + constants.QUANT_INFLATION_MEASURES:
    model = smf.ols(
        formula=f"{m} ~ C(treatment) * C(phase)",  # No intercept (-1)
        data=data,
    )
    regressions[m] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)

# %%
results


# %% [markdown]
### Les participants prennent-ils plus en compte l’inflation pour s’adapter ?
#
# `Régression d’adaptation (1-12, 12-24, 24-36, 36-48) ~ inflation réelle +
# perception_finaleDePeriode + anticipation_débutDePeriode + avant|après +
# intervention_1|intervention_2|contrôle`

# * Get average quantity purchased in each 12-month window
df_adapt = df_opp_cost.copy()
df_adapt["avg_purchase"] = df_adapt.groupby("participant.code")["decision"].transform(
    lambda x: x.rolling(12).mean()
)
df_inf_adapt = df_individual_char.copy()
df_inf_adapt = df_inf_adapt.merge(
    df_adapt[["participant.code", "avg_purchase", "month"]], how="left"
)

# * Get previous window's inflation expectation
df_inf_adapt["previous_expectation"] = df_inf_adapt.groupby("participant.code")[
    "Quant Expectation"
].shift(1)
df_inf_adapt["previous_qual_expectation"] = df_inf_adapt.groupby("participant.code")[
    "Qual Expectation"
].shift(1)

df_inf_adapt.rename(
    columns={
        "Quant Perception": "current_perception",
        "Qual Perception": "current_qual_perception",
    },
    inplace=True,
)

# * Replace qualitative estimates with boolean for stay the same/decrease or increase
# * (see Andrade et al. (2023))
# condition = [df_inf_adapt["current_qual_perception"] <= 0]
# choice = [0]
df_inf_adapt["current_qual_perception"] = np.where(
    df_inf_adapt["current_qual_perception"] <= 0, 0, 1
)
df_inf_adapt["previous_qual_expectation"] = np.where(
    df_inf_adapt["previous_qual_expectation"] <= 0, 0, 1
)

# * Set qualitative estimates as ordinal variables
df_inf_adapt["current_qual_perception"] = pd.Categorical(
    df_inf_adapt["current_qual_perception"],
    ordered=True,
    categories=[0, 1],
)
df_inf_adapt["previous_qual_expectation"] = pd.Categorical(
    df_inf_adapt["previous_qual_expectation"],
    ordered=True,
    categories=[0, 1],
)

assert (
    df_inf_adapt.shape[0] == df_individual_char.shape[0]
    and df_inf_adapt.shape[1] == df_individual_char.shape[1] + 3
)

# %%
regressions = {}

for m in constants.ADAPTATION_MONTHS:
    model = smf.ols(
        formula="""avg_purchase ~ Actual + current_perception + previous_expectation \
        + C(treatment) * C(phase)""",
        data=df_inf_adapt[df_inf_adapt["month"] == m],
    )
    regressions[f"Month {m}"] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)

# %%
results

# %% [markdown]
### Regression with qualitative estimates
regressions = {}

for m in constants.ADAPTATION_MONTHS:
    model = smf.ols(
        formula="""avg_purchase ~ Actual + current_qual_perception + \
            previous_qual_expectation + C(treatment) * C(phase)""",
        data=df_inf_adapt[df_inf_adapt["month"] == m],
    )
    regressions[f"Month {m}"] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)

# %%
results


# %% [markdown]
### Généralité de l’efficacité
#
# `Régression diff_performance ~ intervention_1|intervention_2|contrôle *
# (toutes les caractéristiques)`
#
# Croiser une à une
_, df_performance_pivot = intervention.create_learning_effect_table(
    df_inf_adapt,
    constants.PERFORMANCE_MEASURES
    + constants.QUAL_INFLATION_MEASURES
    + constants.QUANT_INFLATION_MEASURES,
    constants.P_VALUE_THRESHOLDS,
)
df_performance_pivot.rename(
    columns={
        "Change in sreal": "diff_performance",
        "Change in Avg Qual Expectation Accuracy": "diff_avg_qual_exp",
        "Change in Avg Qual Perception Accuracy": "diff_avg_qual_perc",
        "Change in Average Uncertain Expectation": "diff_avg_uncertainty",
        "Change in Perception_sensitivity": "diff_perception_sensitivity",
        "Change in avg_perception_bias": "diff_perception_bias",
        "Change in Expectation_sensitivity": "diff_expectation_sensitivity",
        "Change in avg_expectation_bias": "diff_expectation_bias",
    },
    inplace=True,
)

# %%
df_performance_pivot = df_performance_pivot[
    [
        "participant.label",
        "diff_performance",
        "diff_avg_qual_perc",
        "diff_avg_uncertainty",
        "diff_perception_sensitivity",
        "diff_perception_bias",
        "diff_expectation_sensitivity",
        "diff_expectation_bias",
    ]
]


# %%
df_performance_pivot.columns = df_performance_pivot.columns.droplevel()

# %%
df_performance_pivot.columns = [
    "participant.label",
    "diff_performance",
    "diff_avg_qual_perc",
    "diff_avg_uncertainty",
    "diff_perception_sensitivity",
    "diff_perception_bias",
    "diff_expectation_sensitivity",
    "diff_expectation_bias",
]

# %%
df_inf_adapt = df_inf_adapt.merge(df_performance_pivot, how="left")

# %% [markdown]
### Regression with qualitative estimates
regressions = {}

model = smf.ols(
    formula="""diff_performance ~ C(treatment) / \
            (financial_literacy + numeracy + compound + wisconsin_choice_count \
                + riskPreferences_choice_count + riskPreferences_switches \
                    + lossAversion_choice_count + lossAversion_switches \
                        + timePreferences_choice_count + timePreferences_switches)""",
    data=df_inf_adapt[df_inf_adapt["month"] == 120],
)
results = model.fit()

# %%
results.summary()

# %% [markdown]
#### Analyse de médiation
#
# `Intervention → perception et anticipation → performance`
# `Intervention → performance`
df_inf_adapt.rename(
    columns={"Average Uncertain Expectation": "avg_uncertainty"}, inplace=True
)
data = df_inf_adapt[df_inf_adapt["month"] == 120]

regressions = {}
change_measures = [
    "diff_performance",
    "diff_avg_qual_perc",
    "diff_avg_uncertainty",
    "diff_perception_sensitivity",
    "diff_perception_bias",
    "diff_expectation_sensitivity",
    "diff_expectation_bias",
]
for m in change_measures:
    model = smf.ols(
        formula=f"""{m} ~ C(treatment)""",
        data=data,
    )
    regressions[m] = model.fit()

outcome_model = smf.ols(
    formula="""diff_performance ~ C(treatment) + diff_avg_qual_perc + \
        diff_avg_uncertainty + diff_perception_sensitivity + diff_perception_bias +\
            diff_expectation_sensitivity + diff_expectation_bias
        """,
    data=data,
)
regressions["outcome_model"] = outcome_model.fit()

mediator_model = smf.ols(
    formula="""diff_avg_uncertainty ~ C(treatment)""",
    data=data,
)
regressions["mediator_model"] = mediator_model.fit()

results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)

# %%
results

# %%
# * Replace treatment with dummy categories
criteria = [
    df_inf_adapt["treatment"] == "Intervention 1",
    df_inf_adapt["treatment"] == "Intervention 2",
]
choices = [1, 2]
df_inf_adapt["control"] = np.where(df_inf_adapt["treatment"] == "Control", 1, 0)
df_inf_adapt["intervention_1"] = np.where(
    df_inf_adapt["treatment"] == "Intervention 1", 1, 0
)
df_inf_adapt["intervention_2"] = np.where(
    df_inf_adapt["treatment"] == "Intervention 2", 1, 0
)
# %%
data = df_inf_adapt[df_inf_adapt["month"] == 120]
print(constants.MEDIATION_CONTROL)
mediation_analysis(
    data,
    x=constants.MEDIATION_CONTROL,
    m=[
        "diff_avg_qual_perc",
        "diff_avg_uncertainty",
        "diff_perception_sensitivity",
        "diff_perception_bias",
        "diff_expectation_sensitivity",
        "diff_expectation_bias",
    ],
    y="diff_performance",
    alpha=0.05,
    seed=42,
)

# %%
print(constants.MEDIATION_INTERVENTION_1)
mediation_analysis(
    data,
    x=constants.MEDIATION_INTERVENTION_1,
    m=[
        "diff_avg_qual_perc",
        "diff_avg_uncertainty",
        "diff_perception_sensitivity",
        "diff_perception_bias",
        "diff_expectation_sensitivity",
        "diff_expectation_bias",
    ],
    y="diff_performance",
    alpha=0.05,
    seed=42,
)

# %%
print(constants.MEDIATION_INTERVENTION_2)
mediation_analysis(
    data,
    x=constants.MEDIATION_INTERVENTION_2,
    m=[
        "diff_avg_qual_perc",
        "diff_avg_uncertainty",
        "diff_perception_sensitivity",
        "diff_perception_bias",
        "diff_expectation_sensitivity",
        "diff_expectation_bias",
    ],
    y="diff_performance",
    alpha=0.05,
    seed=42,
)

# %% [markdown]
# The mediation analysis of each treatment shows:
# - <b><u>Control</u></b> neither improves performance (in fact directly worsening performance)
# nor improves mediating factors. The change in average uncertainty does improve
# performance, though.
# - <b><u>Intervention 1</u></b> improves performance through full mediation, improving the
# mediating factor of the change in average uncertainty. The intervention also improves
# expectation sensitivity statistically significantly, but the indirect effect this
# has on performance is not signficant.
# - <b><u>Intervention 2</u></b> improves performance through partial mediation,
# improving performance directly as well as through the mediating factor of the
# change in average uncertainty and expectation sensitivity, but the indirect
# effect these mediators on performance is not signficant.
