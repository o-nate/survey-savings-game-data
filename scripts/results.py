"""Present results from experiment"""

# %%
import logging
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# import seaborn as sns

from src import calc_opp_costs, discontinuity, process_survey

from src.calc_opp_costs import df_opp_cost
from src.preprocess import final_df_dict
from src.utils.logging_helpers import set_external_module_log_levels

# * Logging settings
logger = logging.getLogger(__name__)
set_external_module_log_levels("error")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# * Define `decision quantity` measure
DECISION_QUANTITY = "cum_decision"

# * Define purchase window, i.e. how many months before and after inflation phase change to count
WINDOW = 3

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
calc_opp_costs.plot_savings_and_stock(df_opp_cost, col="phase", palette="tab10")


# %% [markdown]
## Behavior in the Savings Game

# %% [markdown]
## Performance measures: Over-, under-, and wasteful-stocking and purchase adaptation
df_measures = discontinuity.purchase_discontinuity(
    df_opp_cost, DECISION_QUANTITY, WINDOW
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
plt.grid()

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
plt.grid()


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

#### Relationship between expectations, perceptions, and decisions

##### Difference between quantitative and qualitative estimates

# %% [markdown]
## Real life vs. savings game

### Comparison to trends from CAMME in real life

# Figure I – Corrélation entre inflation perçue et anticipée (en %) <br><br>

# Figure II – Effet d’apprentissage des répondants (en points de pourcentage) <br><br>

# Tableau 3 – Réponses à la question qualitative sur l’anticipation à un an <br><br>

# Figure III – Distribution des perceptions et anticipations d’inflation
# (en % des répondants) <br><br>

# Figure IV – Inflation IPCH et anticipations d’inflation 2020-2021 <br><br>

# Figure V – Évolution de l’incertitude des ménages (en %) <br><br>


# %% [markdown]
## The role of individual characteristics and behavior

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
