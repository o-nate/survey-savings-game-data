"""Constants for scripts modules"""

QUESTIONNAIRE_MEASURES = [
    "age",
    "gender",
    "educationLevel",
    "employmentStatus",
    "financialStatusIncome",
    "financialStatusSavings_1",
    "financialStatusSavings_2",
    "financialStatusDebt_1",
    "financialStatusDebt_2",
    "stocks",
    "mutualFunds",
    "bonds",
    "savingsAccounts",
    "lifeInsurance",
    "retirementAccounts",
    "crypto",
]

# * Define `decision quantity` measure
DECISION_QUANTITY = "cum_decision"

# * Define purchase window, i.e. how many months before and after inflation phase change to count
WINDOW = 3

PERFORMANCE_MEASURES_OLD_NAMES = ["sreal_%", "early_%", "excess_%"]
PERFORMANCE_MEASURES_NEW_NAMES = ["Total savings", "Over-stocking", "Wasteful-stocking"]
PURCHASE_ADAPTATION_OLD_NAME = ["avg_q_%"]
PURCHASE_ADAPTATION_NEW_NAME = ["Purchase adaptation"]

P_VALUE_THRESHOLDS = [0.1, 0.05, 0.01]
DECIMAL_PLACES = 15

PERFORMANCE_MEASURES = ["sreal", "early", "excess", "avg_q_%"]
QUANT_INFLATION_MEASURES = [
    "Perception_sensitivity",
    "avg_perception_bias",
    "Expectation_sensitivity",
    "avg_expectation_bias",
]
QUAL_INFLATION_MEASURES = [
    "Avg Qual Expectation Accuracy",
    "Avg Qual Perception Accuracy",
    "Average Uncertain Expectation",
]
KNOWLEDGE_MEASURES = ["financial_literacy", "numeracy", "compound"]
ECON_PREFERENCE_MEASURES = [
    "lossAversion_choice_count",
    "lossAversion_switches",
    "riskPreferences_choice_count",
    "riskPreferences_switches",
    "timePreferences_choice_count",
    "timePreferences_switches",
    "wisconsin_choice_count",
    "wisconsin_PE",
    "wisconsin_SE",
]
TREATMENTS = ["Intervention 1", "Intervention 2", "Control"]
ADAPTATION_MONTHS = [12, 24, 36, 48]
INDIVIDUAL_CHARACTERISTICS = [
    "financial_literacy",
    "numeracy",
    "compound",
    "wisconsin_choice_count",
    "riskPreferences_choice_count",
    "riskPreferences_switches",
    "lossAversion_choice_count",
    "lossAversion_switches",
    "timePreferences_choice_count",
    "timePreferences_switches",
]

# * Mediation analysis dummies
MEDIATION_CONTROL = "control"
MEDIATION_INTERVENTION_1 = "intervention_1"
MEDIATION_INTERVENTION_2 = "intervention_2"
