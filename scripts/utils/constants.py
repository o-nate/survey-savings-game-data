"""Constants for scripts modules"""

# * Define `decision quantity` measure
DECISION_QUANTITY = "cum_decision"

# * Define purchase window, i.e. how many months before and after inflation phase change to count
WINDOW = 3

P_VALUES_THRESHOLDS = [0.1, 0.05, 0.01]
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
