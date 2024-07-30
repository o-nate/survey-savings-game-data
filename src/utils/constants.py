"""Constants for src modules"""

# * Define annualized inflation, per 12 months
INF_1012 = [0.45, 60.79, 0.45, 60.79, 0.45, 60.79, 0.45, 60.79, 0.45, 60.79]
INF_430 = [0.38, 0.47, 26.85, 55.49, 64.18, 0.38, 0.47, 26.85, 55.49, 64.18]

INFLATION_DICT = {
    "participant.inflation": [430 for m in range(40)],  # + [1012 for m in range(19)],
    "participant.round": [1 for m in range(10)]
    + [1 for m in range(10)]
    + [2 for m in range(10)]
    + [2 for m in range(10)],
    "Month": [m * 12 for m in range(1, 11)]
    + [m * 12 if m > 0 else m + 1 for m in range(10)]
    + [m * 12 for m in range(1, 11)]
    + [m * 12 if m > 0 else m + 1 for m in range(10)],
    "Measure": ["Actual" for m in range(10)]
    + ["Upcoming" for m in range(10)]
    + ["Actual" for m in range(10)]
    + ["Upcoming" for m in range(10)],
    "Estimate": INF_430 + INF_430 + INF_430 + INF_430,  # + INF_1012 + INF_1012[1:],
}
