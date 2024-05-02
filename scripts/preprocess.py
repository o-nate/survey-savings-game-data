"""Convert data to dataframes"""

import os
from pathlib import Path
import re
import time
import warnings

import json

import pandas as pd
from tqdm import tqdm

# # Declare file name
FILE = "all_apps_wide_2024-04-30.csv"

# Declare name of output file
FILE_PREFIX = "processed_"


# do not hide rows in output cells
pd.set_option("display.max_rows", None, "display.max_columns", None)

## Disable warnings
warnings.filterwarnings("ignore")

# turn off warning about df copies
pd.options.mode.chained_assignment = None  # default='warn'


# # Participant fields, player fields, and oTree apps


participant_fields = [
    "code",
    "label",
    "instructions",
    "reaction_times",
    "periods_survived",
    "task_results",
    "task_results_1",
    "errors_1",
    "task_results_2",
    "errors_2",
    "lossAversion",
    "riskPreferences",
    "wisconsin",
    "remunerated_behavioral",
    "day_1",
    "inflation",
    "round",
    "day_room_number",
    "error",
    "last_room_day",
    "next_room",
    "err_msg",
    "vars_done",
]

apps = [
    "participant",
    "init",
    "Questionnaire",
    "lossAversion",
    "riskPreferences",
    "wisconsin",
    "timePreferences",
    "Finance",
    "Inflation",
    "Numeracy",
    "task_instructions",
    "task",
    "task_int",
    "redirectapp",
    "sessionResults",
    "redirecttopayment",
]
print(f"Processing apps: {apps}")

# Define new tables per task fields
fields = [
    "decision",
    "total_price",
    "initial_savings",
    "cashOnHand",
    "finalSavings",
    "finalStock",
    "interestEarned",
    "newPrice",
    "realInterest",
    "responseTime",
    "qualitative_estimate",
    "qualitative_expectation",
    "inf_estimate",
    "inf_expectation",
]
print(f"Processing task fields: {fields}")


# # Create df and organize
# make dataframe
parent_dir = Path(__file__).parents[1]
data_dir = parent_dir / "data"
print("data_dir", data_dir)
complete = pd.read_csv(f"{data_dir}/{FILE}")
print(f"Initial dataframe size: {complete.shape}")
print("Removing participants who did not finish or had internet connection issues.")

# remove rows participant.label = NaN
complete = complete[complete["participant.label"].notna()]

# # Remove rows with participant.label = bugs (due to bug)
# complete = complete[complete["participant.label"].isin(finished)]

# remove columns with all NaN
complete = complete.dropna(axis=1, how="all")

# convert participant.time_started_utc value to datetime
complete["participant.time_started_utc"] = pd.to_datetime(
    complete["participant.time_started_utc"]
)

# # add column with intervention
# complete["participant.intervention"] = ""
# complete["participant.day_3"] = complete["participant.day_3"].apply(eval)
# for i in range(len(complete.index)):
#     if complete["participant.day_3"].iloc[i][0] == "task_int_cx":
#         complete["participant.intervention"].iloc[i] = "intervention"
#     else:
#         complete["participant.intervention"].iloc[i] = "control"

# FILTER BY DATE
FROM_TS = "2024-04-30 00:00:00"
# after_ts = '2023-02-17 23:59:00'
complete = complete[complete["participant.time_started_utc"] >= FROM_TS]

# organize rows by participant.label
# and display corresponding codes
complete = complete.sort_values(
    ["participant.label", "participant.time_started_utc"], ascending=[False, True]
)
participant = complete[
    ["participant.label", "participant.code", "participant.time_started_utc"]
]
participant = participant.groupby(
    ["participant.label", "participant.code", "participant.time_started_utc"]
).all()

participant = participant.sort_values(
    ["participant.label", "participant.time_started_utc"], ascending=[False, True]
)

# participant


# # Generate separate df for each app + for participant info
def split_df(df):
    for app in apps:
        df_list.append(app)
        if app == "task" or app == "task_questions":
            df_dict[app] = df.filter(
                regex=f"participant.code|participant.label|participant.time_started_utc|participant.day|participant.inflation|participant.intervention|{app}."
            )
        else:
            df_dict[app] = df.filter(
                regex=f"participant.code|participant.label|participant.time_started_utc|participant.day|participant.intervention|{app}."
            )
    return df_list


df_dict = {}
df_list = []

split_df(complete)

# In[390]:


# participant df
df_dict["participant"].drop("participant.task_results", axis=1, inplace=True)
df_dict["participant"].drop("participant._is_bot", axis=1, inplace=True)

# wisconin df
df_dict["wisconsin"].drop("session.config.wisconsin_fee", axis=1, inplace=True)

# task
df_dict["task"].drop(
    list(
        df_dict["task"].filter(
            regex="participant.task|session|task_int|task_instructions"
        )
    ),
    axis=1,
    inplace=True,
)


# # Remove blank rows
# (i.e. rows for other apps)

# In[391]:
# for app in tqdm(df_list, desc='Removing blank rows from apps'):
for app in df_list:
    if app != "participant":
        df_dict[app].dropna(subset=[f"{app}.1.player.id_in_group"], inplace=True)


# # `timePreferences`: Distribute delays into individual columns for each round

# In[392]:


new_timePreferences = pd.DataFrame(
    data=df_dict["timePreferences"]["timePreferences.1.player.delay"].tolist()
)
new_timePreferences.head()


# In[393]:


new_timePreferences.rename(columns={0: "delay_order"}, inplace=True)

# convert str to dict
new_timePreferences.dicts = new_timePreferences.delay_order.apply(json.loads)
new_timePreferences["dict_list"] = new_timePreferences.dicts

for n in range(len(new_timePreferences["dict_list"][0]["delay"])):
    new_timePreferences[f"{n}"] = n

for i in range(len(new_timePreferences["dict_list"])):
    new_timePreferences["dict_list"][i] = new_timePreferences["dict_list"][i]["delay"]

for k in range(len(new_timePreferences["dict_list"])):
    for j in range(len(new_timePreferences["dict_list"][0])):
        new_timePreferences[f"{j}"][k] = new_timePreferences["dict_list"][k][j]["delay"]
        new_timePreferences[f"{j}"] = new_timePreferences[f"{j}"]

new_new = new_timePreferences.iloc[:, 2:]

# In[394]:


for k in range(len(new_new.index)):
    for j in range(1, 1 + len(new_new.columns)):
        df_dict["timePreferences"][f"timePreferences.{j}.player.delay"].iloc[k] = (
            new_new.iloc[k][j - 1]
        )

# In[395]:
# # `riskPreferences`: Convert `raw_responses` to columns

new_riskPreferences = pd.DataFrame(
    data=df_dict["riskPreferences"]["riskPreferences.1.player.raw_responses"].tolist()
)
new_riskPreferences.rename(columns={0: "responses"}, inplace=True)

# remove trial id and '-' from dict key


def risk_responses(text):
    pattern = r"\d{,} - "
    result = re.sub(pattern, "", text)
    return result


new_riskPreferences["responses"] = new_riskPreferences.responses.map(risk_responses)

# convert str to dict
new_riskPreferences.dicts = new_riskPreferences.responses.apply(json.loads)
new_riskPreferences["responses"] = new_riskPreferences.dicts

# convert dict key:value to columns
new_riskPreferences = pd.concat(
    [new_riskPreferences, new_riskPreferences["responses"].apply(pd.Series)], axis=1
)

# reorder columns
col_order = []
for i in range(1, 11):
    i = i * 10
    col_order.append(f"{i}")

new_riskPreferences = new_riskPreferences[col_order]

new_riskPreferences.head()

# recombine with riskPreferences df
for j in range(1, 11):
    k = j * 10
    df_dict["riskPreferences"][f"riskPreferences.player.probability.{k}"] = (
        new_riskPreferences[f"{k}"].values
    )

df_dict["riskPreferences"].head()

# In[397]:
# # `lossAversion`: Distribute `raw_responses` to columns

new_lossAversion = pd.DataFrame(
    data=df_dict["lossAversion"]["lossAversion.1.player.raw_responses"].tolist()
)
new_lossAversion.rename(columns={0: "responses"}, inplace=True)

# remove trial id and '-' from dict key


def loss_responses(text):
    pattern_1 = r"\d{,} - "
    pattern_2 = r"\s₮"
    result = re.sub(pattern_1, "", text)
    result = re.sub(pattern_2, "", result)
    return result


new_lossAversion["responses"] = new_lossAversion.responses.map(loss_responses)

# convert str to dict
new_lossAversion.dicts = new_lossAversion.responses.apply(json.loads)
new_lossAversion["responses"] = new_lossAversion.dicts

# convert dict key:value to columns
new_lossAversion = pd.concat(
    [new_lossAversion, new_lossAversion["responses"].apply(pd.Series)], axis=1
)

# reorder columns
col_order = []
for i in range(2, 8):
    i = i * 100 * 2  # multiply by 2 because double lottery sizes
    col_order.append(f"{i},00")

new_lossAversion = new_lossAversion[col_order]

# recombine with lossAversion df
for j in range(2, 8):
    k = j * 100 * 2  # multiply by 2 because double lottery sizes
    df_dict["lossAversion"][f"lossAversion.player.loss.{k}"] = new_lossAversion[
        f"{k},00"
    ].values

df_dict["lossAversion"].head()

# In[398]:
# # Unpack `wisconsin` dict

df_wisc = df_dict["wisconsin"].copy()
df_wisc = df_wisc.dropna(axis=0)

# convert str to dict
df_wisc.dicts = df_wisc["wisconsin.1.player.response_time"].apply(json.loads)
df_wisc["wisconsin.1.player.response_time"] = df_wisc.dicts

# convert dict key:value to columns
df_wisc = pd.concat(
    [df_wisc, df_wisc["wisconsin.1.player.response_time"].apply(pd.Series)], axis=1
)

# ## remove `wisconsin.1.player.response_time`
# df_wisc = df_wisc.drop(columns=['wisconsin.1.player.response_time','participant.time_started_utc'])

# create new df to break up dicts
new = df_wisc.iloc[:, :1]
# break dicts into columns for each trial
for col in range(1, 31):
    new[col] = df_wisc[f"{col}"]

# convert list to dict and create columns for each key:value pair
for n in range(len(df_wisc.index)):
    for col in range(1, 31):
        response_dict = json.loads(df_wisc[str(col)].iloc[n][0])
        for i in response_dict:
            new[f"trial_{col}_{i}"] = None

# %%
# replace cell values with corresponding key:value pairs
for n in tqdm(range(len(df_wisc.index)), desc="Wisconsin, extracting responses"):
    for col in range(1, 31):
        response_dict = json.loads(df_wisc[str(col)].iloc[n][0])
        for i in response_dict:
            new[f"trial_{col}_{i}"].iloc[n] = response_dict[i]


# concatenate new columns with df_wisc
df_wisc = pd.concat([df_wisc, new.iloc[:, 31:]], axis=1)

# remove separated trial number columns
columns = []
for col in range(1, 31):
    df_wisc.drop(str(col), axis=1, inplace=True)

# %%
# update original df
df_dict["wisconsin"] = df_wisc


# In[399]:
# # `task`: Further clean data

task = df_dict["task"].copy()

# Drop columns with 'day_'
task.drop(task.filter(regex="day_").columns, axis=1, inplace=True)

# Resort by index
task.sort_index(inplace=True)

# # Convert 'participant.inflation' to list and extract corresponding day's inf sequence
# task["participant.inflation"] = task["participant.inflation"].apply(eval)
# for n in range(len(task.index)):
#     day = task["participant.day"].iloc[n]
#     task["participant.inflation"].iloc[n] = task["participant.inflation"].iloc[n][
#         int(day) - 1
#     ]


def split_df_task(df):
    for field in fields:
        task_df_list.append(field)
        task_df_dict[field] = df.filter(
            regex=f"participant.code|participant.label|utc|participant.day|participant.inflation|participant.intervention|{field}$"
        )
    return task_df_list


task_df_dict = {}
task_df_list = []

split_df_task(task)

# %%
# Extract quantity purchased per month
decision = task_df_dict["decision"].copy()

# Replace NaNs in decision
decision.fillna('{"item": [], "quantity": [], "price": []}', inplace=True)

for month in tqdm(range(1, 121), desc="Extracting decision quantities"):
    col = f"task.{month}.player.decision"
    # Convert str to dict
    decision[col] = decision[col].apply(json.loads)
    ## Convert row value to quantity value
    decision.loc[decision[col].notnull(), col] = decision.loc[
        decision[col].notnull(), col
    ].apply(lambda x: x.get("quantity"))
    ## Extract value from list
    decision.loc[decision[col].notnull(), col] = decision.loc[
        decision[col].notnull(), col
    ].str[0]
    decision[col] = decision[col].fillna(0)


# %%
task_df_dict["decision"] = decision

## Add df's to general dict and and list
final_df_dict = df_dict | task_df_dict
final_df_list = df_list + task_df_list

# In[401]:
# # Calculate payment results

final_payments = df_dict["participant"].loc[
    (df_dict["participant"]["participant._current_app_name"] == "redirecttopayment")
]
final_payments = final_payments[
    [
        "participant.label",
        "participant.payoff",
        "participant.task_results_1",
        "participant.task_results_2",
        "participant.day_1",
        "participant.remunerated_behavioral",
    ]
]


# convert from str to list
final_payments["participant.day_1"] = final_payments["participant.day_1"].map(eval)
# final_payments['participant.day_3'] = final_payments['participant.day_3'].map(eval)
final_payments["participant.remunerated_behavioral"] = final_payments[
    "participant.remunerated_behavioral"
].map(eval)


# remove non-behavioral task names from lists
final_payments["participant.day_1"] = final_payments["participant.day_1"].apply(
    lambda x: [i for i in x if i != "debut"]
)


# # generate columns for each task with corresponding results
# final_payments = pd.concat(
#     [final_payments, final_payments["results_dict1"].apply(pd.Series)], axis=1
# )

final_payments.loc["Total"] = final_payments[
    [
        "participant.payoff",
        "participant.task_results_1",
        "participant.task_results_2",
        "participant.remunerated_behavioral",
    ]
].sum()
final_payments = final_payments.drop(
    [
        "participant.day_1",
    ],
    axis=1,
)
final_payments = final_payments.fillna("-")
final_df_list.append("final_payments")
final_df_dict["final_payments"] = final_payments

# In[402]:
# # Result stats

stats_final_payments = final_payments[
    final_payments["participant.label"] != "-"
].describe()
final_df_list.append("stats_final_payments")
final_df_dict["stats_final_payments"] = stats_final_payments
stats_final_payments

print(f"Complete. Total participants included: {final_payments.shape[0] - 1 }")

# %%
if __name__ == "__main__":
    ## Excel of performance per session
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print(timestr)
    with pd.ExcelWriter(f"{data_dir}/performance_{timestr}.xlsx") as writer:
        participant.to_excel(writer, sheet_name="participant")
        final_df_dict["final_payments"].to_excel(writer, sheet_name="final_payments")
        for df in final_df_dict:
            final_df_dict[df].to_excel(writer, sheet_name=f"{df}")
            print(f"final_df_dict[{df}]")

    print("done")
