"""
Created on Mon 26/01/2026
Ourania Semelidou
Trial history integration during tactile categorization.
2AFC trial history extraction to:
- fit a logistic GLM (Fig. 6)
- calculate and plot coefficients (Fig. 6A)
- plot psychometric curves (Fig. 6)
- plot marginal effect (Fig. 6)
- calculate the effect of categorization on discrimination (Fig. 4C)
WT vs KO, multiple sessions and mice

for eLife paper: Altered cognitive processes shape tactile perception in autism.
Figures 6 and 4C
"""

import pandas as pd
import json
import os
import statsmodels.api as sm
import matplotlib
matplotlib.use('TkAgg')  # or 'Agg' for non-interactive plots
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ttest_ind
from itertools import combinations
from scipy.stats import ttest_rel, ttest_ind, wilcoxon
from statsmodels.formula.api import mixedlm

# Directory with mouse session folders
ROOT_DIR = r"D:\vibrotactile task\2AFC\2AFC_testing_all trials for GGLM" #Folder with data provided on Figshare

def extract_trials_from_session(bpod_xls_path, stim_json_path,
                                mouse_id, genotype, session_id):

    # Read Bpod Excel
    bpod_df = pd.read_excel(bpod_xls_path, skiprows=6)
    # Rename 6th column to 'INFO'
    bpod_df.rename(columns={bpod_df.columns[5]: 'INFO'}, inplace=True)

    with open(stim_json_path, "r") as f:
        stim_data = json.load(f)

    trials = []
    current_trial = None
    prev_choice = None
    prev_outcome = np.nan
    prev_stim_amp = None
    prev_outcome_side = None
    prev_outcome_left = 0
    prev_outcome_right = 0
    trial_idx = -1
    in_response_window = False
    last_event = None

    for _, row in bpod_df.iterrows():
        # Start new trial
        if row["TYPE"] == "TRIAL" and "New trial" in str(row["MSG"]):
            trial_idx += 1
            if trial_idx >= len(stim_data):
                current_trial = None
                in_response_window = False
                continue

            current_trial = {
                "mouse_id": mouse_id,
                "genotype": genotype,
                "session": session_id,
                "trial_number": stim_data[trial_idx]["nb_trial"],
                "stimulus_amplitude": stim_data[trial_idx]["amp"],

                "in_response_window": False,
                "rw_start": None,
                "rw_end": None,
                "choice": None,
                "outcome": None,
                "outcome_side": np.nan,
                "valid_response": False,

                "prev_stim_amp": prev_stim_amp,
                "previous_choice": prev_choice,
                "previous_outcome": prev_outcome,
                "prev_outcome_side": prev_outcome_side,

            }
            response_window = False
            last_event = None
            continue

        # Ignore everything if we are past JSON
        if current_trial is None:
            continue

        # Response window
        if current_trial is not None and row["TYPE"] == "TRANSITION" and row["MSG"] == "ResponseWindow":
            in_response_window = True
        elif row["MSG"] in ["Reward_left", "Reward_right", "Timeout"]:
            in_response_window = False

        #Lick detection
        if (
                row["TYPE"] == "EVENT"
                and in_response_window
                and not current_trial["valid_response"]
        ):
            if row["INFO"] == "Port1In":
                current_trial["choice"] = 1
                current_trial["valid_response"] = True
                last_event = "Port1In"
            elif row["INFO"] == "Port2In":
                current_trial["choice"] = 0
                current_trial["valid_response"] = True
                last_event = "Port2In"

        # Outcome detection and side
        if current_trial is not None and row["TYPE"] == "TRANSITION":
            if row["MSG"] == "Reward_right":
                current_trial["outcome"] = 1
                current_trial["outcome_side"] = "right"
            elif row["MSG"] == "Reward_left":
                current_trial["outcome"] = 1
                current_trial["outcome_side"] = "left"
            elif row["MSG"] == "Timeout":
                current_trial["outcome"] = 0  # timeout
                if last_event == "Port1In":
                    current_trial["outcome_side"] = "right"  # wrong lick right
                elif last_event == "Port2In":
                    current_trial["outcome_side"] = "left"   # wrong lick left

        # End trial: append trial and update previous variables
        if current_trial is not None and row["TYPE"] == "END-TRIAL":
            if current_trial["valid_response"]:
                trials.append(current_trial)

                # Update for next trial
                prev_choice = current_trial["choice"]
                prev_outcome = current_trial["outcome"]
                prev_stim_amp = current_trial["stimulus_amplitude"]
                prev_outcome_side = current_trial.get("outcome_side", np.nan)

            current_trial = None  # reset trial after end
            response_window = False

        assert len(trials) <= len(stim_data), (
          f"Extracted {len(trials)} trials but JSON has {len(stim_data)} trials "
         f"({mouse_id}, session {session_id})"
      )
    return trials


# Main extraction loop
all_trials = []

for mouse_folder in os.listdir(ROOT_DIR):
    mouse_path = os.path.join(ROOT_DIR, mouse_folder)
    if not os.path.isdir(mouse_path):
        continue

    # Identify genotype
    if mouse_folder.startswith("WT"):
        genotype = "WT"
    elif mouse_folder.startswith("KO"):
        genotype = "KO"
    else:
        raise ValueError(f"Unknown genotype in folder: {mouse_folder}")

    mouse_id = mouse_folder

    # Loop over XLS sessions
    xls_files = [f for f in os.listdir(mouse_path) if f.endswith(".xls")]
    for xls_file in xls_files:
        session_id = os.path.splitext(xls_file)[0]
        xls_path = os.path.join(mouse_path, xls_file)
        json_path = os.path.join(mouse_path, session_id + ".json")

        if not os.path.exists(json_path):
            print(f"Warning: no JSON for {xls_file}, skipping")
            continue

        trials = extract_trials_from_session(xls_path, json_path,
                                             mouse_id, genotype, session_id)
        all_trials.extend(trials)

# --- SESSION-WISE PREVIOUS TRIAL HISTORY ---

# Sort by mouse, session, trial
trial_df = pd.DataFrame(all_trials)
trial_df = trial_df.sort_values(["mouse_id", "session", "trial_number"]).reset_index(drop=True)

for (_, sess), g in trial_df.groupby(["mouse_id", "session"]):
    assert np.all(
        g["previous_choice"].values[1:] == g["choice"].values[:-1]
    ), f"History mismatch in {sess}"

consecutive = trial_df.groupby(["mouse_id", "session"])["trial_number"].apply(
    lambda x: np.array_equal(x.values, np.arange(1, len(x)+1))
)

# Compute previous choice and previous outcome session-wise
trial_df["previous_choice"] = trial_df.groupby(["mouse_id", "session"])["choice"].shift(1)
trial_df["previous_outcome"] = trial_df.groupby(["mouse_id", "session"])["outcome"].shift(1)
trial_df["prev_stim_z"] = trial_df.groupby(["mouse_id"])["stimulus_amplitude"].shift(1)

# Fill first-trial history predictors with safe default
trial_df["previous_choice"] = trial_df["previous_choice"].fillna(0)   # assume "left" for first trial
trial_df["previous_outcome"] = trial_df["previous_outcome"].fillna(1) # assume rewarded previous trial
trial_df["prev_stim_z"] = trial_df["prev_stim_z"].fillna(0)           # z-score reference 0

# Z-score stimuli (after handling first trial)
trial_df["stim_z"] = (trial_df["stimulus_amplitude"] - trial_df["stimulus_amplitude"].mean()) / trial_df["stimulus_amplitude"].std()
trial_df["prev_stim_z"] = (trial_df["prev_stim_z"] - trial_df["stimulus_amplitude"].mean()) / trial_df["stimulus_amplitude"].std()

# Drop remaining NaNs (should be none now)
trial_df = trial_df.dropna(subset=["choice", "previous_choice", "previous_outcome", "stim_z", "prev_stim_z"])

# Ensure integer type for choice variables
trial_df["choice"] = trial_df["choice"].astype(int)
trial_df["previous_choice"] = trial_df["previous_choice"].astype(int)
trial_df["previous_outcome"] = trial_df["previous_outcome"].astype(int)

# sanity checks
print("\nChoice × Outcome table:")
print(pd.crosstab(trial_df["choice"], trial_df["outcome"]))

# Session-wise previous-choice mismatch check
mismatch = sum(
    (g["previous_choice"].values[1:] != g["choice"].values[:-1]).sum()
    for _, g in trial_df.groupby(["mouse_id", "session"])
)
print("Previous-choice mismatches (session-wise):", mismatch)

# Z-score stimuli
trial_df["stim_z"] = (trial_df["stimulus_amplitude"] - trial_df["stimulus_amplitude"].mean()) / trial_df["stimulus_amplitude"].std()
trial_df["prev_stim_z"] = trial_df.groupby("mouse_id")["stim_z"].shift(1)

# drop first trial per mouse (after all shifts)
trial_df = (
    trial_df
    .groupby("mouse_id", as_index=False)
    .apply(lambda g: g.iloc[1:])
    .reset_index(drop=True)
)

# fill remaining NaNs
trial_df["prev_stim_z"] = trial_df["prev_stim_z"].fillna(0)

trial_df["choice"] = trial_df["choice"].astype(int)
trial_df["genotype_WT"] = (trial_df["genotype"] == "WT").astype(int)

history_cols = [
    "previous_choice",
    "previous_outcome",
    "prev_stim_z"
]

# Drop trials where history is undefined
trial_df = trial_df.dropna(subset=history_cols)

print(trial_df[[
    "previous_choice",
    "previous_outcome",
    "prev_stim_z"
]].isna().sum())

# FINAL predictor set (non-singular)
predictors = [
    "stim_z",
    "prev_stim_z",
    "previous_outcome",
    "previous_choice",
    "genotype_WT"
]

print(trial_df[[
    "previous_outcome",
    "previous_choice",
    "stim_z",
    "prev_stim_z"
]].isna().sum())
print(trial_df[predictors].nunique())


# Check rightward licks for KO1
mouse_id = "KO_mouse1"
mouse_df = trial_df[trial_df["mouse_id"] == mouse_id]
right_lick_frac = mouse_df.groupby("stimulus_amplitude")["choice"].mean()
right_lick_dict = {f"{amp:.2f}": frac for amp, frac in right_lick_frac.items()}
print("\nRight licks KO1 (post-processing):", right_lick_dict)
trial_df_raw = pd.DataFrame(all_trials)
print("Total raw trials for KO1:", len(trial_df_raw[trial_df_raw["mouse_id"] == "KO_mouse1"]))

X = sm.add_constant(trial_df[predictors])
y = trial_df["choice"]

model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit(
    cov_type="cluster",
    cov_kwds={"groups": trial_df["mouse_id"]}
)

print(result.summary())

# ============================================================
# Genotype × history interaction model

# Create interaction terms
trial_df["genotype_x_prev_outcome"] = trial_df["genotype_WT"] * trial_df["previous_outcome"]
trial_df["genotype_x_prev_choice"]  = trial_df["genotype_WT"] * trial_df["previous_choice"]
trial_df["genotype_x_prev_stim"]    = trial_df["genotype_WT"] * trial_df["prev_stim_z"]
trial_df["genotype_x_stim"]         = trial_df["genotype_WT"] * trial_df["stim_z"]

# Sanity check: variability
print("\nInteraction predictors variability:")
print(trial_df[[
    "genotype_WT",
    "stim_z",
    "prev_stim_z",
    "previous_outcome",
    "previous_choice",
    "genotype_x_prev_outcome",
    "genotype_x_prev_choice",
    "genotype_x_prev_stim",
    "genotype_x_stim"
]].nunique())

# Predictor set with interactions
interaction_predictors = [
    "stim_z",
    "prev_stim_z",
    "previous_outcome",
    "previous_choice",
    "genotype_WT",
    "genotype_x_stim",
    "genotype_x_prev_outcome",
    "genotype_x_prev_choice",
    "genotype_x_prev_stim"
]

# Design matrix
X_int = sm.add_constant(trial_df[interaction_predictors])
y = trial_df["choice"]

# Fit GLM with cluster-robust SEs
interaction_model = sm.GLM(y, X_int, family=sm.families.Binomial())
interaction_result = interaction_model.fit(
    cov_type="cluster",
    cov_kwds={"groups": trial_df["mouse_id"]}
)

print("\nInteraction GLM coefficients:")
print(interaction_result.params)

print("\n=== Genotype × history interaction model ===")
print(interaction_result.summary())

# Psychometric curves taking into account the previous choice/outcome/stim --------------------------------------
import statsmodels.api as sm

# Helper: safe prediction with correct column order
def predict_prob(df, model):
    df = sm.add_constant(df, has_constant="add")
    df = df[model.model.exog_names]
    return model.predict(df)

# Stimulus range for smooth curves
stim_range = np.linspace(trial_df["stim_z"].min(), trial_df["stim_z"].max(), 100)

# Colors
wt_color = (0.0, 179/255, 179/255)  # teal
ko_color = (1.0, 165/255, 0.0)      # orange

# ===========================
# 1. Curves split by previous choice
# ===========================
plt.figure(figsize=(8,6))

for genotype, label, color in zip([0,1], ["KO","WT"], [ko_color, wt_color]):
    for prev_choice, ls, suffix in zip([0,1], ["--","-"], ["Prev Left", "Prev Right"]):
        pred_df = pd.DataFrame({
            "stim_z": stim_range,  # vary current stimulus for psychometric curve
            "prev_stim_z": 0,  # neutral previous stimulus
            "previous_choice": prev_choice,  # here we vary previous choice
            "previous_outcome": 1,  # assume previous trial was rewarded
            "genotype_WT": genotype
        })
        # Interaction terms if present
        if "genotype_x_prev_outcome" in interaction_result.params:
            pred_df["genotype_x_prev_outcome"] = pred_df["genotype_WT"] * pred_df["previous_outcome"]
        if "genotype_x_prev_choice" in interaction_result.params:
            pred_df["genotype_x_prev_choice"] = pred_df["genotype_WT"] * pred_df["previous_choice"]
        if "genotype_x_prev_stim" in interaction_result.params:
            pred_df["genotype_x_prev_stim"] = pred_df["genotype_WT"] * pred_df["prev_stim_z"]
        if "genotype_x_stim" in interaction_result.params:
            pred_df["genotype_x_stim"] = pred_df["genotype_WT"] * pred_df["stim_z"]

        # Predict probabilities safely
        pred_prob = predict_prob(pred_df, interaction_result)

        # Plot
        plt.plot(stim_range, pred_prob, linestyle=ls, linewidth=2, color=color,
                 label=f"{label}, {suffix}")

# Stimulus-only curves (neutral previous choice/outcome)
for genotype, label, color in zip([0,1], ["KO","WT"], [ko_color, wt_color]):
    pred_df = pd.DataFrame({
        "stim_z": stim_range,
        "prev_stim_z": 0,
        "previous_outcome": 0.5,  # neutral
        "previous_choice": 0.5,  # neutral
        "genotype_WT": genotype
    })
    if "genotype_x_prev_outcome" in interaction_result.params:
        pred_df["genotype_x_prev_outcome"] = pred_df["genotype_WT"] * pred_df["previous_outcome"]
    if "genotype_x_prev_choice" in interaction_result.params:
        pred_df["genotype_x_prev_choice"] = pred_df["genotype_WT"] * pred_df["previous_choice"]
    if "genotype_x_prev_stim" in interaction_result.params:
        pred_df["genotype_x_prev_stim"] = genotype * pred_df["prev_stim_z"]
    if "genotype_x_stim" in interaction_result.params:
        pred_df["genotype_x_stim"] = pred_df["genotype_WT"] * pred_df["stim_z"]

    pred_prob = predict_prob(pred_df, interaction_result)

    plt.plot(stim_range, pred_prob, linestyle='-', linewidth=2, color=color, alpha=0.3,
             label=f"{label}, Stim only")

plt.xlabel("Z-scored Stimulus Amplitude")
plt.ylabel("P(choose right)")
plt.title("Psychometric curves by genotype and previous choice")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===========================
# 2. Curves split by previous outcome
# ===========================
plt.figure(figsize=(8,6))

for genotype, label, color in zip([0, 1], ["KO", "WT"], [ko_color, wt_color]):
    for prev_outcome, ls, suffix in zip([0, 1], ["--", "-"], ["Prev Timeout", "Prev Reward"]):
        pred_df = pd.DataFrame({
            "stim_z": stim_range,  # vary current stimulus for psychometric curve
            "prev_stim_z": 0,  # neutral previous stimulus
            "previous_choice": 0.5,  # neutral: halfway between left/right
            "previous_outcome": prev_outcome,  # vary
            "genotype_WT": genotype
        })
        # Interaction terms if present
        if "genotype_x_prev_outcome" in interaction_result.params:
            pred_df["genotype_x_prev_outcome"] = pred_df["genotype_WT"] * pred_df["previous_outcome"]
        if "genotype_x_prev_choice" in interaction_result.params:
            pred_df["genotype_x_prev_choice"] = pred_df["genotype_WT"] * pred_df["previous_choice"]
        if "genotype_x_prev_stim" in interaction_result.params:
            pred_df["genotype_x_prev_stim"] = pred_df["genotype_WT"] * pred_df["prev_stim_z"]
        if "genotype_x_stim" in interaction_result.params:
            pred_df["genotype_x_stim"] = pred_df["genotype_WT"] * pred_df["stim_z"]

        # Predict probabilities safely
        pred_prob = predict_prob(pred_df, interaction_result)

        # Plot
        plt.plot(stim_range, pred_prob, linestyle=ls, linewidth=2, color=color,
                 label=f"{label}, {suffix}")

# Stimulus-only curves (neutral previous choice/outcome)
for genotype, label, color in zip([0,1], ["KO","WT"], [ko_color, wt_color]):
    pred_df = pd.DataFrame({
        "stim_z": stim_range,
        "prev_stim_z": 0,
        "previous_outcome": 0.5,  # neutral
        "previous_choice": 0.5,  # neutral
        "genotype_WT": genotype
    })
    if "genotype_x_prev_outcome" in interaction_result.params:
        pred_df["genotype_x_prev_outcome"] = pred_df["genotype_WT"] * pred_df["previous_outcome"]
    if "genotype_x_prev_choice" in interaction_result.params:
        pred_df["genotype_x_prev_choice"] = pred_df["genotype_WT"] * pred_df["previous_choice"]
    if "genotype_x_prev_stim" in interaction_result.params:
        pred_df["genotype_x_prev_stim"] = genotype * pred_df["prev_stim_z"]
    if "genotype_x_stim" in interaction_result.params:
        pred_df["genotype_x_stim"] = pred_df["genotype_WT"] * pred_df["stim_z"]

    pred_prob = predict_prob(pred_df, interaction_result)

    plt.plot(stim_range, pred_prob, linestyle='-', linewidth=2, color=color, alpha=0.3,
             label=f"{label}, Stim only")

plt.xlabel("Z-scored Stimulus Amplitude")
plt.ylabel("P(choose right)")
plt.title("Psychometric curves by genotype and previous outcome")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===========================
# 3. Curves split by previous stimulus
# ===========================
plt.figure(figsize=(8,6))

prev_stim_values = np.linspace(trial_df["prev_stim_z"].min(),
                               trial_df["prev_stim_z"].max(), 3)  # small number for curve split

for genotype, label, color in zip([0,1], ["KO","WT"], [ko_color, wt_color]):
    for prev_stim, ls, suffix in zip(prev_stim_values, ["--", "-", ":"], [f"Prev Stim {s:.1f}" for s in prev_stim_values]):
        pred_df = pd.DataFrame({
            "stim_z": stim_range,  # vary current stimulus
            "prev_stim_z": prev_stim,  # split curves by previous stimulus
            "previous_choice": 0.5,  # neutral
            "previous_outcome": 0.5,  # neutral
            "genotype_WT": genotype
        })
        # Interaction terms if present
        if "genotype_x_prev_outcome" in interaction_result.params:
            pred_df["genotype_x_prev_outcome"] = pred_df["genotype_WT"] * pred_df["previous_outcome"]
        if "genotype_x_prev_choice" in interaction_result.params:
            pred_df["genotype_x_prev_choice"] = pred_df["genotype_WT"] * pred_df["previous_choice"]
        if "genotype_x_prev_stim" in interaction_result.params:
            pred_df["genotype_x_prev_stim"] = genotype * pred_df["prev_stim_z"]
        if "genotype_x_stim" in interaction_result.params:
            pred_df["genotype_x_stim"] = pred_df["genotype_WT"] * pred_df["stim_z"]

        # Predict probabilities
        pred_prob = predict_prob(pred_df, interaction_result)

        # Plot
        plt.plot(stim_range, pred_prob, linestyle=ls, linewidth=2, color=color,
                 label=f"{label}, {suffix}")

# Stimulus-only curves (neutral previous choice/outcome)
for genotype, label, color in zip([0,1], ["KO","WT"], [ko_color, wt_color]):
    pred_df = pd.DataFrame({
        "stim_z": stim_range,
        "prev_stim_z": 0,
        "previous_outcome": 0.5,  # neutral
        "previous_choice": 0.5,  # neutral
        "genotype_WT": genotype
    })
    if "genotype_x_prev_outcome" in interaction_result.params:
        pred_df["genotype_x_prev_outcome"] = pred_df["genotype_WT"] * pred_df["previous_outcome"]
    if "genotype_x_prev_choice" in interaction_result.params:
        pred_df["genotype_x_prev_choice"] = pred_df["genotype_WT"] * pred_df["previous_choice"]
    if "genotype_x_prev_stim" in interaction_result.params:
        pred_df["genotype_x_prev_stim"] = genotype * pred_df["prev_stim_z"]
    if "genotype_x_stim" in interaction_result.params:
        pred_df["genotype_x_stim"] = pred_df["genotype_WT"] * pred_df["stim_z"]

    pred_prob = predict_prob(pred_df, interaction_result)

    plt.plot(stim_range, pred_prob, linestyle='-', linewidth=2, color=color, alpha=0.3,
             label=f"{label}, Stim only")

plt.xlabel("Z-scored Stimulus Amplitude")
plt.ylabel("P(choose right)")
plt.title("Psychometric curves by genotype and previous stimulus")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#coefficient plots---------------------------------------------------------------------------

# 1. Main effects: first GLM (no interactions)
coefs_main = result.params
stderr_main = result.bse
pvals_main = result.pvalues

# Exclude intercept
main_vars = coefs_main.index.drop('const')
coefs_main = coefs_main[main_vars]
stderr_main = stderr_main[main_vars]
pvals_main = pvals_main[main_vars]

# Colors and alpha based on significance
colors_main = ['darkblue' if p < 0.05 else 'lightgrey' for p in pvals_main]

# --------------------------
# 2. Interactions: from interaction GLM
coefs_int = interaction_result.params
stderr_int = interaction_result.bse
pvals_int = interaction_result.pvalues

# Select only interaction terms (add genotype_x_stim_z)
int_vars = ['genotype_x_stim', 'genotype_x_prev_choice', 'genotype_x_prev_outcome', 'genotype_x_prev_stim']
coefs_int = coefs_int[int_vars]
stderr_int = stderr_int[int_vars]
pvals_int = pvals_int[int_vars]

colors_int = ['darkblue' if p < 0.05 else 'lightgrey' for p in pvals_int]

# --------------------------
# Combine for plotting
all_vars = list(main_vars) + list(int_vars)
all_coefs = np.concatenate([coefs_main.values, coefs_int.values])
all_sterr = np.concatenate([stderr_main.values, stderr_int.values])
all_colors = colors_main + colors_int

y_pos = np.arange(len(all_vars))

plt.figure(figsize=(10,6))

# Plot coefficients with 95% CI
for i, var in enumerate(all_vars):
    plt.plot([all_coefs[i]-1.96*all_sterr[i], all_coefs[i]+1.96*all_sterr[i]],
             [y_pos[i]]*2, color=all_colors[i], lw=3)
    plt.plot(all_coefs[i], y_pos[i], 'o', color=all_colors[i], markersize=8,
             markerfacecolor='white' if var in int_vars else 'black')  # differentiate interactions

plt.axvline(0, color='gray', linestyle='--', lw=1)
plt.yticks(y_pos, all_vars)
plt.xlabel('Coefficient (log-odds)')
plt.title('GGLM Coefficients: Main Effects + Interactions')
plt.gca().invert_yaxis()  # largest at top
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# Marginal Effect Plots------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Colors
COLOR_WT = (0.0, 0.702, 0.702)
COLOR_KO = (1.0, 0.647, 0.0)

# Reference values (held fixed for marginal effects)
REF = {
    "stim_z": 0.0,            # mean stimulus
    "prev_stim_z": 0.0,       # neutral previous stimulus
    "previous_choice": 0.5,   # average over L/R
    "previous_outcome": 0.5   # average over reward/timeout
}

def predict_prob(df, model):
    """Predict choice probability from fitted GLM"""
    df = sm.add_constant(df, has_constant="add")
    df = df[model.model.exog_names]
    return model.predict(df)

#Marginal effect of STIMULUS (psychometric curves)-------------------------------------------------------------
stim_range = np.linspace(
    trial_df["stim_z"].min(),
    trial_df["stim_z"].max(),
    100
)

plt.figure(figsize=(8, 6))

for genotype, label, color in zip([0, 1], ["KO", "WT"], [COLOR_KO, COLOR_WT]):

    pred_df = pd.DataFrame({
        "stim_z": stim_range,
        "prev_stim_z": REF["prev_stim_z"],
        "previous_choice": REF["previous_choice"],
        "previous_outcome": REF["previous_outcome"],
        "genotype_WT": genotype
    })

    # Add interaction terms only if present
    if "genotype_x_stim" in interaction_result.params:
        pred_df["genotype_x_stim"] = genotype * pred_df["stim_z"]
    if "genotype_x_prev_choice" in interaction_result.params:
        pred_df["genotype_x_prev_choice"] = genotype * pred_df["previous_choice"]
    if "genotype_x_prev_outcome" in interaction_result.params:
        pred_df["genotype_x_prev_outcome"] = genotype * pred_df["previous_outcome"]
    if "genotype_x_prev_stim" in interaction_result.params:
        pred_df["genotype_x_prev_stim"] = genotype * pred_df["prev_stim_z"]

    prob = predict_prob(pred_df, interaction_result)

    plt.plot(stim_range, prob, lw=3, color=color, label=label)

plt.xlabel("Stimulus (z-scored)")
plt.ylabel("P(choose right)")
plt.title("Marginal effect of stimulus")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Marginal effect of PREVIOUS CHOICE (binary predictor → point plot)------------------------------------------------
plt.figure(figsize=(6, 5))

for genotype, label, color in zip([0, 1], ["KO", "WT"], [COLOR_KO, COLOR_WT]):

    probs = []

    for prev_choice in [0, 1]:

        pred_df = pd.DataFrame({
            "stim_z": REF["stim_z"],
            "prev_stim_z": REF["prev_stim_z"],
            "previous_choice": prev_choice,
            "previous_outcome": REF["previous_outcome"],
            "genotype_WT": genotype
        }, index=[0])

        if "genotype_x_prev_choice" in interaction_result.params:
            pred_df["genotype_x_prev_choice"] = genotype * pred_df["previous_choice"]
        if "genotype_x_prev_outcome" in interaction_result.params:
            pred_df["genotype_x_prev_outcome"] = genotype * pred_df["previous_outcome"]
        if "genotype_x_prev_stim" in interaction_result.params:
            pred_df["genotype_x_prev_stim"] = genotype * pred_df["prev_stim_z"]
        if "genotype_x_stim" in interaction_result.params:
            pred_df["genotype_x_stim"] = genotype * pred_df["stim_z"]

        probs.append(predict_prob(pred_df, interaction_result)[0])

    plt.plot([0, 1], probs, "-o", lw=3, color=color, label=label)

plt.xticks([0, 1], ["Prev Left", "Prev Right"])
plt.ylabel("P(choose right)")
plt.title("Marginal effect of previous choice")
plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Marginal effect of PREVIOUS OUTCOME---------------------------------------------------------------------------
plt.figure(figsize=(6, 5))

for genotype, label, color in zip([0, 1], ["KO", "WT"], [COLOR_KO, COLOR_WT]):
    probs = []
    for prev_outcome in [0, 1]:
        pred_df = pd.DataFrame({
            "stim_z": REF["stim_z"],
            "prev_stim_z": REF["prev_stim_z"],
            "previous_choice": REF["previous_choice"],
            "previous_outcome": prev_outcome,
            "genotype_WT": genotype
        }, index=[0])

        if "genotype_x_prev_choice" in interaction_result.params:
            pred_df["genotype_x_prev_choice"] = genotype * pred_df["previous_choice"]
        if "genotype_x_prev_outcome" in interaction_result.params:
            pred_df["genotype_x_prev_outcome"] = genotype * pred_df["previous_outcome"]
        if "genotype_x_prev_stim" in interaction_result.params:
            pred_df["genotype_x_prev_stim"] = genotype * pred_df["prev_stim_z"]
        if "genotype_x_stim" in interaction_result.params:
            pred_df["genotype_x_stim"] = genotype * pred_df["stim_z"]

        probs.append(predict_prob(pred_df, interaction_result)[0])

    plt.plot([0, 1], probs, "-o", lw=3, color=color, label=label)

plt.xticks([0, 1], ["Prev Timeout", "Prev Reward"])
plt.ylabel("P(choose right)")
plt.title("Marginal effect of previous outcome")
plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Marginal effect of PREVIOUS STIMULUS-----------------------------------------------------------------------

prev_stim_range = np.linspace(
    trial_df["prev_stim_z"].min(),
    trial_df["prev_stim_z"].max(),
    100
)

plt.figure(figsize=(8,6))

for genotype, label, color in zip([0, 1], ["KO", "WT"], [COLOR_KO, COLOR_WT]):

    pred_df = pd.DataFrame({
        "stim_z": REF["stim_z"],                     # hold current stimulus fixed
        "prev_stim_z": prev_stim_range,              # vary previous stimulus
        "previous_choice": REF["previous_choice"],   # neutral
        "previous_outcome": REF["previous_outcome"], # neutral
        "genotype_WT": genotype
    })

    # ---- interaction terms (ONLY if present in model)
    if "genotype_x_prev_stim" in interaction_result.params:
        pred_df["genotype_x_prev_stim"] = genotype * pred_df["prev_stim_z"]

    if "genotype_x_prev_choice" in interaction_result.params:
        pred_df["genotype_x_prev_choice"] = genotype * pred_df["previous_choice"]

    if "genotype_x_prev_outcome" in interaction_result.params:
        pred_df["genotype_x_prev_outcome"] = genotype * pred_df["previous_outcome"]

    if "genotype_x_stim" in interaction_result.params:
        pred_df["genotype_x_stim"] = genotype * pred_df["stim_z"]

    prob = predict_prob(pred_df, interaction_result)

    plt.plot(prev_stim_range, prob, lw=3, color=color, label=label)

plt.xlabel("Previous stimulus (z-scored)")
plt.ylabel("P(choose right)")
plt.title("Marginal effect of previous stimulus")
plt.ylim(0, 1)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Make sure columns exist
trial_df['prev_choice'] = trial_df['previous_choice']
trial_df['prev_outcome'] = trial_df['previous_outcome']
trial_df['stay'] = (trial_df['choice'] == trial_df['prev_choice']).astype(int)
trial_df['win'] = trial_df['prev_outcome'] == 1
trial_df['lose'] = trial_df['prev_outcome'] == 0

#========================================================================
#Categorization vs discrimination
#========================================================================

# Define genotypes and colors
genotypes = [0, 1]  # 0=KO, 1=WT
COLOR_WT = (0.0, 0.702, 0.702)
COLOR_KO = (1.0, 0.647, 0.0)

# Define stimulus categories
low_stims = [1.33, 1.55]
high_stims = [2.66, 2.88]

# Collect per-mouse Δ(right) per category
mice = trial_df['mouse_id'].unique()
category_labels = ['Low', 'High']

# Initialize storage
delta_per_mouse = {cat: {0: [], 1: []} for cat in category_labels}  # cat -> genotype -> list of Δs

for mouse in mice:
    genotype = trial_df[trial_df['mouse_id'] == mouse]['genotype_WT'].iloc[0]

    # Low category Δ
    df_low = trial_df[(trial_df['mouse_id'] == mouse) & (trial_df['stimulus_amplitude'].isin(low_stims))]
    delta_low = df_low.groupby('stimulus_amplitude')['choice'].mean().diff().iloc[-1]  # last diff is high-low
    delta_per_mouse['Low'][genotype].append(delta_low)

    # High category Δ
    df_high = trial_df[(trial_df['mouse_id'] == mouse) & (trial_df['stimulus_amplitude'].isin(high_stims))]
    delta_high = df_high.groupby('stimulus_amplitude')['choice'].mean().diff().iloc[-1]
    delta_per_mouse['High'][genotype].append(delta_high)

# ------------------------
# Normalization per mouse
# ------------------------
for cat in category_labels:
    for g in genotypes:
        vals = np.array(delta_per_mouse[cat][g])
        max_val = np.max(np.abs(vals))
        if max_val > 0:
            delta_per_mouse[cat][g] = vals / max_val
        else:
            delta_per_mouse[cat][g] = vals  # avoid division by zero

# ------------------------
# Plotting per category with individual mice
# ------------------------
x = np.arange(len(category_labels))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 5))

for i, g in enumerate(genotypes):
    means = [np.mean(delta_per_mouse[cat][g]) for cat in category_labels]
    sems = [np.std(delta_per_mouse[cat][g], ddof=1) / np.sqrt(len(delta_per_mouse[cat][g])) for cat in category_labels]
    color = COLOR_KO if g == 0 else COLOR_WT
    ax.bar(x + (-1) ** i * width / 2, means, width, yerr=sems, color=color, capsize=5, label='KO' if g == 0 else 'WT')

    # overlay individual points
    for j, cat in enumerate(category_labels):
        ax.scatter(np.full(len(delta_per_mouse[cat][g]), x[j] + (-1) ** i * width / 2),
                   delta_per_mouse[cat][g],
                   facecolors='none', edgecolors='darkgrey', s=50)

ax.set_xticks(x)
ax.set_xticklabels(category_labels)
ax.set_ylabel("Normalized Discrimination Index")
ax.set_title("Discrimination by Stimulus Category and Genotype")
ax.legend()
ax.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ------------------------
# Statistics
# ------------------------
# Within-genotype: paired tests (low vs high)
for g, label in zip(genotypes, ['KO', 'WT']):
    low_vals = delta_per_mouse['Low'][g]
    high_vals = delta_per_mouse['High'][g]
    t_stat, p_val = stats.ttest_rel(low_vals, high_vals)
    print(f"{label} Low vs High: t={t_stat:.2f}, p={p_val:.3f}")

# Between-genotype: independent tests for each category
for cat in category_labels:
    ko_vals = delta_per_mouse[cat][0]
    wt_vals = delta_per_mouse[cat][1]
    t_stat, p_val = stats.ttest_ind(ko_vals, wt_vals)
    print(f"{cat} category KO vs WT: t={t_stat:.2f}, p={p_val:.3f}")


# Categorization facilitation of discrimination analysis-----------------------------------------------------------

# -------------------------
# Between categories Delta discrimination(20-18um) - Delta discrimination(Combined low + high salience 2um pairs)
# -------------------------

rows = []

for mouse_id, df_mouse in trial_df.groupby('mouse_id'):
    genotype = int(df_mouse['genotype_WT'].iloc[0])

    # Low deltas
    low_pairs = [(1.33, 1.55), (1.55, 1.77), (1.77, 1.99)]
    low_deltas = []
    for a, b in low_pairs:
        if {a, b} <= set(df_mouse['stimulus_amplitude']):
            p_a = df_mouse.loc[df_mouse['stimulus_amplitude'] == a, 'choice'].mean()
            p_b = df_mouse.loc[df_mouse['stimulus_amplitude'] == b, 'choice'].mean()
            low_deltas.append(p_b - p_a)

    # High deltas
    high_pairs = [(2.22, 2.44), (2.44, 2.66), (2.66, 2.88)]
    high_deltas = []
    for a, b in high_pairs:
        if {a, b} <= set(df_mouse['stimulus_amplitude']):
            p_a = df_mouse.loc[df_mouse['stimulus_amplitude'] == a, 'choice'].mean()
            p_b = df_mouse.loc[df_mouse['stimulus_amplitude'] == b, 'choice'].mean()
            high_deltas.append(p_b - p_a)

    # Combined average
    combined_avg = np.nanmean(low_deltas + high_deltas) if (low_deltas + high_deltas) else np.nan

    # Target delta 18-20
    target_pair = (1.99, 2.22)
    if set(target_pair) <= set(df_mouse['stimulus_amplitude']):
        p_low = df_mouse.loc[df_mouse['stimulus_amplitude'] == target_pair[0], 'choice'].mean()
        p_high = df_mouse.loc[df_mouse['stimulus_amplitude'] == target_pair[1], 'choice'].mean()
        target_delta = p_high - p_low
    else:
        target_delta = np.nan

    # Difference
    delta_minus_combined = target_delta - combined_avg

    rows.append({
        'mouse_id': mouse_id,
        'genotype_WT': genotype,
        'delta_minus_combined': delta_minus_combined
    })

combined_df = pd.DataFrame(rows)

# -------------------------
# Split by genotype
# -------------------------
wt = combined_df[combined_df['genotype_WT'] == 1]['delta_minus_combined']
ko = combined_df[combined_df['genotype_WT'] == 0]['delta_minus_combined']

# -------------------------
# Outlier detection
# -------------------------
wt_out = find_outliers_iqr(wt)
ko_out = find_outliers_iqr(ko)

wt_clean = wt[~wt_out]
ko_clean = ko[~ko_out]

# -------------------------
# Descriptives
# -------------------------
print("\nDESCRIPTIVES 18-20 vs low+high avg (full data)")
describe_group("WT", wt)
describe_group("KO", ko)

print("\nDESCRIPTIVES 18-20 vs low+high avg (outliers removed)")
describe_group("WT (clean)", wt_clean)
describe_group("KO (clean)", ko_clean)

# -------------------------
# Stats: Between-genotype
# -------------------------
t_full, p_full = stats.ttest_ind(wt, ko, equal_var=False)
t_clean, p_clean = stats.ttest_ind(wt_clean, ko_clean, equal_var=False)
print(f"\nWT vs KO (full data) 18-20 vs low+high avg: t={t_full:.2f}, p={p_full:.4f}")
print(f"WT vs KO (clean) 18-20 vs low+high avg: t={t_clean:.2f}, p={p_clean:.4f}")

#------------------------
# Effect sizes
#------------------------

def cohens_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (x.mean() - y.mean()) / pooled_sd

def hedges_g(x, y):
    d = cohens_d(x, y)
    nx, ny = len(x), len(y)
    correction = 1 - (3 / (4 * (nx + ny) - 9))
    return d * correction

d = cohens_d(wt, ko)
g = hedges_g(wt, ko)

print(f"Cohen’s d = {d:.3f}")
print(f"Hedges’ g = {g:.3f}")

d_clean = cohens_d(wt_clean, ko_clean)
g_clean = hedges_g(wt_clean, ko_clean)

print(f"Cohen’s d_clean = {d_clean:.3f}")
print(f"Hedges’ g clean = {g_clean:.3f}")

# -------------------------
# Stats: Within-genotype vs zero
# -------------------------
t_wt0, p_wt0 = stats.ttest_1samp(wt_clean, 0)
t_ko0, p_ko0 = stats.ttest_1samp(ko_clean, 0)
print(f"\nWT (clean) vs 0 18-20 vs low+high avg: t={t_wt0:.2f}, p={p_wt0:.4f}")
print(f"KO (clean) vs 0 18-20 vs low+high avg: t={t_ko0:.2f}, p={p_ko0:.4f}")

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(6, 5))

for i, (vals, outliers, color, label) in enumerate(
        zip([wt, ko], [wt_out, ko_out], [COLOR_WT, COLOR_KO], ['WT', 'KO'])
):
    mean = vals.mean()
    sem = vals.std(ddof=1) / np.sqrt(len(vals))
    plt.bar(i, mean, yerr=sem, color=color, alpha=0.6, capsize=6, label=f'{label} vs low+high')

    # Normal points
    plt.scatter(np.full(len(vals), i)[~outliers], vals[~outliers],
                facecolors='none', edgecolors='k', s=70, zorder=10)

    # Outliers in red
    if outliers.any():
        plt.scatter(np.full(len(vals), i)[outliers], vals[outliers],
                    color='red', s=80, label='Outlier' if i == 0 else None, zorder=11)

plt.axhline(0, color='k', linestyle='--')
plt.xticks([0, 1], ['WT', 'KO'])
plt.ylabel("Δ(18-20) − avg Δ(low+high 2µm pairs)")
plt.title("Target pair vs low+high average")
plt.legend()
plt.tight_layout()
plt.show()

#Extract data
# WT mice
for i, (val, out) in enumerate(zip(wt, wt_out)):
    data.append({
        "Mouse": f"WT_{i+1}",
        "Genotype": "WT",
        "Value": val,
        "Outlier": bool(out)
    })

# KO mice
for i, (val, out) in enumerate(zip(ko, ko_out)):
    data.append({
        "Mouse": f"KO_{i+1}",
        "Genotype": "KO",
        "Value": val,
        "Outlier": bool(out)
    })

df = pd.DataFrame(data)
print("Dataset of delta(18-20)-delta(low+high avg)\n", df)

# --------------------------
# Save data for GLM to excel
# --------------------------
with pd.ExcelWriter("source_data_history_effects_all.xlsx") as writer:
    trial_df.to_excel(writer, sheet_name="Trial_Level", index=False)
    coeff_df.to_excel(writer, sheet_name="GLM_Coefficients", index=False)
    psych_df.to_excel(writer, sheet_name="Psychometric_Curves", index=False)

print("Source data saved to 'source_data_history_effects_all.xlsx'")