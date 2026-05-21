"""created on 12/11/2025
Ourania Semelidou
Analysis of learning curves in 2AFC task - only for mice that completed the training phase.
Includes:
Mixed linear model for group-level learning curves (intercept and slope by genotype).
Per-animal  learning slopes over all days.
Intermediate slopes (middle 3 days).
SD per animal to assess intra-individual variability.

for eLife paper: Altered cognitive processes shape tactile perception in autism.
Figure Suppl.2
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 1. Load Excel file ----------------------------
file_path = "DATA DIR/File name.xlsx"
df = pd.read_excel(file_path)

# Long format
df_long = df.melt(id_vars=["Animal", "Genotype"],
                  var_name="Day",
                  value_name="Value")

# Extract numeric day
df_long['Day'] = df_long['Day'].str.extract('(\d+)').astype(int)

# Drop missing values
df_long = df_long.dropna(subset=['Value'])

# Set Day 1 as baseline
df_long['Day_adj'] = df_long['Day'] - 1

# Ensure WT is reference
df_long['Genotype'] = pd.Categorical(df_long['Genotype'], categories=["WT", "KO"])

# 2. Mixed Linear Model ----------------------------
model = smf.mixedlm(
    "Value ~ Day_adj * Genotype",
    df_long,
    groups=df_long["Animal"],
    re_formula="~Day_adj"   # random intercept + slope
)

result = model.fit()
print(result.summary())


# 3. Extract group-level parameters ----------------------------

intercept_WT = result.params['Intercept']
slope_WT = result.params['Day_adj']

intercept_KO = intercept_WT + result.params.get('Genotype[T.KO]', 0)
slope_KO = slope_WT + result.params.get('Day_adj:Genotype[T.KO]', 0)

print("\n=== Group-level estimates ===")
print(f"WT → Intercept (Day 1): {intercept_WT:.3f}, Slope: {slope_WT:.3f}")
print(f"KO → Intercept (Day 1): {intercept_KO:.3f}, Slope: {slope_KO:.3f}")


# 4. Intraindividual variability: SD per animal (raw variability) ----------------------------

sd_per_animal = df_long.groupby('Animal')['Value'].std().reset_index()
sd_per_animal.columns = ['Animal', 'SD_Value']

# Add genotype info
sd_per_animal = sd_per_animal.merge(df[['Animal', 'Genotype']], on='Animal', how='left')

print("\n=== Standard deviation per animal ===")
print(sd_per_animal)


# 5. Individual learning slopes (FULL training - all days) ----------------------------

full_slopes = []

for animal, group in df_long.groupby('Animal'):
    if len(group) >= 2:
        x = group['Day'].values
        y = group['Value'].values
        slope = np.polyfit(x, y, 1)[0]
        intercept = np.polyfit(x, y, 1)[1]

        full_slopes.append({
            "Animal": animal,
            "Full_Slope": slope,
            "Full_Intercept": intercept
        })

full_slopes_df = pd.DataFrame(full_slopes)
print("\n=== Individual learning slopes (all days) ===")
print(full_slopes_df)

# 6. Intermediate-stage slopes (middle 3 days of each animal) ----------------------------

# Intermediate-stage (middle 3 actual days of each animal)
intermediate_results = []

for animal, group in df_long.groupby('Animal'):
    group_sorted = group.sort_values('Day').reset_index(drop=True)

    if len(group_sorted) >= 3:
        mid_idx = len(group_sorted) // 2

        # Select exactly 3 middle data points
        interm_data = group_sorted.iloc[mid_idx-1:mid_idx+2]

        x = interm_data['Day'].values
        y = interm_data['Value'].values

        slope = np.polyfit(x, y, 1)[0]
        correct_rate = y.mean()

        intermediate_results.append({
            "Animal": animal,
            "Genotype": group['Genotype'].iloc[0],
            "Intermediate_Slope": slope,
            "Intermediate_CorrectRate": correct_rate,
            "Days_used": list(interm_data['Day'])  # for verification
        })

intermediate_df = pd.DataFrame(intermediate_results)

print("\n=== Intermediate (middle 3 actual days) ===")
print(intermediate_df)
print(intermediate_df[['Animal', 'Days_used']])

# 7. Merge all per-animal results ----------------------------

final_df = full_slopes_df.merge(sd_per_animal, on="Animal", how="left")
final_df = final_df.merge(intermediate_df, on="Animal", how="left")

print("\n=== Per-animal summary ===")
print(final_df)

# 8. Save results in excel file
final_df.to_excel("Learning_Curves_analysis_results.xlsx", index=False)