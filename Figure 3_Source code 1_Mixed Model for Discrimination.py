"""
Created on 14/06/2023
edited 13/02/2026
Ourania Semelidou
Mixed-Effects Model for amplitude discrimination and categorization.

for eLife paper: Altered cognitive processes shape tactile perception in autism.
Figure 3
"""

import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, norm, probplot, friedmanchisquare, mannwhitneyu, ttest_ind, ttest_rel, wilcoxon
import numpy as np
import statsmodels.formula.api as smf

# Read the Excel file
file_path = "DATA DIR/File name.xlsx"
df = pd.read_excel(file_path)

# Extract only the measurement columns
measurements = df.loc[:, 12:26]
print("Columns selected for analysis:")
print(measurements.columns)

# Step 1: Shapiro-Wilk test at each Amplitude point
print("Shapiro-Wilk Normality Tests:")
for col in measurements.columns:
    for genotype in df['Genotype'].unique():
        data = df[df['Genotype'] == genotype][col]
        stat, p = shapiro(data)
        print(f"Amplitude {col}, Genotype {genotype}: W={stat:.3f}, p={p:.4f} {'(Not Normal)' if p < 0.05 else '(Normal)'}")

# Interaction Effect (Genotype × Amplitude): Mixed-Effects Model
# Reshape to long format
long_df = df.melt(
    id_vars=['Subject', 'Genotype'],
    value_vars=measurements.columns,
    var_name='Amplitude',
    value_name='Score'
)
long_df['Genotype'] = pd.Categorical(long_df['Genotype'], categories=['WT', 'KO'])
long_df['Amplitude'] = pd.Categorical(long_df['Amplitude'], ordered=True)

# Fit a linear mixed-effects model
model = smf.mixedlm("Score ~ Genotype * Amplitude", long_df, groups=long_df["Subject"])
result = model.fit()
print(result.summary())