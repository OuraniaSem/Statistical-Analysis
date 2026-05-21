"""created on 12/11/2025; modified on 19/02/2026 to add effect size and within group comparisons
Ourania Semelidou
Analysis of datasets to run between- and within-group statistics (with and without bonferroni correction),
detect outliers, and calculate effect size.
Generated figures depict outliers in red.
-- first part: between group comparisons; second part: within group comparisons

for eLife paper: Altered cognitive processes shape tactile perception in autism.

Install dependencies:
numpy==1.26.4
pandas==2.2.1
scipy==1.12.0
matplotlib==3.8.2
seaborn==0.12.2
statsmodels==1.2.0
openpyxl==3.1.2"""

import numpy as np
import pandas as pd
import scipy.stats as sc
from scipy.stats import zscore
from scipy.stats import normaltest, ttest_ind, mannwhitneyu, ttest_rel, wilcoxon, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

# 1. Load the dataset
Learning_curves = "DATA_DIR/Learning_curves.xlsx"
slopes_df = pd.read_excel(Learning_curves)
print("Data loaded:")
print(slopes_df.head())

#Compare to 0 (analysis used for Fig. 2A - comparing criterion c to 0)
# for genotype, group in slopes_df.groupby('Genotype'):
#     c_values = group['c'].values
#     stat, p = stats.wilcoxon(c_values - 0)  # compare to 0
#     print(f"{genotype}: Wilcoxon W={stat:.3f}, p={p:.3f}")
#     if p < 0.05:
#         print(f"{genotype} shows a systematic bias (c differs from 0).\n")
#     else:
#         print(f"{genotype} is unbiased (c not significantly different from 0).\n")

# Name of the column to analyze
value_col = "Intermediate_CorrectRate"  # replace with column of interest

# 2. Check normality of slope distribution (Shapiro–Wilk test)
for genotype, group in slopes_df.groupby('Genotype'):
    stat, p = shapiro(group[value_col])
    print(f"{genotype}: Shapiro–Wilk W={stat:.3f}, p={p:.3f}")
    if p < 0.05:
        print(f"{genotype} distribution deviates from normality.\n")
    else:
        print(f"{genotype} distribution is approximately normal.\n")

# Compute modified Z-scores within each genotype
slopes_df['modified_z'] = slopes_df.groupby('Genotype')[value_col].transform(
    lambda x: 0.6745 * (x - np.median(x)) / np.median(np.abs(x - np.median(x))))

outliers_modz = slopes_df[np.abs(slopes_df['modified_z']) > 3.5]
print("\nPotential outliers per genotype (|modified_z| > 3.5):")
print(outliers_modz if not outliers_modz.empty else "None detected.")

# 4. Define subsets
no_outliers_df = slopes_df[np.abs(slopes_df['modified_z']) <= 3.5]

plt.figure(figsize=(6,4))
sns.boxplot(data=slopes_df, x='Genotype', y=value_col, showfliers=False)

# Highlight potential outliers in red
outliers = slopes_df[np.abs(slopes_df['modified_z']) > 3.5]
sns.scatterplot(data=outliers, x='Genotype', y=value_col, color='red', s=120, label='Detected outlier')

sns.swarmplot(data=slopes_df, x='Genotype', y=value_col, color='black', alpha=0.6)
plt.legend()
plt.title('Boxplot with flagged outliers')
plt.show()

# 5. Compare genotypes before and after outlier exclusion
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_sd = np.sqrt(((nx - 1) * np.var(x, ddof=1) +
                         (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(y) - np.mean(x)) / pooled_sd

def hedges_g(x, y):
    d = cohens_d(x, y)
    nx, ny = len(x), len(y)
    correction = 1 - (3 / (4 * (nx + ny) - 9))
    return d * correction

def compare_groups(df, label):
    wt = df[df['Genotype'] == 'WT'][value_col]
    ko = df[df['Genotype'] == 'KO'][value_col]

    print(f"\n--- {label} ---")
    print(f"WT (n={len(wt)}): mean={wt.mean():.3f}, SD={wt.std():.3f}")
    print(f"KO (n={len(ko)}): mean={ko.mean():.3f}, SD={ko.std():.3f}")

    # Normality
    p_wt = shapiro(wt).pvalue if len(wt) >= 3 else 1
    p_ko = shapiro(ko).pvalue if len(ko) >= 3 else 1

    if p_wt >= 0.05 and p_ko >= 0.05:
        t_stat, p_val = ttest_ind(wt, ko, equal_var=True)
        print(f"T-test: t={t_stat:.3f}, p={p_val:.3f}")
    else:
        u_stat, p_val = mannwhitneyu(wt, ko, alternative='two-sided')
        print(f"Mann–Whitney U: U={u_stat:.3f}, p={p_val:.3f}")

    # Effect sizes
    d = cohens_d(wt, ko)
    g = hedges_g(wt, ko)

    print(f"Cohen’s d = {d:.3f}")
    print(f"Hedges’ g = {g:.3f}")

    return d, g

print(outliers_modz['Genotype'].value_counts())
d_before, g_before = compare_groups(slopes_df, "Before outlier exclusion")
d_after, g_after = compare_groups(no_outliers_df, "After outlier exclusion")


# Analysis with bonferroni correction----------------------------------------------------------
""""alternative analysis script for multiple comparisons"""
#
# ---------- Effect size functions ----------
# def cohens_d(x, y):
#     nx, ny = len(x), len(y)
#     pooled_sd = np.sqrt(
#         ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
#         / (nx + ny - 2)
#     )
#     return (np.mean(x) - np.mean(y)) / pooled_sd
#
# def hedges_g(x, y):
#     d = cohens_d(x, y)
#     nx, ny = len(x), len(y)
#     correction = 1 - (3 / (4 * (nx + ny) - 9))
#     return d * correction
#
# # Load data
# sens = pd.read_excel(
#     r"D:\vibrotactile task\2AFC\Discrimination\5batches\Figure5_miss rate\2AFC_Miss_rate_LowSalience_ForStats.xlsx"
# )
#
# cols_names = list(sens.columns)
# cols_names.remove('Genotype')
#
# results = []
#
# print("\nPost-hoc WT vs KO (Welch t-test)")
# print("=================================")
#
# # Run tests
# for amp in cols_names:
#     data_wt = sens[amp][sens['Genotype'] == 'WT'].dropna()
#     data_ko = sens[amp][sens['Genotype'] == 'KO'].dropna()
#
#     res = sc.ttest_ind(data_wt, data_ko, equal_var=False)  # Welch test
#
#     d = cohens_d(data_wt, data_ko)
#     g = hedges_g(data_wt, data_ko)
#
#     results.append({
#         "Amplitude": amp,
#         "t": res.statistic,
#         "p_uncorrected": res.pvalue,
#         "Cohens_d": d,
#         "Hedges_g": g,
#         "n_WT": len(data_wt),
#         "n_KO": len(data_ko)
#     })
#
#     print(f"{amp}")
#     print(f"t = {res.statistic:.4f}")
#     print(f"p = {res.pvalue:.4f}")
#     print(f"Hedges g = {g:.3f}")
#     print("-------------")
#
# # Bonferroni correction
#
# pvals = [r["p_uncorrected"] for r in results]
# reject, pvals_corrected, _, _ = multipletests(pvals, method='bonferroni')
#
# print("\nBonferroni corrected results")
# print("=================================")
#
# for i, r in enumerate(results):
#     print(f"{r['Amplitude']}")
#     print(f"t = {r['t']:.4f}")
#     print(f"p (corrected) = {pvals_corrected[i]:.4f}")
#     print(f"Significant? {reject[i]}")
#     print("-------------")
#
# alpha_corrected = 0.05 / len(results)
# print(f"\nCorrected alpha = {alpha_corrected:.4f}")
#
# #-------------Within group comparisons-----------
#
# def modified_z_scores(x):
#     """
#     Compute modified Z-scores using MAD.
#     """
#     x = np.asarray(x)
#     median = np.median(x)
#     mad = np.median(np.abs(x - median))
#
#     if mad == 0:
#         return np.zeros_like(x)
#
#     return 0.6745 * (x - median) / mad
#
# def cohens_dz(x, y):
#     """Cohen's dz for paired samples"""
#     diff = x - y
#     return np.mean(diff) / np.std(diff, ddof=1)
#
# def hedges_g_paired(x, y):
#     """Hedges' g for paired samples"""
#     dz = cohens_dz(x, y)
#     n = len(x)
#     correction = 1 - (3 / (4 * n - 1))
#     return dz * correction
#
# def run_paired_stats(x, y, alpha=0.05):
#     """
#     Run paired t-test or Wilcoxon based on Shapiro normality.
#     Returns rich stats dictionary.
#     """
#     diff = x - y
#     mean_diff = np.mean(diff)
#     sd_diff = np.std(diff, ddof=1)
#     p_norm = shapiro(diff).pvalue if len(diff) >= 3 else 1
#
#     if p_norm >= alpha:
#         stat, p = ttest_rel(x, y)
#         test = "Paired t-test"
#         df = len(diff) - 1
#         stat_str = f"t({df}) = {stat:.3f}"
#     else:
#         stat, p = wilcoxon(x, y)
#         test = "Wilcoxon signed-rank"
#         stat_str = f"W = {stat:.3f}"
#
#     dz = cohens_dz(x, y)
#     g = hedges_g_paired(x, y)
#
#     return {
#         "test": test,
#         "stat_str": stat_str,
#         "p": p,
#         "mean_diff": mean_diff,
#         "sd_diff": sd_diff,
#         "dz": dz,
#         "g": g,
#         "n": len(diff)
#     }
#
# # Main within-group comparison
#
# def within_group_comparison(df, genotype, col1, col2,
#                             alpha=0.05,
#                             modz_thresh=3.5):
#     data = df[df['Genotype'] == genotype][[col1, col2]].dropna()
#     x_raw = data[col1].values
#     y_raw = data[col2].values
#     diff_raw = x_raw - y_raw
#
#     print(f"\n=== {genotype}: {col1} vs {col2} ===")
#     print(f"N (raw) = {len(diff_raw)}")
#
#     # Outlier detection
#     modz = modified_z_scores(diff_raw)
#     outliers = np.abs(modz) > modz_thresh
#     n_out = outliers.sum()
#     print(f"Modified Z threshold = ±{modz_thresh}")
#     print(f"Outliers detected = {n_out}")
#
#     # Run stats WITH outliers
#     res_with = run_paired_stats(x_raw, y_raw, alpha)
#
#     # Run stats WITHOUT outliers
#     if n_out > 0:
#         x_clean = x_raw[~outliers]
#         y_clean = y_raw[~outliers]
#         res_without = run_paired_stats(x_clean, y_clean, alpha)
#     else:
#         res_without = res_with
#
#     def print_res(label, r):
#         print(f"\n{label}")
#         print(f"  Test: {r['test']}")
#         print(f"  {r['stat_str']}, p = {r['p']:.4f}")
#         print(f"  Mean diff ± SD = {r['mean_diff']:.3f} ± {r['sd_diff']:.3f}")
#         print(f"  Cohen’s dz = {r['dz']:.3f}")
#         print(f"  Hedges’ g = {r['g']:.3f}")
#         print(f"  N = {r['n']}")
#
#     print_res("WITH outliers", res_with)
#     print_res("WITHOUT outliers", res_without)
#
#     return res_with, res_without
#
# # Comparisons and Bonferroni correction
# comparisons = [
#     ('WT', 'amp12', 'amp14'),
#     ('KO', 'amp12', 'amp14')
# ]
#
# p_values_with = []
# p_values_without = []
# labels = []
#
# for genotype, c1, c2 in comparisons:
#     res_with, res_without = within_group_comparison(
#         slopes_df, genotype, c1, c2
#     )
#     p_values_with.append(res_with['p'])
#     p_values_without.append(res_without['p'])
#     labels.append(f"{genotype}: {c1} vs {c2}")
#
# alpha = 0.05
# m = len(p_values_with)
# alpha_bonf = alpha / m
#
# print("\n--- Bonferroni correction ---")
# print(f"Number of comparisons = {m}")
# print(f"Corrected alpha = {alpha_bonf:.4f}")
#
# print("\nResults WITH outliers:")
# for label, p in zip(labels, p_values_with):
#     sig = "SIGNIFICANT" if p < alpha_bonf else "n.s."
#     print(f"{label}: p = {p:.4f} → {sig}")
#
# print("\nResults WITHOUT outliers:")
# for label, p in zip(labels, p_values_without):
#     sig = "SIGNIFICANT" if p < alpha_bonf else "n.s."
#     print(f"{label}: p = {p:.4f} → {sig}")
