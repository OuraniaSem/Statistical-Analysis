"""
Created on 13/02/2026
Ourania Semelidou
Attention in perceptual decision-making under high cognitive load.
Analysis of missed trials during categorization.
Mixed Linear Model analysis by amplitude category (High or Low salience) with effect sizes

for eLife paper: Altered cognitive processes shape tactile perception in autism.
Figure 5

"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ------------------ LOAD DATA ------------------
file_path = "DATA DIR/File name.xlsx"
df = pd.read_excel(file_path)
df['Subject'] = df['Subject'].astype(str)

low_amps  = [12, 14, 16, 18]
high_amps = [20, 22, 24, 26]

# ------------------ OUTLIER DETECTION ------------------
def modified_z(x):
    x = np.asarray(x)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad

# ------------------ EFFECT SIZE  ------------------
def cohen_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_sd = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
    if pooled_sd == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled_sd

def rank_biserial_r(U, nx, ny):
    return 1 - (2*U)/(nx*ny)

# ------------------ ANALYSIS  ------------------
def analyze_amplitudes(df, amplitudes, label):

    print(f"\n================ {label} =================")

    long_df = df.melt(
        id_vars=['Subject', 'Genotype'],
        value_vars=amplitudes,
        var_name='Amplitude',
        value_name='Score'
    )

    long_df['Amplitude'] = long_df['Amplitude'].astype(str)
    long_df['Genotype']  = pd.Categorical(long_df['Genotype'], ['WT', 'KO'])

    # ---------- Normality ----------
    print("\nShapiro–Wilk Normality check:")
    normality = {}
    for amp in amplitudes:
        for geno in ['WT', 'KO']:
            data = long_df[(long_df.Amplitude == str(amp)) & (long_df.Genotype == geno)]['Score']
            if len(data) >= 3:
                p = shapiro(data).pvalue
                normality[(amp, geno)] = p >= 0.05
                print(f"Amp {amp}, {geno}: p={p:.4f} {'Normal' if p >= 0.05 else 'Not Normal'}")
            else:
                normality[(amp, geno)] = False

    # ---------- Mixed Model ----------
    model = smf.mixedlm(
        "Score ~ Genotype * Amplitude",
        long_df,
        groups=long_df["Subject"]
    )
    result = model.fit()
    print("\nMixed Linear Model:")
    print(result.summary())

    # ------------------ POST-HOC  ------------------
    def run_posthoc(long_df, remove_outliers=False):
        results = []
        for amp in long_df['Amplitude'].unique():
            sub = long_df[long_df['Amplitude'] == amp]
            wt = sub[sub['Genotype']=='WT']['Score'].values
            ko = sub[sub['Genotype']=='KO']['Score'].values

            if remove_outliers:
                wt = wt[np.abs(modified_z(wt)) <= 3.5]
                ko = ko[np.abs(modified_z(ko)) <= 3.5]

            n_wt, n_ko = len(wt), len(ko)
            if n_wt < 2 or n_ko < 2:
                results.append({
                    'Amplitude': amp,
                    'Test': 'NA',
                    'p_uncorrected': np.nan,
                    'p_bonferroni': np.nan,
                    'n_WT': n_wt,
                    'n_KO': n_ko,
                    'Effect_size': np.nan,
                    'WT_desc': np.nan,
                    'KO_desc': np.nan
                })
                continue

            # Decide which test
            if normality.get((int(amp), 'WT'), False) and normality.get((int(amp), 'KO'), False):
                stat, p = ttest_ind(wt, ko, equal_var=False)
                es = cohen_d(wt, ko)
                wt_desc = f"{np.mean(wt):.3f} ± {np.std(wt, ddof=1):.3f}"
                ko_desc = f"{np.mean(ko):.3f} ± {np.std(ko, ddof=1):.3f}"
                test_name = "t-test"
            else:
                stat, p = mannwhitneyu(wt, ko)
                es = rank_biserial_r(stat, n_wt, n_ko)
                wt_desc = f"{np.median(wt):.3f} [{np.percentile(wt,25):.3f}-{np.percentile(wt,75):.3f}]"
                ko_desc = f"{np.median(ko):.3f} [{np.percentile(ko,25):.3f}-{np.percentile(ko,75):.3f}]"
                test_name = "Mann–Whitney U"

            results.append({
                'Amplitude': amp,
                'Test': test_name,
                'p_uncorrected': p,
                'n_WT': n_wt,
                'n_KO': n_ko,
                'Effect_size': es,
                'WT_desc': wt_desc,
                'KO_desc': ko_desc
            })

        # Bonferroni correction
        pvals = [r['p_uncorrected'] for r in results if not np.isnan(r['p_uncorrected'])]
        n_tests = len(pvals)
        i = 0
        for r in results:
            if not np.isnan(r['p_uncorrected']):
                r['p_bonferroni'] = min(r['p_uncorrected']*n_tests, 1.0)
                i += 1
            else:
                r['p_bonferroni'] = np.nan

        return results

    print("\nPost-hoc WT vs KO (NO outlier removal):")
    ph_no = run_posthoc(long_df, remove_outliers=False)
    for r in ph_no:
        print(r)

    print("\nPost-hoc WT vs KO (WITH outlier removal):")
    ph_yes = run_posthoc(long_df, remove_outliers=True)
    for r in ph_yes:
        print(r)

# ------------------ MAIN ------------------
analyze_amplitudes(df, low_amps,  "LOW amplitudes (12–18)")
analyze_amplitudes(df, high_amps, "HIGH amplitudes (20–26)")
