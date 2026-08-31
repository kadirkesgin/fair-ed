import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import dice_ml
from dice_ml import Data, Model, Dice

BASE_DIR = "/Users/kadirkesgin/Documents/akademikcalismalar/2026/mart2026/education_truth"
os.makedirs(BASE_DIR, exist_ok=True)
os.chdir(BASE_DIR)

def calculate_cost(orig, cf, ses, weights):
    penalty = 2.0 if ses == 0 else 1.0
    cost = 0
    for feat in weights:
        diff = abs(cf.iloc[0][feat] - orig.iloc[0][feat])
        cost += (weights[feat] * penalty) * diff
    return cost

def evaluate_recourse_fairness(exp, X_test, y_test, features_to_vary, weights, n_samples=30):
    failed = X_test[y_test == 0]
    failed_low_ses = failed[failed['SES'] == 0]
    failed_high_ses = failed[failed['SES'] == 1]
    
    sample_low = failed_low_ses.sample(min(n_samples, len(failed_low_ses)), random_state=42)
    sample_high = failed_high_ses.sample(min(n_samples, len(failed_high_ses)), random_state=42)
    
    avg_cost_low = 0
    avg_cost_high = 0
    
    valid_low = 0
    for idx in range(len(sample_low)):
        orig = sample_low.iloc[idx:idx+1]
        try:
            dice_exp = exp.generate_counterfactuals(orig, total_CFs=1, desired_class="opposite", features_to_vary=features_to_vary)
            cf = dice_exp.cf_examples_list[0].final_cfs_df
            avg_cost_low += calculate_cost(orig.reset_index(drop=True), cf.reset_index(drop=True), 0, weights)
            valid_low += 1
        except Exception:
            continue
            
    valid_high = 0
    for idx in range(len(sample_high)):
        orig = sample_high.iloc[idx:idx+1]
        try:
            dice_exp = exp.generate_counterfactuals(orig, total_CFs=1, desired_class="opposite", features_to_vary=features_to_vary)
            cf = dice_exp.cf_examples_list[0].final_cfs_df
            avg_cost_high += calculate_cost(orig.reset_index(drop=True), cf.reset_index(drop=True), 1, weights)
            valid_high += 1
        except Exception:
            continue
            
    avg_cost_low = avg_cost_low / max(1, valid_low)
    avg_cost_high = avg_cost_high / max(1, valid_high)
    
    rfd = abs(avg_cost_low - avg_cost_high)
    return avg_cost_low, avg_cost_high, rfd

# ==============================================================================
# PART 1: SYNTHETIC DATA
# ==============================================================================
print("--- PART 1: SYNTHETIC CAUSAL DATA VALIDATION ---")
np.random.seed(42)
n_samples = 2000
SES = np.random.binomial(1, 0.5, n_samples)
U = np.random.normal(0, 1, n_samples)
free_time = np.clip(np.random.normal(3 + 2*SES + U, 1), 1, 5).astype(int)
study_time = np.clip(np.random.normal(0.5*free_time + 0.5*U + 1, 1), 1, 4).astype(int)
absences = np.clip(np.random.normal(15 - 5*SES - 2*U, 5), 0, 30).astype(int)
logit_p = -2.0 + 1.5*study_time - 0.2*absences + 1.0*U
prob = 1 / (1 + np.exp(-logit_p))
passed = np.random.binomial(1, prob)

df_synth = pd.DataFrame({'SES': SES, 'freetime': free_time, 'studytime': study_time, 'absences': absences, 'passed': passed})
X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(df_synth.drop(columns=['passed']), df_synth['passed'], test_size=0.2, random_state=42)

model_synth = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_synth_train, y_synth_train)

d_synth = Data(dataframe=df_synth, continuous_features=['freetime', 'studytime', 'absences'], outcome_name='passed')
m_synth = Model(model=model_synth, backend="sklearn")
exp_synth = Dice(d_synth, m_synth, method="random")

synth_weights = {'freetime': 1.0, 'studytime': 1.5, 'absences': 1.0}
cost_low_synth, cost_high_synth, rfd_synth = evaluate_recourse_fairness(exp_synth, X_synth_test, y_synth_test, ['studytime', 'absences', 'freetime'], synth_weights, n_samples=30)
print(f"Synthetic Data RFD: {rfd_synth:.2f}")

# ==============================================================================
# PART 2: REAL DATA
# ==============================================================================
print("\n--- PART 2: REAL-WORLD APPLICATION (UCI DATASET) ---")
if not os.path.exists("student-mat.csv"):
    urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip", "student.zip")
    with zipfile.ZipFile("student.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

df_real = pd.read_csv("student-mat.csv", sep=";")
df_real['passed'] = (df_real['G3'] >= 10).astype(int)
df_real['SES'] = (df_real['Medu'] > 2).astype(int)

features_real = ['age', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'SES']
df_model_real = df_real[features_real + ['passed']].copy()
X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(df_model_real.drop(columns=['passed']), df_model_real['passed'], test_size=0.2, random_state=42)

model_real = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_real_train, y_real_train)

d_real = Data(dataframe=df_model_real, continuous_features=['age', 'studytime', 'failures', 'absences', 'freetime', 'goout'], outcome_name='passed')
m_real = Model(model=model_real, backend="sklearn")
exp_real = Dice(d_real, m_real, method="random")

real_weights = {'studytime': 1.5, 'absences': 1.0, 'freetime': 1.0, 'goout': 1.0}
cost_low_real, cost_high_real, rfd_real = evaluate_recourse_fairness(exp_real, X_real_test, y_real_test, ['studytime', 'absences', 'freetime', 'goout'], real_weights, n_samples=30)
print(f"Real Data RFD: {rfd_real:.2f}")

# ==============================================================================
# PART 3: GENERATING TABLES AND FIGURES
# ==============================================================================
print("\n--- GENERATING GRAPHICS AND LATEX TABLES ---")

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
labels = ['Low SES ($A=0$)', 'High SES ($A=1$)']

# Synthetic Plot
ax[0].bar(labels, [cost_low_synth, cost_high_synth], color=['#FF6B6B', '#4ECDC4'])
ax[0].set_title("Expected Recourse Cost (Synthetic Data Ground Truth)", fontsize=14, fontweight='bold')
ax[0].set_ylabel(r"Average Causally-Aware Cost ($\mu_A$)", fontsize=12)
max_syn = max(cost_low_synth, cost_high_synth)
ax[0].text(0.5, max_syn*1.05 if max_syn > 0 else 1, f"RFD = {rfd_synth:.2f}", ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Real Data Plot
ax[1].bar(labels, [cost_low_real, cost_high_real], color=['#FF6B6B', '#4ECDC4'])
ax[1].set_title("Expected Recourse Cost (Real UCI Student Dataset)", fontsize=14, fontweight='bold')
ax[1].set_ylabel(r"Average Causally-Aware Cost ($\mu_A$)", fontsize=12)
max_real = max(cost_low_real, cost_high_real)
ax[1].text(0.5, max_real*1.05 if max_real > 0 else 1, f"RFD = {rfd_real:.2f}", ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "recourse_cost_comparison.png"), dpi=300)
print(f"Saved figure: {os.path.join(BASE_DIR, 'recourse_cost_comparison.png')}")

latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Comparison of Expected Recourse Costs and Fairness Difference (RFD)}}
\\label{{tab:recourse_results}}
\\begin{{tabular}}{{l | c c | c}}
\\toprule
\\textbf{{Dataset Environment}} & \\textbf{{Low SES Cost ($\\mu_{{A=0}}$)}} & \\textbf{{High SES Cost ($\\mu_{{A=1}}$)}} & \\textbf{{RFD ($\\Delta$)}} \\\\
\\midrule
Synthetic (Ground Truth) & {cost_low_synth:.2f} & {cost_high_synth:.2f} & \\textbf{{{rfd_synth:.2f}}} \\\\
Real-World (UCI Student) & {cost_low_real:.2f} & {cost_high_real:.2f} & \\textbf{{{rfd_real:.2f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

with open(os.path.join(BASE_DIR, "results_table.tex"), "w") as f:
    f.write(latex_table)
    
print(f"Saved LaTeX table: {os.path.join(BASE_DIR, 'results_table.tex')}")
print("\n--- DONE ---")
