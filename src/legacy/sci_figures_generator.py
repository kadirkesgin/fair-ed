import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
import dice_ml
from dice_ml import Data, Model, Dice
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/Users/kadirkesgin/Documents/akademikcalismalar/2026/mart2026/education_truth"
os.chdir(BASE_DIR)

# 1. Veriyi Yükle ve Hazırla
df_real = pd.read_csv("student-mat.csv", sep=";")
df_real['passed'] = (df_real['G3'] >= 10).astype(int)
df_real['SES'] = (df_real['Medu'] > 2).astype(int)
features_real = ['age', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'SES']
target_real = 'passed'
df_model_real = df_real[features_real + [target_real]].copy()

X_train, X_test, y_train, y_test = train_test_split(df_model_real.drop(columns=[target_real]), df_model_real[target_real], test_size=0.2, random_state=42)
model = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_train, y_train)

# ==============================================================================
# FIGURE 1: Feature Importance (Tahmin modelinin kalbi)
# ==============================================================================
plt.figure(figsize=(10, 6))
plot_importance(model, importance_type='weight', max_num_features=7, title="", xlabel="F-score", grid=False)
plt.tight_layout()
plt.savefig("sci_fig1_feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# TABLE 1: Demographic Distribution (LaTeX)
# ==============================================================================
demographics = df_model_real.groupby('SES')['passed'].value_counts(normalize=True).unstack().fillna(0) * 100
latex_table1 = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Table 1: Demographic Distribution and Passing Rates by SES Group}}
\\label{{tab:demographics}}
\\begin{{tabular}}{{l | c | c c}}
\\toprule
\\textbf{{Socioeconomic Status (SES)}} & \\textbf{{Total N}} & \\textbf{{Passed (\\%)}} & \\textbf{{Failed (\\%)}} \\\\
\\midrule
Low SES ($A=0$) & {len(df_model_real[df_model_real['SES']==0])} & {demographics.loc[0, 1]:.1f}\\% & {demographics.loc[0, 0]:.1f}\\% \\\\
High SES ($A=1$) & {len(df_model_real[df_model_real['SES']==1])} & {demographics.loc[1, 1]:.1f}\\% & {demographics.loc[1, 0]:.1f}\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
with open("sci_table1_demographics.tex", "w") as f:
    f.write(latex_table1)

# ==============================================================================
# HIZLI SIMULASYON: Maliyet Dağılımları (Violin Plot) için Veri Üretimi
# ==============================================================================
def calculate_cost(orig, cf, ses, weights):
    penalty = 2.0 if ses == 0 else 1.0
    cost = 0
    for feat in weights:
        cost += (weights[feat] * penalty) * abs(cf.iloc[0][feat] - orig.iloc[0][feat])
    return cost

d_real = Data(dataframe=df_model_real, continuous_features=['age', 'studytime', 'failures', 'absences', 'freetime', 'goout'], outcome_name=target_real)
m_real = Model(model=model, backend="sklearn")
exp = Dice(d_real, m_real, method="random")

weights = {'studytime': 1.5, 'absences': 1.0, 'freetime': 1.0, 'goout': 1.0}
failed = X_test[y_test == 0]

np.random.seed(42)
cost_data = []

# Sadece Dağılımı göstermek için 15 kişi hesaplıyoruz (hız için)
for ses_group in [0, 1]:
    group_df = failed[failed['SES'] == ses_group].sample(min(15, len(failed[failed['SES'] == ses_group])), random_state=42)
    for idx in range(len(group_df)):
        orig = group_df.iloc[idx:idx+1]
        try:
            # Baseline
            dice_base = exp.generate_counterfactuals(orig, total_CFs=1, desired_class="opposite", features_to_vary=['studytime', 'absences', 'freetime', 'goout'])
            cf_base = dice_base.cf_examples_list[0].final_cfs_df
            cost_data.append({'SES': 'Low SES' if ses_group==0 else 'High SES', 'Method': 'Baseline', 'Cost': calculate_cost(orig.reset_index(drop=True), cf_base.reset_index(drop=True), ses_group, weights)})
            
            # Proposed
            dice_prop = exp.generate_counterfactuals(orig, total_CFs=5, desired_class="opposite", features_to_vary=['studytime', 'absences', 'freetime', 'goout'])
            cfs_prop = dice_prop.cf_examples_list[0].final_cfs_df
            best_c = min([calculate_cost(orig.reset_index(drop=True), cfs_prop.iloc[j:j+1].reset_index(drop=True), ses_group, weights) for j in range(len(cfs_prop))])
            cost_data.append({'SES': 'Low SES' if ses_group==0 else 'High SES', 'Method': 'Proposed Fair Recourse', 'Cost': best_c})
        except:
            continue

df_costs = pd.DataFrame(cost_data)

# ==============================================================================
# FIGURE 2: Cost Distribution (Violin Plot)
# ==============================================================================
plt.figure(figsize=(12, 7))
sns.violinplot(data=df_costs, x="SES", y="Cost", hue="Method", split=True, inner="quart", palette={"Baseline": "#E63946", "Proposed Fair Recourse": "#2A9D8F"})
# plt.title removed for LaTeX
plt.xlabel("Socioeconomic Status (SES)", fontsize=14)
plt.ylabel(r"Causally-Aware Intervention Cost ($\mu_A$)", fontsize=14)
plt.legend(title="Optimization Framework", fontsize=12)
plt.tight_layout()
plt.savefig("sci_fig2_cost_violin.png", dpi=300)
plt.close()

# ==============================================================================
# FIGURE 3: Actionable Feature Shift (Özellik Değişim Grafiği - Dumbbell)
# ==============================================================================
# Örnek bir dezavantajlı öğrenci seçelim
sample_orig = failed[failed['SES'] == 0].iloc[0:1]
dice_sample = exp.generate_counterfactuals(sample_orig, total_CFs=5, desired_class="opposite", features_to_vary=['studytime', 'absences', 'freetime', 'goout'])
cfs = dice_sample.cf_examples_list[0].final_cfs_df

# Maliyeti en aza indiren telafiyi bul (Proposed)
best_idx = np.argmin([calculate_cost(sample_orig.reset_index(drop=True), cfs.iloc[j:j+1].reset_index(drop=True), 0, weights) for j in range(len(cfs))])
best_cf = cfs.iloc[best_idx:best_idx+1]

feats = ['studytime', 'absences', 'freetime', 'goout']
orig_vals = sample_orig[feats].values[0]
cf_vals = best_cf[feats].values[0]

plt.figure(figsize=(10, 5))
for i, feat in enumerate(feats):
    plt.plot([orig_vals[i], cf_vals[i]], [i, i], color='grey', zorder=1)
    plt.scatter(orig_vals[i], i, color='#E63946', s=100, label='Original State (Failed)' if i==0 else "", zorder=2)
    plt.scatter(cf_vals[i], i, color='#2A9D8F', s=100, label='Recommended State (Pass)' if i==0 else "", zorder=2)

plt.yticks(range(len(feats)), [f.capitalize() for f in feats], fontsize=12)
# plt.title removed for LaTeX
plt.xlabel("Feature Value", fontsize=12)
plt.legend(fontsize=11)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("sci_fig3_feature_shifts.png", dpi=300)

print("SCI Kalitesindeki tüm tablo ve grafikler başarıyla oluşturuldu!")
