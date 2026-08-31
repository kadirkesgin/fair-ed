import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import dice_ml
from dice_ml import Data, Model, Dice
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/Users/kadirkesgin/Documents/akademikcalismalar/2026/mart2026/education_truth"
os.chdir(BASE_DIR)

print("Veri yükleniyor...")
df_real = pd.read_csv("student-mat.csv", sep=";")
df_real['passed'] = (df_real['G3'] >= 10).astype(int)
df_real['SES'] = (df_real['Medu'] > 2).astype(int)
features_real = ['age', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'SES']
target_real = 'passed'

df_model_real = df_real[features_real + [target_real]].copy()
X_train, X_test, y_train, y_test = train_test_split(df_model_real.drop(columns=[target_real]), df_model_real[target_real], test_size=0.2, random_state=42)

model = XGBClassifier(random_state=42, eval_metric='logloss').fit(X_train, y_train)

d_real = Data(dataframe=df_model_real, continuous_features=['age', 'studytime', 'failures', 'absences', 'freetime', 'goout'], outcome_name=target_real)
m_real = Model(model=model, backend="sklearn")
exp = Dice(d_real, m_real, method="random")

y_pred = model.predict(X_test)
failed_students = X_test[y_pred == 0].sample(min(40, len(X_test[y_pred == 0])), random_state=42)
weights = {'studytime': 1.5, 'absences': 1.0, 'freetime': 1.0, 'goout': 1.0}
print(f"Toplam {len(failed_students)} başarısız öğrenci üzerinde etkinlik testi yapılıyor...")

before_probs = []
after_probs = []
validity = 0

for idx in range(len(failed_students)):
    orig = failed_students.iloc[idx:idx+1]
    ses_val = orig['SES'].values[0]
    
    # Before Probability
    prob_before = model.predict_proba(orig)[0][1] # Probability of Class 1 (Pass)
    before_probs.append(prob_before)
    
    try:
        # After: Fair Optimization
        dice_prop = exp.generate_counterfactuals(orig, total_CFs=5, desired_class="opposite", features_to_vary=['studytime', 'absences', 'freetime', 'goout'])
        cfs_prop = dice_prop.cf_examples_list[0].final_cfs_df
        
        # En düşük maliyetli (en adil) müdahaleyi seçme
        best_cost = float('inf')
        best_cf = None
        for j in range(len(cfs_prop)):
            diffs = 0
            for feat in weights:
                diffs += weights[feat] * abs(cfs_prop.iloc[j:j+1].reset_index(drop=True).loc[0, feat] - orig.reset_index(drop=True).loc[0, feat])
            c = diffs * (2.0 if ses_val == 0 else 1.0)
            if c < best_cost:
                best_cost = c
                best_cf = cfs_prop.iloc[j:j+1]
                
        # After Probability
        best_cf_no_target = best_cf.drop(columns=[target_real], errors='ignore')
        prob_after = model.predict_proba(best_cf_no_target)[0][1]
        after_probs.append(prob_after)
        validity += 1
    except:
        after_probs.append(np.nan) # Bulunamadı

validity_rate = (validity / len(failed_students)) * 100

data = pd.DataFrame({
    'Student_ID': range(len(before_probs)),
    'Pre-Intervention Probability': before_probs,
    'Post-Intervention Probability': after_probs
}).dropna()

# ==============================================================================
# FIGURE 4: Intervention Efficacy (Density Plot)
# ==============================================================================
plt.figure(figsize=(10, 6))
sns.kdeplot(data=data['Pre-Intervention Probability'], fill=True, color="#E63946", label="Before Intervention (Failed)", alpha=0.6)
sns.kdeplot(data=data['Post-Intervention Probability'], fill=True, color="#2A9D8F", label="After Fair Recourse (Recommended)", alpha=0.6)

plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label="Decision Boundary (Pass/Fail)")
# plt.title removed for LaTeX
plt.xlabel("Predicted Probability of Passing ($P(Y=1|X)$)", fontsize=14)
plt.ylabel("Density (Number of Students)", fontsize=14)
plt.legend(fontsize=12, loc="upper right")
plt.xlim(-0.1, 1.1)

# Annotations
plt.text(0.15, plt.ylim()[1]*0.5, f"Avg Pre: {data['Pre-Intervention Probability'].mean():.2f}", color="darkred", fontsize=12, fontweight='bold')
plt.text(0.70, plt.ylim()[1]*0.5, f"Avg Post: {data['Post-Intervention Probability'].mean():.2f}", color="darkgreen", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("sci_fig4_intervention_efficacy.png", dpi=300)

print(f"Çalışma tamamlandı. Müdahale Etkinliği Grafiği Şuraya Kaydedildi: sci_fig4_intervention_efficacy.png")
print(f"Başarı (Geçerlilik) Oranı: %{validity_rate:.1f}")
print(f"Müdahale Öncesi Ortalama Geçme İhtimali: {data['Pre-Intervention Probability'].mean():.2f}")
print(f"Müdahale Sonrası Ortalama Geçme İhtimali: {data['Post-Intervention Probability'].mean():.2f}")
