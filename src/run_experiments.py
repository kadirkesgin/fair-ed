import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# Set random seed for reproducibility
np.random.seed(42)

# Directory Setup
BASE_DIR = "/Users/kadirkesgin/Documents/akademikcalismalar/2026/mart2026/education_truth"
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("==========================================================")
print("   RUNNING OFFICIAL DAG-INFORMED ACTIONABLE RECOURSE PIPELINE")
print("==========================================================")

# 1. DATA LOADING & COMPOSITE SES INDEX CONSTRUCTION
data_path = os.path.join(BASE_DIR, "student-por.csv")
df = pd.read_csv(data_path, sep=";")
print(f"Loaded dataset: student-por.csv (N = {len(df)})")

# Binary target: passed = 1 if G3 >= 10 else 0
df['passed'] = (df['G3'] >= 10).astype(int)

# Primary Composite SES Index: Medu + Fedu + famsize_small + internet_yes >= median
df['famsize_small'] = (df['famsize'] == 'LE3').astype(int)
df['internet_yes'] = (df['internet'] == 'yes').astype(int)
df['composite_score'] = df['Medu'] + df['Fedu'] + df['famsize_small'] + df['internet_yes']

median_composite = df['composite_score'].median()
df['SES'] = (df['composite_score'] >= median_composite).astype(int)

print(f"Primary Composite SES Index constructed (median threshold = {median_composite}):")
print(f"  Low SES (SES=0): {sum(df['SES'] == 0)} students")
print(f"  High SES (SES=1): {sum(df['SES'] == 1)} students")
print(f"Class Prevalence: Passed (y=1): {sum(df['passed'] == 1)} ({sum(df['passed'] == 1)/len(df):.1%}), Failed (y=0): {sum(df['passed'] == 0)} ({sum(df['passed'] == 0)/len(df):.1%})")

# Frozen feature set
features = ['age', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'SES']
target = 'passed'

X = df[features].copy()
y = df[target].copy()

# 2. PREDICTIVE PERFORMANCE EVALUATION (5-FOLD STRATIFIED CV)
print("\n--- Running 5-Fold Stratified CV Predictive Performance Evaluation ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accs, precs, recs, f1s, aucs = [], [], [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
    
    model_cv = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model_cv.fit(X_tr, y_tr)
    
    y_pred = model_cv.predict(X_va)
    y_proba = model_cv.predict_proba(X_va)[:, 1]
    
    accs.append(accuracy_score(y_va, y_pred))
    precs.append(precision_score(y_va, y_pred, zero_division=0))
    recs.append(recall_score(y_va, y_pred, zero_division=0))
    f1s.append(f1_score(y_va, y_pred, zero_division=0))
    aucs.append(roc_auc_score(y_va, y_proba))

cv_results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
    'Mean': [np.mean(accs), np.mean(precs), np.mean(recs), np.mean(f1s), np.mean(aucs)],
    'SD': [np.std(accs), np.std(precs), np.std(recs), np.std(f1s), np.std(aucs)]
})
cv_results_df.to_csv(os.path.join(TABLES_DIR, "predictive_performance.csv"), index=False)
print("Predictive Performance (5-Fold CV, mean ± SD):")
for idx, row in cv_results_df.iterrows():
    print(f"  {row['Metric']}: {row['Mean']:.4f} ± {row['SD']:.4f}")

# 3. GROUP-INDEPENDENT EVALUATION METRIC & SCORING DEFINITIONS
bounds_norm = {
    'studytime': 1.0 / (4.0 - 1.0),
    'absences': 1.0 / (93.0 - 0.0),
    'freetime': 1.0 / (5.0 - 1.0),
    'goout': 1.0 / (5.0 - 1.0)
}

# Group-independent evaluation cost: Pure normalized feature distance
def compute_standard_eval_cost(orig, cf):
    cost = 0.0
    for feat in bounds_norm:
        diff = abs(cf[feat] - orig[feat])
        cost += bounds_norm[feat] * diff
    return cost

def compute_scoring_cost(orig, cf, ses, method_name):
    if method_name == 'Unconstrained discrete search (DiCE-style)':
        return sum(bounds_norm[ft] * abs(cf[ft] - orig[ft]) for ft in bounds_norm)
    elif 'Actionability-constrained' in method_name or 'Ustun-aligned' in method_name:
        return sum(bounds_norm[ft] * abs(cf[ft] - orig[ft]) for ft in bounds_norm)
    elif method_name == 'Fairness-weighted discrete search':
        w_domain = {'studytime': 1.0, 'absences': 1.5, 'freetime': 1.0, 'goout': 1.0}
        return sum(w_domain[ft] * bounds_norm[ft] * abs(cf[ft] - orig[ft]) for ft in bounds_norm)
    else: # Proposed SES-Sensitive Difficulty-Weighted Recourse
        if ses == 0:
            w_custom = {'studytime': 1.0, 'absences': 2.0, 'freetime': 1.0, 'goout': 1.0}
        else:
            w_custom = {'studytime': 1.0, 'absences': 1.0, 'freetime': 1.0, 'goout': 1.0}
        return sum(w_custom[ft] * bounds_norm[ft] * abs(cf[ft] - orig[ft]) for ft in bounds_norm)

# 4. MULTI-SEED RECOURSE EVALUATION & ABLATION STUDY
seeds = list(range(42, 52)) # 10 random seeds
ablation_methods = [
    'Unconstrained discrete search (DiCE-style)',
    'Actionability-constrained search (Ustun-aligned bounds)',
    'Fairness-weighted discrete search',
    'Proposed SES-Sensitive Difficulty-Weighted Recourse',
    'Manifold nearest-neighbor recourse baseline'
]

seed_results = {m: [] for m in ablation_methods}
pop_flow_records = []
pre_post_probs = {m: {'pre': [], 'post': []} for m in ablation_methods}
feature_shifts_log = []

# Full candidate discrete space: 4 * 94 * 5 * 5 = 9,400 points
# Directional candidate search grid constructed under actionability bounds: K = 700 candidates
unconstrained_deltas = []
for d_s in [-2, -1, 0, 1, 2, 3]:
    for d_a in [0, -2, -5, -8, -12, -15, 2, 5]:
        for d_f in [-2, -1, 0, 1, 2]:
            for d_g in [-2, -1, 0, 1, 2]:
                unconstrained_deltas.append((d_s, d_a, d_f, d_g))

constrained_deltas = []
for d_s in [0, 1, 2, 3]:
    for d_a in [0, -2, -5, -8, -12, -15, -20]:
        for d_f in [-2, -1, 0, 1, 2]:
            for d_g in [-2, -1, 0, 1, 2]:
                constrained_deltas.append((d_s, d_a, d_f, d_g))

GRID_SEARCH_BUDGET_K = len(constrained_deltas) # K = 700 candidates

for seed in seeds:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed)
    clf.fit(X_tr, y_tr)
    
    y_pred_te = clf.predict(X_te)
    failed_mask = (y_pred_te == 0)
    failed_test = X_te[failed_mask]
    
    low_failed = failed_test[failed_test['SES'] == 0]
    high_failed = failed_test[failed_test['SES'] == 1]
    
    pop_flow_records.append({
        'Seed': seed,
        'N_test': len(X_te),
        'N_failed': len(failed_test),
        'N_failed_low': len(low_failed),
        'N_failed_high': len(high_failed)
    })
    
    train_passing = X_tr[y_tr == 1]

    for method in ablation_methods:
        low_costs, high_costs = [], []
        valid_low, valid_high = 0, 0
        pre_list, post_list = [], []
        
        deltas = unconstrained_deltas if 'Unconstrained' in method else constrained_deltas
        deltas_arr = np.array(deltas)
        
        for idx in range(len(failed_test)):
            orig = failed_test.iloc[idx].copy()
            ses = int(orig['SES'])
            
            orig_prob = clf.predict_proba(pd.DataFrame([orig]))[0, 1]
            pre_list.append(orig_prob)
            
            best_cf = None
            best_score = float('inf')
            
            if 'Manifold nearest-neighbor' in method:
                # Manifold nearest-neighbor recourse baseline evaluating passing training instances satisfying directional bounds
                valid_neighbors = train_passing[
                    (train_passing['studytime'] >= orig['studytime']) &
                    (train_passing['absences'] <= orig['absences'])
                ].copy()
                
                if len(valid_neighbors) > 0:
                    valid_neighbors['age'] = orig['age']
                    valid_neighbors['failures'] = orig['failures']
                    valid_neighbors['SES'] = orig['SES']
                    
                    # Compute distances to passing training instances
                    dists = [compute_standard_eval_cost(orig, valid_neighbors.iloc[n_i]) for n_i in range(len(valid_neighbors))]
                    valid_neighbors['dist'] = dists
                    
                    probs = clf.predict_proba(valid_neighbors[features])[:, 1]
                    valid_mask = (probs >= 0.5)
                    
                    if np.any(valid_mask):
                        passing_neighbors = valid_neighbors[valid_mask]
                        for n_idx in range(len(passing_neighbors)):
                            neighbor = passing_neighbors.iloc[n_idx]
                            c = compute_standard_eval_cost(orig, neighbor)
                            if c < best_score:
                                best_score = c
                                best_cf = neighbor
            else:
                # Vectorized candidate grid evaluation (K = 700 candidates)
                cands = np.tile(orig[features].values, (len(deltas_arr), 1))
                cands[:, 1] = np.clip(cands[:, 1] + deltas_arr[:, 0], 1, 4) # studytime
                cands[:, 3] = np.clip(cands[:, 3] + deltas_arr[:, 1], 0, 93) # absences
                cands[:, 4] = np.clip(cands[:, 4] + deltas_arr[:, 2], 1, 5) # freetime
                cands[:, 5] = np.clip(cands[:, 5] + deltas_arr[:, 3], 1, 5) # goout
                
                df_cands = pd.DataFrame(cands, columns=features).drop_duplicates()
                probs = clf.predict_proba(df_cands)[:, 1]
                valid_mask = (probs >= 0.5)
                
                if np.any(valid_mask):
                    valid_df = df_cands[valid_mask]
                    
                    # Evaluation and scoring
                    for v_i in range(len(valid_df)):
                        cand = valid_df.iloc[v_i]
                        score = compute_scoring_cost(orig, cand, ses, method)
                        if score < best_score:
                            best_score = score
                            best_cf = cand
                            
            if best_cf is not None:
                final_prob = clf.predict_proba(pd.DataFrame([best_cf[features]]))[0, 1]
                post_list.append(final_prob)
                
                eval_cost = compute_standard_eval_cost(orig, best_cf)
                
                if ses == 0:
                    low_costs.append(eval_cost)
                    valid_low += 1
                else:
                    high_costs.append(eval_cost)
                    valid_high += 1
                    
                if seed == 42 and 'Proposed' in method:
                    feature_shifts_log.append({
                        'Method': method,
                        'SES': 'Low SES' if ses == 0 else 'High SES',
                        'd_studytime': best_cf['studytime'] - orig['studytime'],
                        'd_absences': best_cf['absences'] - orig['absences'],
                        'd_freetime': best_cf['freetime'] - orig['freetime'],
                        'd_goout': best_cf['goout'] - orig['goout']
                    })
            else:
                post_list.append(orig_prob)
                
        mean_low = np.mean(low_costs) if len(low_costs) > 0 else 0.0
        mean_high = np.mean(high_costs) if len(high_costs) > 0 else 0.0
        rfd_seed = abs(mean_low - mean_high) # per-seed RFD
        total_failed_count = len(failed_test)
        total_valid_count = valid_low + valid_high
        validity = total_valid_count / max(1, total_failed_count)
        
        seed_results[method].append({
            'seed': seed,
            'mean_low': mean_low,
            'mean_high': mean_high,
            'rfd': rfd_seed,
            'validity': validity,
            'valid_count': total_valid_count,
            'failed_count': total_failed_count,
            'valid_low': valid_low,
            'valid_high': valid_high
        })
        
        pre_post_probs[method]['pre'].extend(pre_list)
        pre_post_probs[method]['post'].extend(post_list)

# Aggregate Multi-Seed Ablation Results
ablation_summary = []
base_low_mean = np.mean([r['mean_low'] for r in seed_results['Unconstrained discrete search (DiCE-style)']])

for method in ablation_methods:
    res = seed_results[method]
    lows = [r['mean_low'] for r in res]
    highs = [r['mean_high'] for r in res]
    rfds = [r['rfd'] for r in res] # Seed-level absolute differences
    vals = [r['validity'] for r in res]
    val_counts = [r['valid_count'] for r in res]
    fail_counts = [r['failed_count'] for r in res]
    
    if 'Unconstrained' in method:
        rbr_l_str = "Reference"
    else:
        rbr_l_list = [(base_low_mean - l) / max(1e-5, base_low_mean) * 100 for l in lows]
        rbr_l_str = f"{np.mean(rbr_l_list):.1f}% ± {np.std(rbr_l_list):.1f}%"
    
    macro_rfd = abs(np.mean(lows) - np.mean(highs))
    mean_val_count = np.mean(val_counts)
    mean_fail_count = np.mean(fail_counts)
    
    ablation_summary.append({
        'Methodology': method,
        'Low SES Cost (mu_L)': f"{np.mean(lows):.4f} ± {np.std(lows):.4f}",
        'High SES Cost (mu_H)': f"{np.mean(highs):.4f} ± {np.std(highs):.4f}",
        'RFD (E[|mu_L,s - mu_H,s|])': f"{np.mean(rfds):.4f} ± {np.std(rfds):.4f}",
        'Macro |mu_L - mu_H|': f"{macro_rfd:.4f}",
        'RBR_L (%)': rbr_l_str,
        'Validity (%)': f"{np.mean(vals):.1%} ± {np.std(vals):.1%}",
        'Valid Student Count': f"{mean_val_count:.1f} / {mean_fail_count:.1f}",
        'raw_low': np.mean(lows),
        'raw_high': np.mean(highs),
        'raw_rfd': np.mean(rfds),
        'raw_val': np.mean(vals),
        'lows_list': lows,
        'highs_list': highs
    })

ablation_df = pd.DataFrame(ablation_summary)
ablation_export_df = ablation_df.drop(columns=['lows_list', 'highs_list'])
ablation_export_df.to_csv(os.path.join(TABLES_DIR, "ablation_aggregate.csv"), index=False)

# Export per-seed detailed recourse results table to heldout_by_seed.csv
heldout_records = []
for method in ablation_methods:
    for r in seed_results[method]:
        heldout_records.append({
            'Seed': r['seed'],
            'Methodology': method,
            'Low_SES_Cost_mu_L': r['mean_low'],
            'High_SES_Cost_mu_H': r['mean_high'],
            'Per_Seed_RFD': r['rfd'],
            'Validity_Rate': r['validity'],
            'Valid_Student_Count': r['valid_count'],
            'Failed_Student_Count': r['failed_count'],
            'Valid_Low_SES': r['valid_low'],
            'Valid_High_SES': r['valid_high']
        })
heldout_df = pd.DataFrame(heldout_records)
heldout_df.to_csv(os.path.join(TABLES_DIR, "heldout_by_seed.csv"), index=False)

print("\n--- Multi-Seed Ablation Results (10 Seeds, mean ± SD) ---")
print(ablation_export_df[['Methodology', 'Low SES Cost (mu_L)', 'High SES Cost (mu_H)', 'RFD (E[|mu_L,s - mu_H,s|])', 'RBR_L (%)', 'Validity (%)', 'Valid Student Count']].to_string(index=False))

# 5. STATISTICAL SIGNIFICANCE TESTS ACROSS SEEDS
print("\n--- Running Statistical Significance Tests Across 10 Seeds ---")
stat_tests = []

unconstrained_lows = ablation_summary[0]['lows_list']
actionable_lows = ablation_summary[1]['lows_list']
proposed_lows = ablation_summary[3]['lows_list']
knn_lows = ablation_summary[4]['lows_list']

# Pair 1: Unconstrained vs Actionable (Ustun-aligned)
t_stat, p_val_t = stats.ttest_rel(unconstrained_lows, actionable_lows)
w_stat, p_val_w = stats.wilcoxon(unconstrained_lows, actionable_lows)
stat_tests.append({
    'Comparison': 'Unconstrained vs Actionable (Ustun-aligned)',
    'Metric': 'Low SES Cost (mu_L)',
    'Mean Diff (B - A)': np.mean(actionable_lows) - np.mean(unconstrained_lows),
    't-statistic': t_stat,
    'p-value (t-test)': p_val_t,
    'Wilcoxon W': w_stat,
    'p-value (Wilcoxon)': p_val_w,
    'Significant (p < 0.05)': p_val_t < 0.05
})

# Pair 2: Actionable vs Proposed Difficulty-Weighted
t_stat, p_val_t = stats.ttest_rel(actionable_lows, proposed_lows)
w_stat, p_val_w = stats.wilcoxon(actionable_lows, proposed_lows)
stat_tests.append({
    'Comparison': 'Actionable vs Proposed Difficulty-Weighted',
    'Metric': 'Low SES Cost (mu_L)',
    'Mean Diff (B - A)': np.mean(proposed_lows) - np.mean(actionable_lows),
    't-statistic': t_stat,
    'p-value (t-test)': p_val_t,
    'Wilcoxon W': w_stat,
    'p-value (Wilcoxon)': p_val_w,
    'Significant (p < 0.05)': p_val_t < 0.05
})

# Pair 3: Actionable vs Manifold kNN Baseline
t_stat, p_val_t = stats.ttest_rel(actionable_lows, knn_lows)
w_stat, p_val_w = stats.wilcoxon(actionable_lows, knn_lows)
stat_tests.append({
    'Comparison': 'Actionable vs Manifold kNN Baseline',
    'Metric': 'Low SES Cost (mu_L)',
    'Mean Diff (B - A)': np.mean(knn_lows) - np.mean(actionable_lows),
    't-statistic': t_stat,
    'p-value (t-test)': p_val_t,
    'Wilcoxon W': w_stat,
    'p-value (Wilcoxon)': p_val_w,
    'Significant (p < 0.05)': p_val_t < 0.05
})

stat_df = pd.DataFrame(stat_tests)
stat_df.to_csv(os.path.join(TABLES_DIR, "statistical_tests.csv"), index=False)
print(stat_df[['Comparison', 'Metric', 'Mean Diff (B - A)', 'p-value (t-test)', 'p-value (Wilcoxon)', 'Significant (p < 0.05)']].to_string(index=False))

# Population Flow Table
pop_df = pd.DataFrame(pop_flow_records)
pop_summary = pd.DataFrame([{
    'Total Population (N)': len(df),
    'Pass Prevalence': f"{sum(df['passed'] == 1)} ({sum(df['passed'] == 1)/len(df):.1%})",
    'Fail Prevalence': f"{sum(df['passed'] == 0)} ({sum(df['passed'] == 0)/len(df):.1%})",
    'Mean Test Set N': np.mean(pop_df['N_test']),
    'Mean Predicted Fail N': np.mean(pop_df['N_failed']),
    'Mean Low SES Fail N': np.mean(pop_df['N_failed_low']),
    'Mean High SES Fail N': np.mean(pop_df['N_failed_high'])
}])
pop_summary.to_csv(os.path.join(TABLES_DIR, "population_flow.csv"), index=False)
print("\n--- Population Flow Summary ---")
print(pop_summary.to_string(index=False))

# 6. SES SENSITIVITY ANALYSIS TABLE (Evaluated across 10 random seeds under equal pipeline)
print("\n--- Running SES Sensitivity Analysis Across 10 Seeds ---")
sens_records = []

for ses_def_name in ['Primary Composite Index', 'Medu > 2', 'Medu + Fedu > 4']:
    if ses_def_name == 'Primary Composite Index':
        df['SES_temp'] = df['SES']
    elif ses_def_name == 'Medu > 2':
        df['SES_temp'] = (df['Medu'] > 2).astype(int)
    else:
        df['SES_temp'] = ((df['Medu'] + df['Fedu']) > 4).astype(int)
        
    X_sens = df[['age', 'studytime', 'failures', 'absences', 'freetime', 'goout', 'SES_temp']].rename(columns={'SES_temp': 'SES'})
    y_sens = df['passed']
    
    seed_lows, seed_highs, seed_rfds = [], [], []
    for seed in seeds:
        X_tr, X_te, y_tr, y_te = train_test_split(X_sens, y_sens, test_size=0.2, random_state=seed, stratify=y_sens)
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed)
        clf.fit(X_tr, y_tr)
        
        failed_test = X_te[clf.predict(X_te) == 0]
        low_c, high_c = [], []
        for idx in range(len(failed_test)):
            orig = failed_test.iloc[idx].copy()
            ses = int(orig['SES'])
            
            cands = np.tile(orig[features].values, (len(constrained_deltas), 1))
            cands[:, 1] = np.clip(cands[:, 1] + np.array(constrained_deltas)[:, 0], 1, 4)
            cands[:, 3] = np.clip(cands[:, 3] + np.array(constrained_deltas)[:, 1], 0, 93)
            cands[:, 4] = np.clip(cands[:, 4] + np.array(constrained_deltas)[:, 2], 1, 5)
            cands[:, 5] = np.clip(cands[:, 5] + np.array(constrained_deltas)[:, 3], 1, 5)
            
            df_cands = pd.DataFrame(cands, columns=features).drop_duplicates()
            probs = clf.predict_proba(df_cands)[:, 1]
            valid_idx = np.where(probs >= 0.5)[0]
            
            if len(valid_idx) > 0:
                costs = [compute_standard_eval_cost(orig, df_cands.iloc[v_i]) for v_i in valid_idx]
                cost = min(costs)
                if ses == 0:
                    low_c.append(cost)
                else:
                    high_c.append(cost)
                    
        m_low = np.mean(low_c) if len(low_c) > 0 else 0.0
        m_high = np.mean(high_c) if len(high_c) > 0 else 0.0
        seed_lows.append(m_low)
        seed_highs.append(m_high)
        seed_rfds.append(abs(m_low - m_high))
        
    sens_records.append({
        'SES Definition': ses_def_name,
        'Low SES Count': sum(X_sens['SES'] == 0),
        'High SES Count': sum(X_sens['SES'] == 1),
        'Low SES Cost': f"{np.mean(seed_lows):.4f} ± {np.std(seed_lows):.4f}",
        'High SES Cost': f"{np.mean(seed_highs):.4f} ± {np.std(seed_highs):.4f}",
        'RFD': f"{np.mean(seed_rfds):.4f} ± {np.std(seed_rfds):.4f}"
    })

sens_df = pd.DataFrame(sens_records)
sens_df.to_csv(os.path.join(TABLES_DIR, "ses_sensitivity.csv"), index=False)

# 7. GENERATE HIGH-RESOLUTION FIGURES
sns.set_theme(style="whitegrid", font_scale=1.1)

# Figure 1: Optimization Results Bar Plot
fig, ax = plt.subplots(figsize=(10, 6))
methods_short = ['Unconstrained\n(DiCE-style)', 'Actionable\n(Ustun-aligned)', 'Fairness\nWeighted', 'Proposed\nDifficulty-Wtd', 'Manifold\nkNN']
low_vals = [a['raw_low'] for a in ablation_summary]
high_vals = [a['raw_high'] for a in ablation_summary]

x = np.arange(len(methods_short))
width = 0.35

rects1 = ax.bar(x - width/2, low_vals, width, label='Low SES Distance (mu_L)', color='#E63946')
rects2 = ax.bar(x + width/2, high_vals, width, label='High SES Distance (mu_H)', color='#457B9D')

ax.set_ylabel('Mean Normalized Recourse Distance')
ax.set_title('Group-Wise Recourse Feature Distance Across Optimization Methods (UCI Student Dataset)')
ax.set_xticks(x)
ax.set_xticklabels(methods_short)
ax.legend()

for rect in rects1 + rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig1_recourse_cost_comparison.png"), dpi=300)
plt.close()

# Figure 2: Pre vs Post Passing Probability
fig, ax = plt.subplots(figsize=(8, 5))
prop_pre = pre_post_probs['Proposed SES-Sensitive Difficulty-Weighted Recourse']['pre']
prop_post = pre_post_probs['Proposed SES-Sensitive Difficulty-Weighted Recourse']['post']

mean_pre = np.mean(prop_pre)
mean_post = np.mean(prop_post)

sns.kdeplot(prop_pre, ax=ax, color='#E63946', fill=True, label=f'Pre-Recourse (Mean = {mean_pre:.2f})', alpha=0.4)
sns.kdeplot(prop_post, ax=ax, color='#2A9D8F', fill=True, label=f'Post-Recourse (Mean = {mean_post:.2f})', alpha=0.4)

ax.axvline(0.5, color='gray', linestyle='--', label='Decision Boundary (0.50)')
ax.set_xlabel('Model-Predicted Passing Probability')
ax.set_ylabel('Density')
ax.set_title('Model Prediction Reversal: Pre vs Post Recourse Probability')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig2_pre_post_probability.png"), dpi=300)
plt.close()

# Figure 3: Feature Shifts (Original UCI Scales) for Proposed Actionable Recourse
if len(feature_shifts_log) > 0:
    shifts_df = pd.DataFrame(feature_shifts_log)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    
    sns.countplot(data=shifts_df, x='d_studytime', hue='SES', ax=axes[0, 0], palette='Set1')
    axes[0, 0].set_title('Study Time Shift (Non-negative: delta_study >= 0)')
    axes[0, 0].set_xlabel('Delta Study Time')
    
    sns.countplot(data=shifts_df, x='d_absences', hue='SES', ax=axes[0, 1], palette='Set1')
    axes[0, 1].set_title('Absences Shift (Non-positive: delta_abs <= 0)')
    axes[0, 1].set_xlabel('Delta Absences')
    
    sns.countplot(data=shifts_df, x='d_freetime', hue='SES', ax=axes[1, 0], palette='Set1')
    axes[1, 0].set_title('Free Time Shift (1-5 Ordinal Scale)')
    axes[1, 0].set_xlabel('Delta Free Time')
    
    sns.countplot(data=shifts_df, x='d_goout', hue='SES', ax=axes[1, 1], palette='Set1')
    axes[1, 1].set_title('Go Out Shift (1-5 Ordinal Scale)')
    axes[1, 1].set_xlabel('Delta Go Out')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig3_feature_shifts.png"), dpi=300)
    plt.close()

# 8. GENERATE RESULTS_FOR_PAPER.MD
markdown_results = f"""# Official Empirical Execution Results (DAG-Informed Recourse Analytics)

**Dataset**: UCI Student Performance (`student-por.csv`, N = {len(df)})  
**Primary SES Definition**: Composite Index (`(Medu + Fedu + famsize_small + internet_yes) >= median`)  
**Predictive Model**: Gradient Boosting (5-Fold Stratified CV, no G1/G2)  
**Multi-Seed Recourse Evaluation**: 10 Random Seeds (42..51)

---

## 1. Predictive Performance (5-Fold Stratified CV)

- **Class Prevalence**: Passed (G3 >= 10): {sum(df['passed'] == 1)} ({sum(df['passed'] == 1)/len(df):.1%}), Failed (G3 < 10): {sum(df['passed'] == 0)} ({sum(df['passed'] == 0)/len(df):.1%})
- **Accuracy**: {cv_results_df.loc[cv_results_df['Metric']=='Accuracy', 'Mean'].values[0]:.4f} ± {cv_results_df.loc[cv_results_df['Metric']=='Accuracy', 'SD'].values[0]:.4f}
- **Precision**: {cv_results_df.loc[cv_results_df['Metric']=='Precision', 'Mean'].values[0]:.4f} ± {cv_results_df.loc[cv_results_df['Metric']=='Precision', 'SD'].values[0]:.4f}
- **Recall**: {cv_results_df.loc[cv_results_df['Metric']=='Recall', 'Mean'].values[0]:.4f} ± {cv_results_df.loc[cv_results_df['Metric']=='Recall', 'SD'].values[0]:.4f}
- **F1-Score**: {cv_results_df.loc[cv_results_df['Metric']=='F1-Score', 'Mean'].values[0]:.4f} ± {cv_results_df.loc[cv_results_df['Metric']=='F1-Score', 'SD'].values[0]:.4f}
- **ROC AUC**: {cv_results_df.loc[cv_results_df['Metric']=='ROC AUC', 'Mean'].values[0]:.4f} ± {cv_results_df.loc[cv_results_df['Metric']=='ROC AUC', 'SD'].values[0]:.4f}

---

## 2. Multi-Seed Recourse & Ablation Results (10 Seeds, mean ± SD)

| Methodology | Low SES Cost (mu_L) | High SES Cost (mu_H) | RFD (E[|mu_L,s - mu_H,s|]) | RBR_L (%) | Validity (%) | Valid Students |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""

for a in ablation_summary:
    markdown_results += f"| {a['Methodology']} | {a['Low SES Cost (mu_L)']} | {a['High SES Cost (mu_H)']} | {a['RFD (E[|mu_L,s - mu_H,s|])']} | {a['RBR_L (%)']} | {a['Validity (%)']} | {a['Valid Student Count']} |\n"

markdown_results += f"""
---

## 3. Statistical Significance Tests Across 10 Seeds

| Comparison | Metric | Mean Diff | p-value (t-test) | p-value (Wilcoxon) | Significant (p < 0.05) |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""

for s in stat_tests:
    markdown_results += f"| {s['Comparison']} | {s['Metric']} | {s['Mean Diff (B - A)']:.4f} | {s['p-value (t-test)']:.4e} | {s['p-value (Wilcoxon)']:.4e} | {s['Significant (p < 0.05)']} |\n"

markdown_results += f"""
---

## 4. Model Prediction Reversal Probability

- **Pre-Recourse Mean Passing Probability (E[f(x)] for failed students)**: {mean_pre:.4f}
- **Post-Recourse Mean Passing Probability (E[f(x+delta)] after valid recourse)**: {mean_post:.4f}

---

## 5. SES Sensitivity Analysis (10-Seed Average)

| SES Definition | Low SES Count | High SES Count | Low SES Cost | High SES Cost | RFD |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""

for s in sens_records:
    markdown_results += f"| {s['SES Definition']} | {s['Low SES Count']} | {s['High SES Count']} | {s['Low SES Cost']} | {s['High SES Cost']} | {s['RFD']} |\n"

with open(os.path.join(BASE_DIR, "RESULTS_FOR_PAPER.md"), "w") as f:
    f.write(markdown_results)

print("\n==========================================================")
print(" PIPELINE COMPLETED SUCCESSFULLY! Outputs written to:")
print("   - outputs/tables/")
print("   - outputs/figures/")
print("   - RESULTS_FOR_PAPER.md")
print("==========================================================")
