# Official Empirical Execution Results (DAG-Informed Actionable Recourse)

**Dataset**: UCI Student Performance (`student-por.csv`, N = 649)  
**Primary SES Definition**: Composite Index (`(Medu + Fedu + famsize_small + internet_yes) >= median`)  
**Predictive Model**: Gradient Boosting (5-Fold Stratified CV, no G1/G2)  
**Multi-Seed Recourse Evaluation**: 10 Random Seeds (42..51)

---

## 1. Predictive Performance (5-Fold Stratified CV)

- **Class Prevalence**: Passed (G3 >= 10): 549 (84.6%), Failed (G3 < 10): 100 (15.4%)
- **Accuracy**: 0.8335 ± 0.0156
- **Precision**: 0.8747 ± 0.0158
- **Recall**: 0.9381 ± 0.0106
- **F1-Score**: 0.9051 ± 0.0083
- **ROC AUC**: 0.7263 ± 0.0556

---

## 2. Multi-Seed Recourse & Ablation Results (10 Seeds, mean ± SD)

| Methodology | Low SES Cost (mu_L) | High SES Cost (mu_H) | RFD (E[|mu_L,s - mu_H,s|]) | RBR_L (%) | Validity (%) | Valid Students |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Unconstrained discrete search (DiCE-style) | 0.2248 ± 0.0734 | 0.1496 ± 0.0726 | 0.1131 ± 0.0635 | Reference | 98.5% ± 3.0% | 11.9 / 12.1 |
| Actionable Recourse (Ustun-style discrete grid) | 0.3381 ± 0.1457 | 0.2638 ± 0.1633 | 0.1585 ± 0.1038 | -50.4% ± 64.8% | 95.5% ± 4.7% | 11.5 / 12.1 |
| Fairness-weighted discrete search | 0.3381 ± 0.1457 | 0.2638 ± 0.1633 | 0.1585 ± 0.1038 | -50.4% ± 64.8% | 95.5% ± 4.7% | 11.5 / 12.1 |
| Proposed Fair DAG-Informed Recourse | 0.3408 ± 0.1441 | 0.2638 ± 0.1633 | 0.1613 ± 0.1038 | -51.6% ± 64.1% | 95.5% ± 4.7% | 11.5 / 12.1 |
| FACE-like kNN graph recourse (Equal budget K) | 0.3795 ± 0.1462 | 0.3078 ± 0.1740 | 0.1736 ± 0.0967 | -68.8% ± 65.0% | 98.6% ± 4.3% | 11.9 / 12.1 |

---

## 3. Statistical Significance Tests Across 10 Seeds

| Comparison | Metric | Mean Diff | p-value (t-test) | p-value (Wilcoxon) | Significant (p < 0.05) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Unconstrained vs Actionable (Ustun) | Low SES Cost (mu_L) | 0.1133 | 4.6610e-03 | 3.9062e-03 | True |
| Actionable vs Proposed Fair DAG | Low SES Cost (mu_L) | 0.0027 | 1.7078e-01 | 5.0000e-01 | False |
| Actionable vs FACE-like kNN | Low SES Cost (mu_L) | 0.0413 | 9.8024e-02 | 9.7656e-02 | False |

---

## 4. Model Prediction Reversal Probability

- **Pre-Recourse Mean Passing Probability (E[f(x)] for failed students)**: 0.2866
- **Post-Recourse Mean Passing Probability (E[f(x+delta)] after valid recourse)**: 0.5975

---

## 5. SES Sensitivity Analysis (10-Seed Average)

| SES Definition | Low SES Count | High SES Count | Low SES Cost | High SES Cost | RFD |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Primary Composite Index | 307 | 342 | 0.3381 ± 0.1457 | 0.2638 ± 0.1633 | 0.1585 ± 0.1038 |
| Medu > 2 | 335 | 314 | 0.3315 ± 0.1160 | 0.2256 ± 0.0961 | 0.1257 ± 0.1066 |
| Medu + Fedu > 4 | 321 | 328 | 0.3798 ± 0.1710 | 0.2383 ± 0.1261 | 0.1883 ± 0.1958 |
