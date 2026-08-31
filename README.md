# Fair-Ed: Fair Causal Recourse for Student Academic Performance Prediction

This repository contains the official source code, datasets, figures, and evaluation pipeline for the paper:
**"Fair causal recourse for student academic performance prediction"**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22199775.svg)](https://doi.org/10.5281/zenodo.22199775)

## 📌 Overview
Predicting student failure is only the first step. This project introduces a **Fair Causal Recourse** framework that not only identifies at-risk students but also provides **actionable, realistic, and equitable** recourse pathways to support academic success. It incorporates a **Composite Socioeconomic Status (SES)** index layered over a **DAG-constrained / actionability-constrained** candidate search to ensure that recommended changes (e.g., modifying study time or absences) respect directional educational constraints and avoid unfair friction for disadvantaged groups.

## 🗂️ Directory Structure
*   `student-por.csv`: UCI Student Performance dataset ($N=649$).
*   `src/run_experiments.py`: Single official execution pipeline script for model training (5-fold CV), recourse optimization across 10 random seeds (seeds 42..51), ablation studies, table exports, and figure generation.
*   `outputs/tables/`: Generated aggregate statistics (`ablation_aggregate.csv`, `predictive_performance.csv`, `population_flow.csv`, `benchmark_comparison.csv`, `ses_sensitivity.csv`).
*   `outputs/figures/`: High-resolution visual plots (Recourse cost comparison, pre/post probabilities, feature shift distributions).
*   `main.tex`: LaTeX manuscript for publication.
*   `references.bib`: Complete bibliography database.
*   `response_to_reviewers.md`: Point-by-point response to Editor and Reviewers.
*   `legacy/`: Archived scratch scripts from initial exploratory phases.

## 🚀 Quickstart & Reproducibility

1. **Install requirements**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. **Run the Official Pipeline**:
   ```bash
   python src/run_experiments.py
   ```
   This will execute the complete multi-seed analysis, compute 5-fold predictive metrics, evaluate all recourse ablation methods, generate all high-resolution figures in `outputs/figures/`, and export all table CSVs to `outputs/tables/` alongside `RESULTS_FOR_PAPER.md`.

## 📊 Key Summary Metrics (10 Random Seeds, mean ± SD)
*   **Predictive Model (5-Fold CV without G1/G2)**: Accuracy: $0.8335 \pm 0.0156$, ROC AUC: $0.7263 \pm 0.0556$, Class Prevalence: $84.6\%$ pass, $15.4\%$ fail.
*   **Recourse Validity**: Maintains $95.5\% \pm 4.7\%$ validity in reversing model predictions.
*   **Unconstrained Search Distance ($\mu_L$)**: $0.2248 \pm 0.0734$.
*   **Actionability-Constrained Distance ($\mu_L$)**: $0.3381 \pm 0.1457$.
*   **Fairness-Weighted Distance ($\mu_L$)**: $0.3381 \pm 0.1457$.
*   **Proposed Fair Causal Recourse Distance ($\mu_L$)**: $0.3408 \pm 0.1441$.
*   **High SES Recourse Distance ($\mu_H$)**: $0.2638 \pm 0.1633$.
*   **Recourse Fairness Difference ($\text{RFD}$)**: $0.1613 \pm 0.1038$.
*   **Prediction Reversal Shift**: Mean predicted passing probability increases from $0.2866$ before recourse to $0.5975$ after recourse.

## 📄 Code Availability & Zenodo Release
The repository is maintained on GitHub at [https://github.com/kadirkesgin/fair-ed](https://github.com/kadirkesgin/fair-ed) and permanently archived on Zenodo at [https://doi.org/10.5281/zenodo.22199775](https://doi.org/10.5281/zenodo.22199775).
