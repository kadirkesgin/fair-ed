# Fair-Ed: DAG-Informed Actionable Recourse for Student Academic Performance Prediction

This repository contains the official source code, datasets, figures, and evaluation pipeline for the paper:
**"DAG-informed actionable recourse for student academic performance prediction"**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22199775.svg)](https://doi.org/10.5281/zenodo.22199775)

## 📌 Overview
Predicting student failure is only the first step. This project introduces a **DAG-Informed Actionable Recourse** framework that moves beyond prediction-oriented early warning toward actionability-constrained educational decision support. It incorporates a **Composite Socioeconomic Status (SES)** index layered over a **DAG-constrained / actionability-constrained** candidate search to ensure that recommended changes (e.g., modifying study time or absences) respect directional educational constraints ($\delta_{\text{study}} \ge 0, \delta_{\text{abs}} \le 0$).

## 🗂️ Directory Structure
*   `student-por.csv`: UCI Student Performance dataset ($N=649$).
*   `src/run_experiments.py`: **Single official execution pipeline script** for model training (5-fold CV), recourse optimization across 10 random seeds (seeds 42..51), statistical significance testing, ablation studies, table exports, and figure generation.
*   `src/draw_dag.py`: Helper script generating Figure 1 DAG diagram (`sci_fig0_causal_dag.png`).
*   `src/legacy/`: Archived historical scratch scripts (`comprehensive_experiments.py`, `evaluate_efficacy.py`, `fix_bar_chart.py`, `sci_figures_generator.py`) preserved for archival tracking.
*   `outputs/tables/`: Generated aggregate statistics (`ablation_aggregate.csv`, `heldout_by_seed.csv`, `statistical_tests.csv`, `predictive_performance.csv`, `population_flow.csv`, `ses_sensitivity.csv`).
*   `outputs/figures/`: High-resolution visual plots (Recourse cost comparison, pre/post probabilities, feature shift distributions).
*   `paper/main.tex`: LaTeX manuscript for publication.
*   `paper/references.bib`: Complete bibliography database.
*   `response_to_reviewers.md`: Point-by-point response to Editor and Reviewers.
*   `requirements.txt`: Pinned Python dependencies (`scikit-learn==1.8.0`, `pandas==2.3.3`, `numpy==1.26.4`, `scipy==1.17.0`, `xgboost==3.2.0`).

## 🚀 Quickstart & Reproducibility

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Official Pipeline**:
   ```bash
   python src/run_experiments.py
   ```
   This will execute the complete multi-seed analysis, compute 5-fold predictive metrics, evaluate all recourse ablation methods, perform paired $t$-tests and Wilcoxon signed-rank tests across 10 seeds, generate all high-resolution figures in `outputs/figures/`, and export all table CSVs to `outputs/tables/` alongside `RESULTS_FOR_PAPER.md`.

## 📊 Key Summary Metrics (10 Random Seeds, mean ± SD)
*   **Predictive Model (5-Fold CV without G1/G2)**: Accuracy: $0.8335 \pm 0.0156$, ROC AUC: $0.7263 \pm 0.0556$, Class Prevalence: $84.6\%$ pass, $15.4\%$ fail.
*   **Recourse Validity**: Maintains $95.5\% \pm 4.7\%$ validity ($11.5 / 12.1$ failing test students per fold) in reversing model predictions.
*   **Unconstrained Search Distance ($\mu_L$)**: $0.2248 \pm 0.0734$.
*   **Actionability-Constrained Distance ($\mu_L$)**: $0.3381 \pm 0.1457$.
*   **Fairness-Weighted Distance ($\mu_L$)**: $0.3381 \pm 0.1457$.
*   **Proposed Difficulty-Weighted Recourse Distance ($\mu_L$)**: $0.3408 \pm 0.1441$.
*   **High SES Recourse Distance ($\mu_H$)**: $0.2638 \pm 0.1633$.
*   **Seed-Indexed Recourse Difference ($\text{RFD}$)**: $0.1613 \pm 0.1038$ (macro difference $|\bar{\mu}_L - \bar{\mu}_H| = 0.0771$).
*   **Prediction Reversal Shift**: Mean predicted passing probability increases from $0.2866$ before recourse to $0.5975$ after recourse.

## 📄 Code Availability & Zenodo Release
The repository is maintained on GitHub at [https://github.com/kadirkesgin/fair-ed](https://github.com/kadirkesgin/fair-ed) (v2.0.0-major-revision release) and permanently archived on Zenodo at [https://doi.org/10.5281/zenodo.22199775](https://doi.org/10.5281/zenodo.22199775).
