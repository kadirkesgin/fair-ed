# Fair-Ed: Causal Algorithmic Recourse for Educational Interventions

This repository contains the source code, datasets, figures, and deployment dashboard for the paper:
**"From Prediction to Intervention: Fair Causal Recourse for Student Academic Performance"**

## 📌 Overview
Predicting student failure is only the first step. This project introduces a **Causal Machine Learning** framework that not only identifies at-risk students but also provides **actionable, realistic, and equitable** intervention pathways (recourse) to help them succeed. It uses a custom **Socioeconomic Status (SES)** weighted cost constraint layered over a **Structural Causal Model (DAG)** to ensure the recommended changes (e.g., modifying study time or absences) do not unfairly burden disadvantaged groups.

## 🗂️ Directory Structure
*   `data/`: Contains the UCI Student Performance dataset (`student-mat.csv`, `student-por.csv`).
*   `src/`: Core Python pipeline scripts.
    *   `comprehensive_experiments.py`: Generates the predictive metrics (XGBoost) and the fair causal optimization.
    *   `draw_dag.py`: Draws the NetworkX graph.
    *   `sci_figures_generator.py`: Generates all statistical Q1-quality plots.
*   `figures/`: Generated SCI visuals (Violin plots, Actionable feature shifts, DAG).
*   `dashboard/`: The `Fair-Ed` interactive pedagogical tool for school counselors (HTML/JS/CSS).
*   `paper/`: Final LaTeX manuscript and bibliography.

## 🚀 Quickstart

1.  **Install requirements**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Main Optimization Pipeline**:
    ```bash
    cd src
    python comprehensive_experiments.py
    ```
    This will calculate the **Recourse Fairness Difference (RFD)**, generate the baseline interventions, and compare them against the proposed Fair Recourse approach.

3.  **Generate Figures**:
    ```bash
    python sci_figures_generator.py
    ```

4.  **Launch the Teacher Dashboard**:
    Open `dashboard/index.html` in any modern web browser to interact with the system via the Web UI.

## ⚖️ Key Metrics
*   **Predictive Validation**: The XGBoost engine passes 5-Fold CV with `Accuracy: 0.87` and `AUC: 0.92`.
*   **Actionable Validity**: Maintains a `96.4%` validity rate in crossing the passing boundary.
*   **Equity Gain**: Minimizes the Recourse Fairness Difference (RFD) from `4.22` (Standard DiCE) to `0.85` (Proposed System).

## 📄 Citation
(Citation information to be updated upon publication)
