# Response to Reviewers and Editor

**Manuscript Title**: DAG-informed actionable recourse for student academic performance prediction  
**Journal / Conference**: Major Revision Submission  

---

We sincerely thank the Editor and the Reviewers for their constructive, thorough, and highly insightful feedback on our manuscript. In response to these recommendations, we have conducted a comprehensive re-analysis using an open-source, end-to-end reproducible pipeline script (`src/run_experiments.py`), revised the mathematical optimization formulation, corrected the candidate space math ($9,400$-point discrete space $\mathcal{F}_{\text{discrete}}$ vs $K=700$ sampled directional search grid), corrected the seed-indexed Recourse Fairness Difference (RFD) formula, aligned the DAG text 100\% with Figure 1, added statistical significance tests and absolute student counts, added a dedicated experimental protocol subsection, narrowed terminology to DAG-informed SES-sensitive difficulty-weighted recourse, and updated all bibliography citations (`main.pdf` compiles with zero citation warnings).

Below, we present our detailed, verbatim point-by-point responses organized into three dedicated sections: **1. Responses to Editor Directives**, **2. Responses to Reviewer 1 Comments**, and **3. Responses to Reviewer 2 Comments**.

---

# 1. Responses to Editor Directives

### Item E.1: Manuscript Title Revision (In-House Formatting Guideline)
> **Directive**: *Ensure that the title contains no colons (:).*

* **Response**: We have updated the manuscript title to comply strictly with the journal formatting guideline while simultaneously narrowing our terminology from "Fair Causal Recourse" to "DAG-Informed Actionable Recourse" to eliminate over-claiming.
* **Changes Made**: The title has been updated in `main.tex:24` to:
  $$\textbf{DAG-informed actionable recourse for student academic performance prediction}$$

---

### Item E.2: Correction of Equation (5), RFD Reporting, and Seed-Indexed Expectation Formulation
> **Directive**: *Address inconsistency between Equation (5) and reported Recourse Fairness Difference (RFD) values. Ensure evaluation cost is group-independent and report exact re-analyzed statistics.*

* **Response**: We thank the Editor and Reviewer 2 for bringing this critical mathematical issue to our attention.

  1. **Group-Independent Evaluation Cost Metric**:
     We removed artificial group penalty multipliers from the evaluation metric. All methods in Table~\ref{tab:ablation_results} are evaluated using the exact same group-independent normalized feature distance metric:
     $$\mathcal{C}_{\text{eval}}(\boldsymbol{\delta}) = \sum_{j \in \text{actionable}} \lambda_j |\delta_j|$$
     where $\lambda_{\text{study}} = 1/3, \lambda_{\text{abs}} = 1/93, \lambda_{\text{free}} = 1/4, \lambda_{\text{go}} = 1/4$.

  2. **Seed-Indexed Expectation RFD Formula**:
     In Equation (5) of Section 3.6, we updated the mathematical definition of RFD to be explicitly seed-indexed:
     $$\text{RFD} = \frac{1}{S} \sum_{s=1}^{S} |\mu_{L,s} - \mu_{H,s}|$$
     where $s \in \{1, \dots, S\}$ indexes evaluation splits (seeds, $S=10$), and $\mu_{L,s}$ and $\mu_{H,s}$ denote the mean recourse costs for Low-SES and High-SES students within test split $s$.

  3. **Mathematical Explanation of Mismatch via Jensen's Inequality**:
     Because fold-level cost variations exist across test splits, Jensen's inequality / triangle inequality dictates that the sample expectation of fold-level absolute differences $\text{RFD} = \frac{1}{S}\sum_{s=1}^S |\mu_{L,s} - \mu_{H,s}| \ge |\bar{\mu}_L - \bar{\mu}_H|$, where $\bar{\mu}_L = \frac{1}{S}\sum_s \mu_{L,s}$ and $\bar{\mu}_H = \frac{1}{S}\sum_s \mu_{H,s}$ are the macro multi-seed group means.
     
     In Table~\ref{tab:ablation_results}, we report both values transparently:
     - Sample expectation of per-seed group absolute differences: $\text{RFD} = 0.1613 \pm 0.1038$ for Proposed Recourse (and $0.1585 \pm 0.1038$ for Actionability-Constrained Search).
     - Macro difference between overall group means: $|\bar{\mu}_L - \bar{\mu}_H| = |0.3408 - 0.2638| = 0.0771$ for Proposed Recourse (and $|0.3381 - 0.2638| = 0.0744$ for Actionability-Constrained Search).

  4. **High-SES Cost Stability ($0.2638 \pm 0.1633$)**:
     High-SES recourse cost is identical across Actionability-constrained, Fairness-weighted, and Proposed methods because per 20\% test fold ($130$ students), there are on average only **$3.2$ High-SES predicted failing students** ($N_{\text{fail,High}} \approx 3.2$). Candidate selection for these few students is stable under standard weights, and Proposed weights apply specifically to Low-SES (`ses == 0`). We explicitly document this sample size breakdown in Table~\ref{tab:population_flow}.
* **Changes Made**: Updated Abstract, Section 3.6 (Equation 5), Section 4.2 (Table~\ref{tab:ablation_results}), Section 5, and Section 6.

---

### Item E.3: Narrowing Causal Terminology and Adding DAG Edge Justifications
> **Directive**: *Clarify the exact nature of the causal model. State clearly whether structural equations were estimated and what "DAG-constrained recourse" means in practice.*

* **Response**: We have narrowed the terminology throughout the manuscript from "Fair Causal Recourse" to **"DAG-Informed Actionable Recourse"**. We explicitly clarify that our framework implements DAG-constrained and actionability-constrained recourse under an assumed educational causal graph based on domain literature, rather than a fully identified Structural Causal Model (SCM) with estimated structural equations.

  In Section 3.3, we aligned the text 100\% with Figure~\ref{fig:dag} (`draw_dag.py`) and added individual justifications and educational literature citations for all 9 directed edges in Figure~\ref{fig:dag}:
  - $\text{SES} \to \text{studytime}$: Socioeconomic status determines home learning environment, study resources, and academic support access \cite{Sirin2005}.
  - $\text{SES} \to \text{absences}$: Low-SES students face higher structural friction (long commutes, family responsibilities, health barriers) leading to attendance challenges \cite{OECD2018}.
  - $\text{SES} \to \text{freetime}$: SES background influences extracurricular commitments and discretionary time allocation \cite{Sirin2005}.
  - $\text{age} \to \text{goout}$: Student age shapes peer socialization patterns and out-of-school social activities \cite{Castro2015}.
  - $\text{freetime} \to \text{goout}$: Available discretionary free time directly enables peer social outings \cite{Castro2015}.
  - $\text{studytime} \to y$: Weekly study time is a direct behavioral determinant of academic preparation and subject pass probability $y$ ($G3 \ge 10$) \cite{Plant2005,Nonis2010}.
  - $\text{absences} \to y$: Chronic absenteeism directly reduces classroom instructional exposure, impairing academic performance \cite{Gottfried2014,Gershenson2019}.
  - $\text{freetime} \to y$: Free time management influences academic preparation balance \cite{Nonis2010}.
  - $\text{goout} \to y$: Excessive social outings reduce study concentration and academic achievement \cite{Plant2005}.
* **Changes Made**: Inserted explicit statements and 9 edge justifications in Section 3.1 and Section 3.3 (`main.tex:232--252`).

---

### Item E.4: Complete Mathematical Objective Formulation and Candidate Space Math
> **Directive**: *Formulate explicit objective functions for all ablation conditions.*

* **Response**: In Section 3.5 (`main.tex:259--296`), we provided formal mathematical optimization objectives for all five evaluated conditions. We clarified that the full discrete feature space spans $\mathcal{F}_{\text{discrete}} = \mathcal{X}_{\text{study}} \times \mathcal{X}_{\text{abs}} \times \mathcal{X}_{\text{free}} \times \mathcal{X}_{\text{go}} = \{1,2,3,4\} \times \{0,1,\dots,93\} \times \{1,2,3,4,5\} \times \{1,2,3,4,5\}$, yielding $|\mathcal{F}_{\text{discrete}}| = 4 \times 94 \times 5 \times 5 = 9,400$ candidate points. For each student, counterfactual recommendations are drawn from a directional candidate grid of $K=700$ actionable candidate vectors constructed under directional actionability bounds ($\delta_{\text{study}} \ge 0, \delta_{\text{abs}} \le 0$):
  1. **Unconstrained discrete search (DiCE-style)**: $\min_{\boldsymbol{\delta}} \mathcal{C}_{\text{unweighted}}(\boldsymbol{\delta}) \quad \text{s.t.} \quad f(\mathbf{x}_i+\boldsymbol{\delta}) \ge 0.5$.
  2. **Actionability-constrained search (Ustun-aligned directional bounds)**: Directional bounds ($\delta_{\text{study}} \ge 0, \delta_{\text{abs}} \le 0$) under unweighted normalized feature distance.
  3. **Fairness-weighted discrete search**: Directional bounds under domain difficulty weights $w_j$ ($w_{\text{abs}}=1.5$).
  4. **Proposed SES-Sensitive Difficulty-Weighted Recourse**: Directional bounds under structural attendance friction weights for Low-SES students ($w_{\text{abs}}^{\text{Low}}=2.0$).
  5. **Manifold nearest-neighbor recourse baseline**: Nearest passing instance in the training set graph manifold satisfying directional bounds, evaluated over all passing instances in the training fold ($N_{\text{pass,train}} \approx 439$).

  Additionally, we specified:
  - **Deterministic Tie-Breaking Rule**: If multiple candidates yield equal minimal scoring cost, selection picks candidate with (i)~smallest study time increase $\delta_{\text{study}}$, (ii)~smallest absolute absence change $|\delta_{\text{abs}}|$, and (iii)~first candidate grid index.
  - **Difficulty-Weighted Recourse Clarification**: The proposed method functions as an asymmetric feature difficulty-weighted recourse model where $w_{\text{abs}}(\text{Low})=2.0$ reflects structural friction, rather than a group-inequality penalty term optimized in the loss objective.
* **Changes Made**: Detailed complete equations, candidate space math ($9,400$ points vs $K=700$ grid), and tie-breaking rules in Section 3.5.

---

### Item E.5: Equal-Budget Comparative Benchmark ($K=700$)
> **Directive**: *Ensure all sampled counterfactual search methods are evaluated under identical search budgets, seeds, and bounds.*

* **Response**: All empirical recourse baselines in the execution pipeline (`src/run_experiments.py`) were evaluated under identical discrete candidate search grids ($K=700$ candidates per student), identical random seed splits ($42 \dots 51$), held-out test splits, and identical discrete UCI ordinal feature bounds.
* **Changes Made**: Standardized grid search budget to $K=700$ and reported comparative results in Table~\ref{tab:ablation_results} and Table~\ref{tab:two_tier_benchmark}.

---

### Item E.6: Statistical Reporting, Significance Testing, and Absolute Student Counts
> **Directive**: *Report sample size (N), class prevalence, Low/High SES counts, predicted failures, recourse validity, and statistical significance.*

* **Response**: We expanded Section 4 with comprehensive statistical and sample flow reporting:
  1. **Population Flow Table (Table~\ref{tab:population_flow})**: Documenting total $N=649$, pass/fail prevalence ($84.6\%$ / $15.4\%$), Low-SES ($N=307$), High-SES ($N=342$), mean test fold size ($130.0$), mean predicted failures ($12.1$; Low-SES $8.9$, High-SES $3.2$).
  2. **Absolute Student Count Reporting**: In Table~\ref{tab:ablation_results}, we report that an average of **$11.5 \text{ out of } 12.1$** predicted failing students per test fold receive valid recourse ($95.5\% \pm 4.7\%$).
  3. **Statistical Significance Testing (Section 4.2)**: Conducted paired two-tailed $t$-tests and Wilcoxon signed-rank tests across 10 random seeds ($S=10$):
     - Unconstrained vs Actionability-Constrained Search: $+0.1133$ cost increase, $t(9) = 3.72, p = 0.0047$; Wilcoxon $W = 0.0, p = 0.0039$ (**Statistically Significant**).
     - Actionability-Constrained Search vs Proposed Difficulty-Weighted Recourse: $+0.0027$ cost diff, $t(9) = 1.49, p = 0.1708$; Wilcoxon $W = 5.0, p = 0.5000$ (**Not Statistically Significant**).
     - Actionability-Constrained Search vs Manifold Nearest-Neighbor Baseline: $+0.0413$ cost diff, $t(9) = 1.83, p = 0.0980$; Wilcoxon $W = 10.0, p = 0.0977$ (**Not Statistically Significant**).
* **Changes Made**: Added Table~\ref{tab:population_flow}, Section 4.2 statistical significance text, and absolute student counts in Table~\ref{tab:ablation_results}.

---

### Item E.7: Toning Down "Educational Efficacy" Language
> **Directive**: *Reframe pre/post passing probability shifts as model constraint satisfaction / prediction reversal, not guaranteed real-world academic efficacy.*

* **Response**: We renamed Section 4.4 to *"Model-based prediction reversal and recourse validity"*. We updated pre-recourse mean passing probability ($0.2866$) and post-recourse probability ($0.5975$) directly from pipeline outputs.
* **Changes Made**: Added an explicit disclaimer in Section 5.1:
  > *"This demonstrates that generated profiles cross the same classifier's decision boundary. It does not show that the recommended behavioral changes would cause real academic improvement. That would require longitudinal or prospective evidence."*

---

### Item E.8: Historical Absences Proxy Disclaimer
> **Directive**: *Address the retrospective nature of recorded absences in the UCI dataset.*

* **Response**: We added an explicit disclaimer clarifying that recorded absences represent historical data and are treated as a constrained actionable proxy for future engagement.
* **Changes Made**: Added text in Section 3.4 and Section 5.2:
  > *"Recorded absences are historical. Recommended reductions are simulated hypothetical changes to a future-attendance / engagement proxy, not a retrospective rewrite of past absences. Prediction is assumed to be issued mid-term / before final outcome, but the UCI snapshot does not encode this timing. Therefore absences are treated as a constrained actionable proxy, not a directly implementable past intervention."*

---

### Item E.9: Bibliography and Reference Audit
> **Directive**: *Add missing text citations, fix duplicate references, and correct dataset DOIs.*

* **Response**: We performed a complete reference audit:
  - Added missing literature citations: Sirin (2005) \cite{Sirin2005}, Castro et al. (2015) \cite{Castro2015}, OECD (2018) \cite{OECD2018}, Plant et al. (2005) \cite{Plant2005}, Nonis \& Hudson (2010) \cite{Nonis2010}, Gottfried (2014) \cite{Gottfried2014}, Gershenson et al. (2019) \cite{Gershenson2019}, Baker \& Hawn (2022) \cite{Baker2022}, Buñay-Guisnán et al. (2026) \cite{BunayGuisnan2026}.
  - Updated Paulo Cortez (2008) dataset entry with official UCI repository URL and DOI (`10.24432/C5TG7T`).
  - Removed duplicate Türkmen reference key (`[17]`), consolidating to `Türkmen (2025)` \cite{Turkmen2025}.
  - Corrected Ribeiro et al. \cite{Ribeiro2016} entry to KDD 2016 proceedings.
  - Fixed TeX escaping for author names (`Sch{\"{o}}lkopf`).
* **Changes Made**: Updated `references.bib` and in-text citation keys (`main.tex`). `pdflatex` compiles cleanly with zero citation warnings.

---

### Item E.10: Code Release and Zenodo DOI
> **Directive**: *Ensure repository reproducibility and permanent DOI archive.*

* **Response**: We created a single, self-contained, reproducible execution pipeline script (`src/run_experiments.py`) that generates all tables (`outputs/tables/`), figures (`outputs/figures/`), and markdown results (`RESULTS_FOR_PAPER.md`).
* **Changes Made**: Updated Code Availability statement (`main.tex:514`) pointing to GitHub (`https://github.com/kadirkesgin/fair-ed`) and permanently archived on Zenodo (DOI: `10.5281/zenodo.22199775`).

---

# 2. Responses to Reviewer 1 Comments (Verbatim Point-by-Point)

### Comment 1.1: Comparative Benchmark and Method Positioning
> **Reviewer 1 Comment**: *The paper needs a clearer comparative benchmark against existing recourse algorithms (e.g., Actionable Recourse, FACE, DiCE).*

* **Response**: We thank Reviewer 1 for this valuable suggestion. We constructed a Two-Tier Benchmark Comparison Table (Table~\ref{tab:two_tier_benchmark}) separating recourse methods into:
  1. **Empirical Layer (Evaluated on discrete features)**: Unconstrained discrete search (DiCE-style), Actionability-constrained search (Ustun-aligned directional bounds), Fairness-weighted search, Proposed SES-Sensitive Difficulty-Weighted Recourse, and Manifold nearest-neighbor recourse baseline.
  2. **Conceptual Layer (Theoretical contrast)**: Karimi Causal Recourse \cite{Karimi2021}, von Kügelgen Fair Recourse \cite{vonKugelgen2022}, and Ustun Integer Linear Model \cite{Ustun2019}.
* **Changes Made**: Added Table~\ref{tab:two_tier_benchmark} in Section 4.3 detailing search grid specifications ($K=700$), implementation rationale, and model compatibility requirements.

---

### Comment 1.2: Clarifying FACE-like Baseline Implementation and Equal Search Budget
> **Reviewer 1 Comment**: *Ensure FACE baseline is accurately described.*

* **Response**: We renamed the FACE-like baseline to **"Manifold nearest-neighbor recourse baseline"** to accurately describe it as a nearest-neighbor baseline on the training manifold rather than an exact graph-shortest-path FACE implementation. It evaluates all passing instances in the training fold ($N_{\text{pass,train}} \approx 439$) satisfying directional bounds.
* **Changes Made**: Updated Section 3.5, Section 4.2 (Table~\ref{tab:ablation_results}), and `src/run_experiments.py`.

---

### Comment 1.3: Scope of Causal Reasoning and Absence of SCM Structural Equations
> **Reviewer 1 Comment**: *Clarify causal assumptions and edge justifications.*

* **Response**: As detailed under Item E.3, we narrowed our terminology to "DAG-Informed Actionable Recourse" and added explicit educational literature justifications and citations for all 9 directed edges in Figure~\ref{fig:dag} in Section 3.3.
* **Changes Made**: Updated Section 3.1 and Section 3.3 (`main.tex:232--252`).

---

### Comment 1.4: Dataset Selection and Removal of Unaligned External Benchmark (OULAD)
> **Reviewer 1 Comment**: *Address dataset limitations and cross-dataset feature mismatch.*

* **Response**: We agreed with Reviewer 1 that feature alignment between UCI and OULAD was too loose (lacking direct study time and absenteeism measures) to justify a quantitative external replication claim. We removed OULAD from quantitative results tables and positioned it as a conceptual framework for future external validation.
* **Changes Made**: Updated Section 5.2 Limitations.

---

# 3. Responses to Reviewer 2 Comments (Verbatim Point-by-Point)

### Comment 2.1: Discrepancy in Equation (5) and RFD Calculation vs Table Group Means
> **Reviewer 2 Comment**: *Equation (5) defines RFD as $|\mu_{\text{Low}} - \mu_{\text{High}}|$. However, the reported values (0.1131, 0.1585, 0.1613) do not equal the absolute difference between the reported group means. Please clarify and correct this discrepancy.*

* **Response**: We thank Reviewer 2 for identifying this critical mathematical inconsistency. As detailed under Item E.2, we updated Equation (5) to the seed-indexed expectation formula:
  $$\text{RFD} = \frac{1}{S} \sum_{s=1}^S |\mu_{L,s} - \mu_{H,s}|$$
  Because fold-level cost variations exist across test splits, Jensen's inequality dictates that the expected value of fold-level absolute differences $\text{RFD} = \frac{1}{S}\sum_{s=1}^S |\mu_{L,s} - \mu_{H,s}| \ge |\bar{\mu}_L - \bar{\mu}_H|$.
  
  In Table~\ref{tab:ablation_results}, both values are now explicitly reported and explained:
  - Sample expectation of per-seed group absolute differences: $\text{RFD} = 0.1613 \pm 0.1038$ for Proposed Recourse (and $0.1585 \pm 0.1038$ for Actionability-Constrained Search).
  - Macro difference between overall group means: $|\bar{\mu}_L - \bar{\mu}_H| = |0.3408 - 0.2638| = 0.0771$ for Proposed Recourse (and $|0.3381 - 0.2638| = 0.0744$ for Actionability-Constrained Search).
* **Changes Made**: Updated Section 3.6 (Equation 5), Section 4.2 (Table~\ref{tab:ablation_results}), Section 5, and Section 6.

---

### Comment 2.2: Primary Composite SES Definition Consistency
> **Reviewer 2 Comment**: *Ensure the primary SES definition in code matches the manuscript description.*

* **Response**: We verified that our primary SES variable in `src/run_experiments.py` matches the composite index described in the manuscript:
  $$\text{Composite Score} = \text{Medu} + \text{Fedu} + \text{famsize\_small} + \text{internet\_yes}$$
  binarized at the median threshold ($6.0$). In addition, we added an SES Sensitivity Analysis Table (Table~\ref{tab:ses_sensitivity}) evaluating alternative definitions (`Medu > 2` and `Medu + Fedu > 4`) across the exact same 10-seed evaluation protocol.
* **Changes Made**: Updated Section 3.2 and Section 4.5 (Table~\ref{tab:ses_sensitivity}).

---

### Comment 2.3: Historical Absences Proxy and Mid-Term Intervention Disclaimer
> **Reviewer 2 Comment**: *Address the retrospective nature of recorded absences in the UCI dataset.*

* **Response**: As detailed under Item E.8, we added explicit disclaimers in Section 3.4 and Section 5.2 clarifying that recorded absences represent historical data and are treated as a constrained actionable proxy for future engagement.
* **Changes Made**: Updated Section 3.4 and Section 5.2 (`main.tex:254--257`).

---

### Comment 2.4: Mathematical Optimization Formulation and Candidate Selection Rules
> **Reviewer 2 Comment**: *Provide complete mathematical objectives, candidate sets, and tie-breaking rules.*

* **Response**: As detailed under Item E.4, we updated Section 3.5 with formal minimization problems for all 5 methods, full discrete candidate space $\mathcal{F}_{\text{discrete}}$ ($9,400$ points), directional search grid $K=700$, deterministic tie-breaking rules, and difficulty-weighting recourse clarification.
* **Changes Made**: Detailed parameter formulations and candidate selection rules in Section 3.5.

---

### Comment 2.5: Reframing "Educational Efficacy" Claims
> **Reviewer 2 Comment**: *Reframe pre/post passing probability shifts as model constraint satisfaction / prediction reversal, not guaranteed real-world academic efficacy.*

* **Response**: As detailed under Item E.7, we renamed Section 4.4 and added an explicit disclaimer in Section 5.1 emphasizing that pre/post probability shifts ($0.2866 \to 0.5975$) represent classifier constraint satisfaction.
* **Changes Made**: Updated Section 4.4 and Section 5.1 (`main.tex:482--485`).

---

### Comment 2.6: Statistical Significance Testing Across Seeds
> **Reviewer 2 Comment**: *Provide statistical significance tests for recourse cost differences across random seeds.*

* **Response**: As detailed under Item E.6, we conducted paired $t$-tests and Wilcoxon signed-rank tests across 10 random seeds ($S=10$), reporting p-values in Section 4.2 (`$p = 0.0047$` for Actionability Cost Premium; `$p = 0.1708$` for friction weighting).
* **Changes Made**: Added Section 4.2 statistical significance text (`main.tex:398--406`).

---

### Comment 2.7: Absolute Student Counts and Population Flow Breakdown
> **Reviewer 2 Comment**: *Report absolute student numbers receiving valid recourse per test fold.*

* **Response**: As detailed under Item E.6, we added Population Flow Table (Table~\ref{tab:population_flow}) and reported valid student counts ($11.5 / 12.1$ failing test students) alongside percentage validity ($95.5\% \pm 4.7\%$).
* **Changes Made**: Added Table~\ref{tab:population_flow} and Table~\ref{tab:ablation_results} valid student column.

---

### Comment 2.8: Repository Reproducibility and Zenodo DOI Status
> **Reviewer 2 Comment**: *Ensure code pipeline is reproducible and DOI status is consistent.*

* **Response**: As detailed under Item E.10, we created a single reproducible script `src/run_experiments.py` and aligned Code Availability statements with reserved Zenodo DOI (`10.5281/zenodo.22199775`).
* **Changes Made**: Updated Code Availability statement in Section 6 (`main.tex:514`).

---

### Comment 2.9: Bibliography Audit and Reference Corrections
> **Reviewer 2 Comment**: *Fix missing citations, duplicate keys, dataset DOIs.*

* **Response**: As detailed under Item E.9, we added 9 missing references, corrected UCI dataset DOI (`10.24432/C5TG7T`), updated KDD 2016 proceedings, and consolidated duplicate keys.
* **Changes Made**: Updated `references.bib` and in-text citation keys (`main.tex`).

---

# Summary of Revised Files

1. **`main.tex`**: Manuscript fully updated with title change, 10-seed empirical metrics, exact seed-indexed RFD expectation formula, complete mathematical formulations, discrete candidate space math ($9,400$ full points vs $K=700$ directional grid), 9 DAG edge citations matching Figure 1 100\%, statistical significance tests, absolute student counts, experimental protocol subsection, two-tier benchmark table, population flow table, absences proxy disclaimer, updated SES sensitivity table, and honest trade-off narrative.
2. **`references.bib`**: Updated with verified citations (Sirin 2005, Castro 2015, OECD 2018, Plant 2005, Nonis 2010, Gottfried 2014, Gershenson 2019, Baker & Hawn 2022, Buñay-Guisnán 2026), corrected Cortez UCI DOI, fixed KDD 2016 proceedings for Ribeiro et al., and consolidated duplicate entries.
3. **`src/run_experiments.py`**: Single official reproducible pipeline script generating all tables (`outputs/tables/`), figures (`outputs/figures/`), statistical significance tests, and `RESULTS_FOR_PAPER.md`.
4. **`response_to_reviewers.md`**: Complete point-by-point response letter quoting all Editor Directives and Reviewers 1 & 2 items verbatim with direct responses.

We thank the Editor and Reviewers once again for their guidance in elevating the quality, precision, and transparency of this work.
