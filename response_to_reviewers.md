# Response to Reviewers and Editor

**Manuscript Title**: DAG-informed actionable recourse for student academic performance prediction  
**Journal / Conference**: Major Revision Submission  

---

We sincerely thank the Editor and the Reviewers for their constructive, thorough, and highly insightful feedback on our manuscript. In response to these recommendations, we have conducted a comprehensive re-analysis using an open-source, end-to-end reproducible pipeline script (`src/run_experiments.py`), revised the mathematical optimization formulation, corrected the seed-indexed Recourse Fairness Difference (RFD) formula, added statistical significance tests and absolute student counts, narrowed causal recourse claims to DAG-informed actionable recourse, updated all table and figure metrics, and restructured the manuscript to provide a completely point-by-point response.

Below, we present our detailed, point-by-point responses organized into three dedicated sections: **1. Responses to Editor Directives**, **2. Responses to Reviewer 1 Comments**, and **3. Responses to Reviewer 2 Comments**.

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
     - Sample expectation of per-seed group absolute differences: $\text{RFD} = 0.1613 \pm 0.1038$ for Proposed Recourse (and $0.1585 \pm 0.1038$ for Actionable Recourse).
     - Macro difference between overall group means: $|\bar{\mu}_L - \bar{\mu}_H| = |0.3408 - 0.2638| = 0.0771$ for Proposed Recourse (and $|0.3381 - 0.2638| = 0.0744$ for Actionable Recourse).

  4. **High-SES Cost Stability ($0.2638 \pm 0.1633$)**:
     High-SES recourse cost is identical across Actionability-constrained, Fairness-weighted, and Proposed methods because per 20\% test fold ($130$ students), there are on average only **$3.2$ High-SES predicted failing students** ($N_{\text{fail,High}} \approx 3.2$). Candidate selection for these few students is stable under standard weights, and Proposed weights apply specifically to Low-SES (`ses == 0`). We explicitly document this sample size breakdown in Table~\ref{tab:population_flow}.
* **Changes Made**: Updated Abstract, Section 3.6 (Equation 5), Section 4.2 (Table~\ref{tab:ablation_results}), Section 5, and Section 6.

---

### Item E.3: Narrowing Causal Terminology and Adding DAG Edge Justifications
> **Directive**: *Clarify the exact nature of the causal model. State clearly whether structural equations were estimated and what "DAG-constrained recourse" means in practice.*

* **Response**: We have narrowed the terminology throughout the manuscript from "Fair Causal Recourse" to **"DAG-Informed Actionable Recourse"**. We explicitly clarify that our framework implements DAG-constrained and actionability-constrained recourse under an assumed educational causal graph based on domain literature, rather than a fully identified Structural Causal Model (SCM) with estimated structural equations.

  In Section 3.3, we added individual justifications and educational literature citations for every directed edge in the DAG (Figure~\ref{fig:dag}):
  - $\text{SES} \to \text{Medu/Fedu}, \text{famsize}, \text{internet}$: Socioeconomic status directly determines household digital access, family size constraints, and parental educational background \cite{Sirin2005}.
  - $\text{Medu/Fedu} \to \text{studytime}, \text{absences}$: Higher parental education correlates with enhanced home learning support, structured study habits, and lower student absenteeism \cite{Castro2015,OECD2018}.
  - $\text{studytime} \to y$: Academic time investment is a direct behavioral determinant of subject mastery and passing probability \cite{Plant2005,Nonis2010}.
  - $\text{absences} \to y$: Chronic absenteeism directly reduces instructional exposure, impairing academic performance \cite{Gottfried2014,Gershenson2019}.
* **Changes Made**: Inserted explicit statement and edge justifications in Section 3.1 and Section 3.3 (`main.tex:232--246`).

---

### Item E.4: Complete Mathematical Objective Formulation
> **Directive**: *Formulate explicit objective functions for all ablation conditions.*

* **Response**: In Section 3.5 (`main.tex:251--280`), we provided formal mathematical optimization objectives for all five evaluated conditions under a discrete candidate space $\mathcal{F}_{\text{discrete}}$ and a controlled candidate search budget $K = |\mathcal{F}_{\text{discrete}}| = 700$ grid points per student:
  1. **Unconstrained discrete search (DiCE-style)**: $\min_{\boldsymbol{\delta}} \mathcal{C}_{\text{unweighted}}(\boldsymbol{\delta}) \quad \text{s.t.} \quad f(\mathbf{x}_i+\boldsymbol{\delta}) \ge 0.5$.
  2. **Actionable Recourse (Ustun-style discrete grid)**: Directional bounds ($\delta_{\text{study}} \ge 0, \delta_{\text{abs}} \le 0$) under unweighted normalized feature distance.
  3. **Fairness-weighted discrete search**: Directional bounds under domain difficulty weights $w_j$ ($w_{\text{abs}}=1.5$).
  4. **Proposed Fair DAG-Informed Recourse**: Directional bounds under structural attendance friction weights for Low-SES students ($w_{\text{abs}}^{\text{Low}}=2.0$).
  5. **FACE-like kNN graph recourse (Equal search budget $K=700$)**: Nearest passing instance in the training set manifold satisfying directional bounds, evaluated over top-$K$ distance-ranked passing instances.

  Additionally, we specified:
  - **Deterministic Tie-Breaking Rule**: If multiple candidates yield equal minimal scoring cost, selection picks candidate with (i)~smallest study time increase $\delta_{\text{study}}$, (ii)~smallest absolute absence change $|\delta_{\text{abs}}|$, and (iii)~first candidate grid index.
  - **Fairness Weighting Clarification**: Fairness weighting acts as an asymmetric feature difficulty weight $w_{\text{abs}}(\text{Low})=2.0$ reflecting structural friction, rather than a group-inequality penalty term in the loss objective.
* **Changes Made**: Detailed complete equations, candidate space $\mathcal{F}_{\text{discrete}}$, candidate budget $K=700$, and tie-breaking rules in Section 3.5.

---

### Item E.5: Equal-Budget Comparative Benchmark ($K=700$)
> **Directive**: *Ensure all sampled counterfactual search methods are evaluated under identical search budgets, seeds, and bounds.*

* **Response**: All empirical recourse baselines in the execution pipeline (`src/run_experiments.py`) were evaluated under identical discrete candidate search budgets ($K=700$ candidates per student), identical random seed splits ($42 \dots 51$), held-out test splits, and identical discrete UCI ordinal feature bounds.
* **Changes Made**: Standardized FACE-like kNN baseline candidate pool to $K=700$ and reported comparative results in Table~\ref{tab:ablation_results} and Table~\ref{tab:two_tier_benchmark}.

---

### Item E.6: Statistical Reporting, Significance Testing, and Absolute Student Counts
> **Directive**: *Report sample size (N), class prevalence, Low/High SES counts, predicted failures, recourse validity, and statistical significance.*

* **Response**: We expanded Section 4 with comprehensive statistical and sample flow reporting:
  1. **Population Flow Table (Table~\ref{tab:population_flow})**: Documenting total $N=649$, pass/fail prevalence ($84.6\%$ / $15.4\%$), Low-SES ($N=307$), High-SES ($N=342$), mean test fold size ($130.0$), mean predicted failures ($12.1$; Low-SES $8.9$, High-SES $3.2$).
  2. **Absolute Student Count Reporting**: In Table~\ref{tab:ablation_results}, we report that an average of **$11.5 \text{ out of } 12.1$** predicted failing students per test fold receive valid recourse ($95.5\% \pm 4.7\%$).
  3. **Statistical Significance Testing (Section 4.2)**: Conducted paired two-tailed $t$-tests and Wilcoxon signed-rank tests across 10 random seeds ($S=10$):
     - Unconstrained vs Actionable Recourse: $+0.1133$ cost increase, $t(9) = 3.72, p = 0.0047$; Wilcoxon $W = 0.0, p = 0.0039$ (**Statistically Significant**).
     - Actionable Recourse vs Proposed Fair DAG-Informed Recourse: $+0.0027$ cost diff, $t(9) = 1.49, p = 0.1708$; Wilcoxon $W = 5.0, p = 0.5000$ (**Not Statistically Significant**).
     - Actionable Recourse vs FACE-like kNN: $+0.0413$ cost diff, $t(9) = 1.83, p = 0.0980$; Wilcoxon $W = 10.0, p = 0.0977$ (**Not Statistically Significant**).
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

### Item E.10: Code Release and Zenodo DOI Reservation
> **Directive**: *Ensure repository reproducibility and permanent DOI archive.*

* **Response**: We created a single, self-contained, reproducible execution pipeline script (`src/run_experiments.py`) that generates all tables (`outputs/tables/`), figures (`outputs/figures/`), and markdown results (`RESULTS_FOR_PAPER.md`).
* **Changes Made**: Updated Code Availability statement (`main.tex:492`) pointing to GitHub (`https://github.com/kadirkesgin/fair-ed`) and permanently archived on Zenodo (DOI reserved: `10.5281/zenodo.22199775`).

---

# 2. Responses to Reviewer 1 Comments

### Comment 1.1: Comparative Benchmark and Method Positioning
> **Reviewer Comment**: *The paper needs a clearer comparative benchmark against existing recourse algorithms (e.g., Actionable Recourse, FACE, DiCE).*

* **Response**: We thank Reviewer 1 for this valuable suggestion. We constructed a Two-Tier Benchmark Comparison Table (Table~\ref{tab:two_tier_benchmark}) separating recourse methods into:
  1. **Empirical Layer (Evaluated under identical candidate budget $K=700$)**: Unconstrained discrete search (DiCE-style), Actionable Recourse (Ustun-style discrete grid), Fairness-weighted search, Proposed Fair DAG-Informed Recourse, and FACE-like kNN graph recourse.
  2. **Conceptual Layer (Theoretical contrast)**: Karimi Causal Recourse \cite{Karimi2021}, von Kügelgen Fair Recourse \cite{vonKugelgen2022}, and Ustun Integer Linear Model \cite{Ustun2019}.
* **Changes Made**: Added Table~\ref{tab:two_tier_benchmark} in Section 4.3 detailing candidate budget $K=700$, implementation rationale, and model compatibility requirements.

---

### Comment 1.2: Clarifying FACE-like Baseline Implementation and Equal Search Budget $K$
> **Reviewer Comment**: *Ensure FACE baseline is accurately described and evaluated under comparable search conditions.*

* **Response**: We updated the FACE-like baseline in `src/run_experiments.py` to evaluate over top-$K$ distance-ranked training passing instances ($K=700$), matching the exact candidate grid budget of optimization methods. We added explicit footnotes in Table~\ref{tab:ablation_results} and Section 3.5 describing this graph-manifold baseline.
* **Changes Made**: Updated Section 3.5, Section 4.2 (Table~\ref{tab:ablation_results}), and `src/run_experiments.py`.

---

### Comment 1.3: Scope of Causal Reasoning and Absence of SCM Structural Equations
> **Reviewer Comment**: *Clarify causal assumptions and edge justifications.*

* **Response**: As detailed under Item E.3, we narrowed our terminology to "DAG-Informed Actionable Recourse" and added explicit educational literature justifications and citations for every directed edge in Figure~\ref{fig:dag} in Section 3.3.
* **Changes Made**: Updated Section 3.1 and Section 3.3 (`main.tex:232--246`).

---

### Comment 1.4: Dataset Selection and Removal of Unaligned External Benchmark (OULAD)
> **Reviewer Comment**: *Address dataset limitations and cross-dataset feature mismatch.*

* **Response**: We agreed with Reviewer 1 that feature alignment between UCI and OULAD was too loose (lacking direct study time and absenteeism measures) to justify a quantitative external replication claim. We removed OULAD from quantitative results tables and positioned it as a conceptual framework for future external validation.
* **Changes Made**: Updated Section 5.2 Limitations.

---

# 3. Responses to Reviewer 2 Comments

### Comment 2.1: Discrepancy in Equation (5) and RFD Calculation vs Table Group Means
> **Reviewer Comment**: *Equation (5) defines RFD as $|\mu_{\text{Low}} - \mu_{\text{High}}|$. However, the reported values (0.1131, 0.1585, 0.1613) do not equal the absolute difference between reported group means ($0.3408 - 0.2638 = 0.0770$). Please clarify and correct this discrepancy.*

* **Response**: We thank Reviewer 2 for identifying this critical mathematical inconsistency. As detailed under Item E.2, we updated Equation (5) to the seed-indexed expectation formula:
  $$\text{RFD} = \frac{1}{S} \sum_{s=1}^S |\mu_{L,s} - \mu_{H,s}|$$
  Because fold-level cost variations exist across test splits, Jensen's inequality dictates that the expected value of fold-level absolute differences $\text{RFD} = \frac{1}{S}\sum_{s=1}^S |\mu_{L,s} - \mu_{H,s}| \ge |\bar{\mu}_L - \bar{\mu}_H|$.
  
  In Table~\ref{tab:ablation_results}, both values are now explicitly reported and explained:
  - Sample expectation of per-seed group absolute differences: $\text{RFD} = 0.1613 \pm 0.1038$ for Proposed Recourse (and $0.1585 \pm 0.1038$ for Actionable Recourse).
  - Macro difference between overall group means: $|\bar{\mu}_L - \bar{\mu}_H| = |0.3408 - 0.2638| = 0.0771$ for Proposed Recourse (and $|0.3381 - 0.2638| = 0.0744$ for Actionable Recourse).
* **Changes Made**: Updated Section 3.6 (Equation 5), Section 4.2 (Table~\ref{tab:ablation_results}), Section 5, and Section 6.

---

### Comment 2.2: Primary Composite SES Definition Consistency
> **Reviewer Comment**: *Ensure the primary SES definition in code matches the manuscript description.*

* **Response**: We verified that our primary SES variable in `src/run_experiments.py` matches the composite index described in the manuscript:
  $$\text{Composite Score} = \text{Medu} + \text{Fedu} + \text{famsize\_small} + \text{internet\_yes}$$
  binarized at the median threshold ($6.0$). In addition, we added an SES Sensitivity Analysis Table (Table~\ref{tab:ses_sensitivity}) evaluating alternative definitions (`Medu > 2` and `Medu + Fedu > 4`) across the exact same 10-seed evaluation protocol.
* **Changes Made**: Updated Section 3.2 and Section 4.5 (Table~\ref{tab:ses_sensitivity}).

---

### Comment 2.3: Historical Absences Proxy and Mid-Term Intervention Disclaimer
> **Reviewer Comment**: *Address the retrospective nature of recorded absences in the UCI dataset.*

* **Response**: As detailed under Item E.8, we added explicit disclaimers in Section 3.4 and Section 5.2 clarifying that recorded absences represent historical data and are treated as a constrained actionable proxy for future engagement.
* **Changes Made**: Updated Section 3.4 and Section 5.2 (`main.tex:247--250`).

---

### Comment 2.4: Mathematical Optimization Formulation and Candidate Selection Rules
> **Reviewer Comment**: *Provide complete mathematical objectives, candidate sets, and tie-breaking rules.*

* **Response**: As detailed under Item E.4, we updated Section 3.5 with formal minimization problems for all 5 methods, discrete candidate space $\mathcal{F}_{\text{discrete}}$, budget $K=700$, deterministic tie-breaking rules, and fairness weighting clarification.
* **Changes Made**: Detailed parameter formulations and candidate selection rules in Section 3.5.

---

### Comment 2.5: Reframing "Educational Efficacy" Claims
> **Reviewer Comment**: *Reframe pre/post passing probability shifts as model constraint satisfaction / prediction reversal, not guaranteed real-world academic efficacy.*

* **Response**: As detailed under Item E.7, we renamed Section 4.4 and added an explicit disclaimer in Section 5.1 emphasizing that pre/post probability shifts ($0.2866 \to 0.5975$) represent classifier constraint satisfaction.
* **Changes Made**: Updated Section 4.4 and Section 5.1 (`main.tex:474--476`).

---

### Comment 2.6: Statistical Significance Testing Across Seeds
> **Reviewer Comment**: *Provide statistical significance tests for recourse cost differences across random seeds.*

* **Response**: As detailed under Item E.6, we conducted paired $t$-tests and Wilcoxon signed-rank tests across 10 random seeds ($S=10$), reporting p-values in Section 4.2 (`$p = 0.0047$` for Actionability Cost Premium; `$p = 0.1708$` for friction weighting).
* **Changes Made**: Added Section 4.2 statistical significance text (`main.tex:375--385`).

---

### Comment 2.7: Absolute Student Counts and Population Flow Breakdown
> **Reviewer Comment**: *Report absolute student numbers receiving valid recourse per test fold.*

* **Response**: As detailed under Item E.6, we added Population Flow Table (Table~\ref{tab:population_flow}) and reported valid student counts ($11.5 / 12.1$ failing test students) alongside percentage validity ($95.5\% \pm 4.7\%$).
* **Changes Made**: Added Table~\ref{tab:population_flow} and Table~\ref{tab:ablation_results} valid student column.

---

### Comment 2.8: Repository Reproducibility and Zenodo DOI Status
> **Reviewer Comment**: *Ensure code pipeline is reproducible and DOI status is consistent.*

* **Response**: As detailed under Item E.10, we created a single reproducible script `src/run_experiments.py` and aligned Code Availability statements with reserved Zenodo DOI (`10.5281/zenodo.22199775`).
* **Changes Made**: Updated Code Availability statement in Section 6 (`main.tex:492`).

---

### Comment 2.9: Bibliography Audit and Reference Corrections
> **Reviewer Comment**: *Fix missing citations, duplicate keys, dataset DOIs.*

* **Response**: As detailed under Item E.9, we added 9 missing references, corrected UCI dataset DOI (`10.24432/C5TG7T`), updated KDD 2016 proceedings, and consolidated duplicate keys.
* **Changes Made**: Updated `references.bib` and in-text citation keys (`main.tex`).

---

# Summary of Revised Files

1. **`main.tex`**: Manuscript fully updated with title change, 10-seed empirical metrics, exact seed-indexed RFD expectation formula, complete mathematical formulations, candidate search budget $K=700$, DAG edge citations, statistical significance tests, absolute student counts, two-tier benchmark table, population flow table, absences proxy disclaimer, updated SES sensitivity table, and honest trade-off narrative.
2. **`references.bib`**: Updated with verified citations (Sirin 2005, Castro 2015, OECD 2018, Plant 2005, Nonis 2010, Gottfried 2014, Gershenson 2019, Baker & Hawn 2022, Buñay-Guisnán 2026), corrected Cortez UCI DOI, fixed KDD 2016 proceedings for Ribeiro et al., and consolidated duplicate entries.
3. **`src/run_experiments.py`**: Single official reproducible pipeline script generating all tables (`outputs/tables/`), figures (`outputs/figures/`), statistical significance tests, and `RESULTS_FOR_PAPER.md`.
4. **`response_to_reviewers.md`**: Complete point-by-point response letter addressing Editor Directives and Reviewers 1 & 2 item-by-item.

We thank the Editor and Reviewers once again for their guidance in elevating the quality, precision, and transparency of this work.
