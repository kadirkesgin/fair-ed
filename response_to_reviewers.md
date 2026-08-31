# Response to Reviewers and Editor

**Manuscript Title**: Fair causal recourse for student academic performance prediction  
**Journal / Conference**: Major Revision Submission  

---

We sincerely thank the Editor and the Reviewers for their constructive, thorough, and highly insightful feedback on our manuscript. In response to these recommendations, we have conducted a comprehensive re-analysis using an open-source, end-to-end reproducible pipeline, revised the mathematical optimization formulation, removed artificial group multipliers from evaluation metrics, toned down causality and efficacy claims, updated all table and figure metrics, and restructured the narrative to reflect the honest empirical trade-off between unconstrained search and actionable recourse.

Below, we provide a point-by-point response organized into three dedicated sections: **Editor Directives**, **Reviewer 1 Comments**, and **Reviewer 2 Comments**.

---

# 1. Responses to Editor Directives

### Item E.1: Manuscript Title Revision (In-House Formatting Guideline)
> **Directive**: *Ensure that the title contains no colons (:).*

* **Response**: We have updated the manuscript title to comply strictly with the journal formatting guideline.
* **Changes Made**: The title has been changed from *"From Prediction to Intervention: Fair Causal Recourse for Student Academic Performance"* to:
  $$\textbf{Fair causal recourse for student academic performance prediction}$$

---

### Item E.2: Correction of Equation (5), RFD Reporting, and Group-Independent Metric Formulation
> **Directive**: *Address inconsistency between Equation (5) and reported Recourse Fairness Difference (RFD) values. Ensure evaluation cost is group-independent and report exact re-analyzed statistics.*

* **Response**: We thank the Editor and Reviewer 2 for bringing this critical issue to our attention.

  1. **Group-Independent Evaluation Cost Metric**:
     We removed the artificial group penalty multiplier ($\mu_S = 2.0$) from the benchmark evaluation cost formula to prevent RFD from becoming an artifact of metric definition. All methods in the ablation table are evaluated using the exact same group-independent normalized feature distance metric:
     $$\mathcal{C}_{\text{eval}}(\boldsymbol{\delta}) = \sum_{j \in \text{actionable}} \lambda_j |\delta_j|$$
     where $\lambda_{\text{study}} = 1/3, \lambda_{\text{abs}} = 1/93, \lambda_{\text{free}} = 1/4, \lambda_{\text{go}} = 1/4$.

  2. **Equivalence of Actionability-Constrained and Fairness-Weighted Rows**:
     Under the common evaluation metric, fairness-weighted search selects the exact same candidate actions as actionability-constrained search ($\mu_L = 0.3381 \pm 0.1457, \mu_H = 0.2638 \pm 0.1633$). We do not disguise this equivalence as two separate empirical wins; we explicitly add a footnote ($\dagger$) in Table~\ref{tab:ablation_results} clarifying that the two rows are identical by design of the selected counterfactuals under standard domain weights.

  3. **High-SES Cost Stability ($0.2638 \pm 0.1633$)**:
     High-SES recourse cost is identical across Actionability-constrained, Fairness-weighted, and Proposed methods because per 20\% test fold ($130$ students), there are on average only **$3.2$ High-SES failing students** ($N_{\text{fail,High}} \approx 3.2$). Candidate selection for these few students is stable under standard weights, and Proposed weights apply specifically to Low-SES (`ses == 0`). We explicitly document this sample size imbalance in Table~\ref{tab:population_flow}.

  4. **Honest Scientific Framing (Actionability Cost Premium)**:
     Under Equation (5), unconstrained baseline RFD is $0.1131 \pm 0.0635$, actionability-constrained RFD is $0.1585 \pm 0.1038$, fairness-weighted RFD is $0.1585 \pm 0.1038$, and proposed fair causal RFD is $0.1613 \pm 0.1038$.
     
     We do not claim that fairness weighting reduced numerical cost or that proposed recourse is empirically superior to actionability constraints. We unbolded the Proposed row in Table~\ref{tab:ablation_results} and structure our core narrative around three concise, honest points:
     - *Unconstrained search* achieves lower numerical feature distance ($0.2248$) because it exploits educationally invalid shortcut directions (e.g. recommending reduced study time or increased absences).
     - *Directional actionability constraints* ($\delta_{\text{study}} \ge 0, \delta_{\text{abs}} \le 0$) eliminate these invalid shortcuts; numerical distance rises ($0.3381$, reflected as an Actionability Cost Premium $\text{RBR}_L = -50.4\%$), while validity remains high ($95.5\% \pm 4.7\%$).
     - *Fairness weighting and proposed recourse* do not produce an empirical cost reduction over actionability constraints alone ($0.3381$ vs $0.3408$). The primary value of the proposed framework lies in eliminating invalid non-causal shortcuts and ensuring high validity under plausible directional bounds.
* **Changes Made**: Updated Abstract, Section 3.5, Section 3.8, Section 4.2 (Table~\ref{tab:ablation_results}), Section 5, and Section 6.

---

### Item E.3: Clarification of Causal Model Scope and Assumptions
> **Directive**: *Clarify the exact nature of the causal model. State clearly whether structural equations were estimated and what "DAG-constrained recourse" means in practice.*

* **Response**: We have refined the causal language throughout the manuscript. We explicitly clarify that our framework implements DAG-constrained and actionability-constrained recourse under an assumed educational causal graph based on domain literature, rather than a fully identified SCM with estimated structural equations.
* **Changes Made**: Inserted the following explicit statement in Section 3.1 and Section 3.3:
  > *"The framework implements DAG-constrained / actionability-constrained recourse under an assumed educational causal graph. It is not a fully identified SCM with estimated structural equations. DAG parent restrictions mean immutable/sensitive features remain fixed while only actionable features vary under directional constraints. Structural equations are not estimated, and downstream $do()$ propagation is not performed."*

---

### Item E.4: Complete Mathematical Objective Formulation
> **Directive**: *Formulate explicit objective functions for all ablation conditions.*

* **Response**: In Section 3.5, we have provided formal mathematical objectives for all evaluated ablation conditions:
  1. **Unconstrained discrete search (DiCE-style)**: $\min_{\boldsymbol{\delta}} \mathcal{C}_{\text{unweighted}}(\boldsymbol{\delta}) \quad \text{s.t.} \quad f(\mathbf{x}+\boldsymbol{\delta}) \ge 0.5$.
  2. **Actionability-constrained discrete search**: Directional bounds ($\delta_{\text{study}} \ge 0, \delta_{\text{abs}} \le 0$) under unweighted normalized feature distance.
  3. **Fairness-weighted discrete search**: Directional bounds under domain difficulty weights $w_j$.
  4. **Proposed Fair Causal Recourse**: Directional bounds under structural attendance friction weights $w_j(S)$.
* **Changes Made**: Detailed parameters $w_j$, bounds normalization factors $\lambda_j$, and candidate selection mechanisms in Section 3.5.

---

### Item E.5: Equal-Budget Comparative Benchmark
> **Directive**: *Ensure all sampled counterfactual search methods are evaluated under identical search budgets, seeds, and bounds.*

* **Response**: All empirical search baselines in the re-analysis were evaluated using identical discrete candidate search grids (evaluating ~150 unique candidate vectors per student under directional bounds), identical random seed sets ($42 \dots 51$), held-out test splits, and identical discrete UCI ordinal feature bounds.
* **Changes Made**: Reported multi-seed aggregate metrics in Table~\ref{tab:ablation_results}.

---

### Item E.6: Statistical Reporting and Population Flow
> **Directive**: *Report sample size (N), class prevalence, Low/High SES counts, predicted failures, and recourse validity.*

* **Response**: We added a dedicated Population and Sample Flow Table (Table~\ref{tab:population_flow}) documenting:
  - Total population: $N = 649$ (`student-por.csv`)
  - Class prevalence: Passed $549$ ($84.6\%$), Failed $100$ ($15.4\%$)
  - Low-SES ($N=307$), High-SES ($N=342$) under primary composite index
  - Mean test set size per fold ($130.0$), mean predicted failures ($12.1$; $N_{\text{fail,Low}}=8.9$, $N_{\text{fail,High}}=3.2$)
  - Recourse validity rate ($95.5\% \pm 4.7\%$).
* **Changes Made**: Added Table~\ref{tab:predictive_perf} and Table~\ref{tab:population_flow} in Section 4.1.

---

### Item E.7: Toning Down "Educational Efficacy" Language
> **Directive**: *Reframe pre/post passing probability shifts as model constraint satisfaction / prediction reversal, not guaranteed real-world academic efficacy.*

* **Response**: We renamed Section 4.8 to *"Model-based prediction reversal and recourse validity"*. We updated pre-recourse mean passing probability ($0.2866$) and post-recourse probability ($0.5975$) directly from script outputs.
* **Changes Made**: Added an explicit disclaimer in Section 5.1:
  > *"This demonstrates that generated profiles cross the same classifier's decision boundary. It does not show that the recommended behavioral changes would cause real academic improvement. That would require longitudinal or prospective evidence."*

---

### Item E.8: Historical Absences Proxy Disclaimer
> **Directive**: *Address the retrospective nature of recorded absences in the UCI dataset.*

* **Response**: We added an explicit disclaimer clarifying that recorded absences represent historical data and are treated as a constrained actionable proxy for future engagement.
* **Changes Made**: Added the following text in Section 3.4 and Section 5.2:
  > *"Recorded absences are historical. Recommended reductions are simulated hypothetical changes to a future-attendance / engagement proxy, not a retrospective rewrite of past absences. Prediction is assumed to be issued mid-term / before final outcome, but the UCI snapshot does not encode this timing. Therefore absences are treated as a constrained actionable proxy, not a directly implementable past intervention."*

---

### Item E.9: Bibliography and Reference Cleanup
> **Directive**: *Add missing text citations, fix duplicate references, and correct dataset DOIs.*

* **Response**: We performed a full reference audit:
  - Added missing entries: Baker and Hawn (2022) \cite{Baker2022}, Buñay-Guisnán et al. (2026) \cite{BunayGuisnan2026}.
  - Updated Paulo Cortez (2008) dataset entry with official UCI repository URL and DOI (`10.24432/C5TG7T`).
  - Removed duplicate Türkmen reference key (`[17]`), consolidating to `Türkmen (2025)` \cite{Turkmen2025}.
  - Corrected Ribeiro et al. \cite{Ribeiro2016} entry to KDD 2016 proceedings.
  - Fixed TeX escaping for special author names (`Sch{\"{o}}lkopf`).
* **Changes Made**: Updated `references.bib` and in-text citation keys.

---

### Item E.10: Code Release and Zenodo DOI
> **Directive**: *Ensure repository reproducibility and permanent DOI archive.*

* **Response**: We created a unified execution pipeline script (`src/run_experiments.py`) that produces all tables (`outputs/tables/`), figures (`outputs/figures/`), and markdown results (`RESULTS_FOR_PAPER.md`).
* **Changes Made**: Updated Code Availability statement to point to GitHub (`https://github.com/kadirkesgin/fair-ed`) and reserved a DOI placeholder for Zenodo archive release.

---

# 2. Responses to Reviewer 1 Comments

### Comment 1.1: Comparative Benchmark and Method Positioning
> **Reviewer Comment**: *The paper needs a clearer comparative benchmark against existing recourse algorithms (e.g., Actionable Recourse, FACE, DiCE).*

* **Response**: We thank Reviewer 1 for this suggestion. We constructed a Two-Tier Benchmark Comparison (Table~\ref{tab:two_tier_benchmark}) separating methods into:
  1. **Empirical Layer (Evaluated under identical candidate pool search budget)**: Unconstrained discrete search (DiCE-style), Actionability-constrained discrete search, Fairness-weighted search, Proposed Fair Causal Recourse, and FACE-like kNN graph recourse.
  2. **Conceptual Layer (Theoretical contrast)**: Karimi Causal Recourse \cite{Karimi2021}, von Kügelgen Fair Recourse \cite{vonKugelgen2022}, and Ustun Actionable Recourse \cite{Ustun2019}.
* **Changes Made**: Added Table~\ref{tab:two_tier_benchmark} in Section 4.3 detailing implementation rationale and model compatibility requirements.

---

### Comment 1.2: Clarifying FACE-like Baseline Implementation
> **Reviewer Comment**: *Ensure FACE baseline is accurately described.*

* **Response**: We added an explicit footnote in Table~\ref{tab:ablation_results} and Section 3.5 noting that our FACE-like baseline represents a nearest-neighbor counterfactual search on the training data manifold satisfying actionability bounds, rather than a sampled candidate optimization method.
* **Changes Made**: Footnote added to Table~\ref{tab:ablation_results}: *\textsuperscript{*}Nearest-neighbor baseline on dataset manifold, not a sampled-K optimization method.*

---

# 3. Responses to Reviewer 2 Comments

### Comment 2.1: Discrepancy in Equation (5) and RFD Calculation
> **Reviewer Comment**: *Equation (5) defines RFD as $|\mu_{\text{Low}} - \mu_{\text{High}}|$. However, the reported values (0.78, 0.82, 0.85) do not equal the absolute difference between the reported group means. Please clarify and correct this discrepancy.*

* **Response**: We thank Reviewer 2 for identifying this critical inconsistency. We verified our analysis code, recomputed all group means, fixed the evaluation cost formula to be group-independent, and corrected all tables, figures, abstract, results, and discussion text.

  Under Equation (5), baseline unconstrained RFD is $0.1131 \pm 0.0635$, actionability-constrained RFD is $0.1585 \pm 0.1038$, fairness-weighted RFD is $0.1585 \pm 0.1038$, and proposed fair causal RFD is $0.1613 \pm 0.1038$. We no longer claim that between-group absolute disparity decreased. The substantively robust finding is that enforcing directional actionability constraints ($\delta_{\text{study}} \ge 0, \delta_{\text{abs}} \le 0$) reflects realistic educational friction, increasing Low-SES action feature distance from $0.2248$ to $0.3381$ while maintaining high validity ($95.5\% \pm 4.7\%$).

  We now explicitly report both (i) disadvantaged-group recourse burden reduction ($\text{RBR}_L$) and (ii) between-group RFD neutrally.
* **Changes Made**: All numbers in Abstract, Section 3.8, Section 4 (Table~\ref{tab:ablation_results}, Figures 1--3), Section 5, and Section 6 were updated to match exact 10-seed pipeline outputs.

---

### Comment 2.2: SES Definition Consistency
> **Reviewer Comment**: *Ensure the primary SES definition in code matches the manuscript description.*

* **Response**: We verified that our primary SES variable in `src/run_experiments.py` matches the composite index described in the manuscript:
  $$\text{Composite Score} = \text{Medu} + \text{Fedu} + \text{famsize\_small} + \text{internet\_yes}$$
  binarized at the median threshold ($6.0$). In addition, we added an SES Sensitivity Analysis Table (Table~\ref{tab:ses_sensitivity}) evaluating alternative definitions (`Medu > 2` and `Medu + Fedu > 4`) across the exact same 10-seed multi-split evaluation protocol.
* **Changes Made**: Updated Section 3.2 and Section 4.5 (Table~\ref{tab:ses_sensitivity}).

---

# Summary of Revised Files

1. **`main.tex`**: Manuscript fully updated with new title, 10-seed empirical metrics, exact formulas, two-tier benchmark table, population flow table, absences proxy disclaimer, updated SES sensitivity table, and honest trade-off narrative.
2. **`references.bib`**: Updated with verified citations (Baker & Hawn 2022, Buñay-Guisnán 2026), corrected Cortez UCI DOI, fixed KDD 2016 proceedings for Ribeiro et al., and consolidated duplicate entries.
3. **`src/run_experiments.py`**: Single official reproducible pipeline script generating all tables, figures, and `RESULTS_FOR_PAPER.md`.
4. **`response_to_reviewers.md`**: Detailed point-by-point response letter to Editor and Reviewers.

We thank the Editor and Reviewers once again for their guidance in elevating the quality, precision, and transparency of this work.
