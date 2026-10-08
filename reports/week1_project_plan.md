# Week 1 — Project Planning Package
## End-to-End Machine Learning Pipeline for Predicting Student Dropout and Academic Success

**Module:** University Machine Learning Project — Week 1 Deliverable (Planning Only)
**Status:** Planning decisions documented. No dataset downloaded. No models trained. No experimental results reported.
**Scope:** Sections 1–21 as required for Week 1. Methods below are *proposed* unless marked as completed scaffold work. Results are *future work* for Weeks 2–6.

**Repository locations:**
- Scaffold: `data/`, `notebooks/`, `src/`, `models/`, `reports/`, `tests/`, `README.md`, `requirements.txt`
- This plan: `reports/week1_project_plan.md` (this file)

---

## 1. Project Introduction

Higher education attrition remains a persistent academic, economic and social challenge. Early identification of students at risk of dropout enables timely pedagogical intervention, targeted support, and improved resource allocation. Understanding factors linked to completion (Graduate) and ongoing progression (Enrolled) equally supports curriculum design and student-success policy.

This project proposes an **end-to-end, reproducible machine learning pipeline** for three-way classification of student academic outcome as **Dropout, Enrolled or Graduate**, using the UCI *Predict Students' Dropout and Academic Success* dataset [1,2]. The pipeline will span ingestion, preprocessing, feature engineering, training, validation, optimisation, interpretation and reporting, in Python with scikit-learn and documented for academic audit.

**Week 1 position:** foundation only — problem framing, literature grounding, dataset justification, pipeline design, methodology, schedule, risks, evaluation protocol, deployment and reproducibility strategy. No EDA, engineering or modelling in Week 1.

## 2. Research and Literature Review

**Early academic indicators dominate.** First- and second-semester curricular-unit outcomes are consistently among the strongest predictors of final outcome [1,3]. Week 2 must therefore handle these features without leakage (only data available at the defined prediction point may be used).

**Context features need ethical care.** Parental education/occupation, scholarship, tuition status, displacement and age at enrolment add context but raise fairness concerns if used naively [3,4]. Inclusion rationale will be documented; error distribution across subgroups examined qualitatively in Weeks 4–5.

**Multiclass framing is harder than binary dropout.** *Enrolled* (still registered at observation time) overlaps temporally with the other classes. Work on this UCI dataset reports *Enrolled* as the minority and hardest class, motivating macro-averaged metrics and confusion-matrix analysis over accuracy alone [1,5].

**Imbalance demands evaluation discipline.** Education data are typically imbalanced. Literature recommends stratified sampling, stratified cross-validation, macro-F1 / balanced accuracy, and per-class precision-recall over raw accuracy [4,5]. Adopted in Section 16.

**Leakage control via pipelines.** `Pipeline` / `ColumnTransformer` objects fitted only on training folds are standard to avoid leakage from imputation, scaling and encoding [6]. Adopted from Week 2 onward.

**Gaps addressed:** (i) staged baseline-to-holdout reasoning instead of a single split; (ii) explicit per-class reporting for *Enrolled*; (iii) modular `src/` code plus fixed seeds and audit trail instead of throwaway notebooks.

> Planning note: literature justifies design choices. No empirical comparison is claimed in Week 1.

## 3. Research and Ideation

**Framing:** (a) binary Dropout vs. rest, (b) native three-class Dropout / Enrolled / Graduate, (c) performance regression were considered. **Decision:** retain (b) for academic fidelity, per brief.

**Features:** demographics, socio-economics, enrolment context, semester performance. **Decision:** include all groups in Week 2 EDA; prune only with evidence (multicollinearity, leakage, negligible importance) in Weeks 2/5.

**Models:** logistic regression (interpretable baseline), tree / forest, gradient boosting (HistGradientBoosting), optional k-NN / SVM contrasts, MLP only if time permits. **Decision:** logistic plus tree ensembles first in Week 3.

**Use metaphor:** batch risk-scoring dashboard for advisors vs. real-time API. **Decision (planning):** batch, human-in-the-loop advisory use; no autonomous decisions (Sections 7, 17).

**Deferred:** deep learning (disproportionate for tabular n ~ 4k), causal inference (predictive scope only), live system integration (architectural planning only, no live data in repo).

## 4. Problem Statement

Institutions hold rich enrolment and semester records but lack a reproducible, validated pipeline mapping them to **Dropout / Enrolled / Graduate** while handling mixed types, likely imbalance, and the need for interpretable, ethically aware reporting.

The task: **design, implement, validate and document an end-to-end Python pipeline that, from features available at a defined prediction point, predicts the most likely of the three classes, with honest uncertainty reporting, per-class evaluation, and clear limitations.**

Constraints: small-to-medium tabular data, mixed categorical/numerical features, likely imbalance (especially *Enrolled*), six-week budget, single-developer context, strict reproducibility and ethics requirements.

## 5. Research Question

**RQ1:** To what extent can a reproducible end-to-end pipeline trained on the UCI dataset distinguish Dropout, Enrolled and Graduate students, and which feature groups contribute most under rigorous validation?

- **RQ1a (feasibility):** macro-F1, balanced accuracy, per-class precision/recall under stratified CV, without leakage?
- **RQ1b (features):** which demographic, socio-economic and curricular groups associate most strongly, and how stable across folds?
- **RQ1c (imbalance):** how does imbalance (minority *Enrolled*) affect recall, and do Week 5 weighting/resampling options help without harming calibration?
- **RQ1d (robustness):** sensitivity to preprocessing and seeds; what do ablations and error analysis reveal?

> Week 1 note: questions *to be answered* in Weeks 2–6. No answers or figures claimed here.

## 6. Project Aim and Objectives

**Aim:** to deliver a validated, interpretable, reproducible end-to-end ML pipeline that classifies students into Dropout, Enrolled or Graduate from the UCI dataset, with per-class evaluation, imbalance awareness, and a Week 6 academic report.

**Objectives:**
- **O1 (Week 1):** formalise problem, questions, pipeline, methodology, schedule, risks and evaluation protocol. *Success: this document approved; scaffold clean.*
- **O2 (Week 2):** ingest and profile data; build leakage-safe preprocessing (`ColumnTransformer`); engineer justified features; document in `notebooks/` and `src/`.
- **O3 (Week 3):** implement baseline pipelines (logistic regression, tree ensembles) with stratified train/validation split and fixed seeds.
- **O4 (Week 4):** evaluate with stratified CV: macro-F1, balanced accuracy, per-class precision/recall, confusion matrices, OvR ROC-AUC, calibration outlook; error analysis with focus on *Enrolled*.
- **O5 (Week 5):** tune hyperparameters, test class-weight/resampling options, ablate feature groups, check seed stability; select candidate with justification.
- **O6 (Week 6):** final holdout reporting (once), interpretation (coefficients/importances), limitations, ethics reflection, and full report in `reports/`.

Out of scope: causal claims, production deployment, live student data.

## 7. Scope, Assumptions and Constraints

**In scope:** single tabular UCI dataset; three-class task; Python/scikit-learn pipeline; stratified validation; imbalance-aware metrics; interpretability and ethics discussion; academic report.

**Assumptions (to be verified in Week 2 EDA):**
- A1: dataset matches published UCI description (~4,424 rows, 37 attributes, target `Target` with three levels) [1,2].
- A2: features are available at or before the prediction point (no post-outcome leakage); any violating feature will be excluded or lagged.
- A3: rows are independent student records; no duplicate-leakage across splits (to be checked via ID/duplication audit).
- A4: missingness is modest and manageable via pipeline imputation; categorical levels are stable across splits.

**Constraints:**
- C1: six-week, single-developer, CPU-only budget — favour efficient models (logistic, forests, histogram boosting).
- C2: small n limits complex models and fine subgroup analysis; uncertainty must be reported (fold variance, CIs where feasible).
- C3: *Enrolled* ambiguity — still-registered status is time-dependent; predictions are correlational snapshots, not destiny labels.
- C4: ethics — no autonomous punitive use; outputs are advisory and must be presented with confidence and limitations.
- C5: repo hygiene — no data, models, secrets or results fabricated in advance; everything reproducible from `src/` + seeds.

## 8. Dataset Description and Justification

**Chosen dataset:** UCI Machine Learning Repository — *Predict Students' Dropout and Academic Success* [1,2] (donated by Realinho, Vieira Martins, Machado & Baptista; related publication [1]). Accessed via the UCI repository page; **not downloaded in Week 1** and not committed to this repo (see `data/` + `.gitignore`: `data/*.csv`, `*.parquet`).

**Why this dataset:**
- Directly matches the brief: native three-class target (**Dropout, Enrolled, Graduate**) — no relabelling required.
- Rich mixed-type feature set (published as 37 attributes): demographics (e.g. age at enrolment, gender, displaced), socio-economics (parental qualifications/occupations, scholarship, tuition, debtor status), enrolment context (application mode/order, course, attendance time, previous qualification), and curricular performance over two semesters (units credited/enrolled/evaluated/approved, grades, unemployment/Inflation/GDP context features) [1,2].
- Public, documented and versioned — supports reproducibility and academic citation, unlike scraped or synthetic alternatives.
- Non-trivial but tractable: n ~ 4.4k rows suits six-week CPU modelling with proper CV, while imbalance and the *Enrolled* class provide genuine evaluation challenge.

**Target definition (planning):** column `Target` with levels `Dropout`, `Enrolled`, `Graduate` to be confirmed on load in Week 2. Any discrepancy (naming, encoding, extra level) will be recorded in the EDA notebook and this plan revisited.

**Data handling (planning):** raw file lands in `data/` only (never committed); a loader in `src/` will read it with explicit dtypes and a schema check; processed outputs (if any) are regenerated, not committed. No personal data beyond the published anonymised UCI file will be introduced.

## 9. Proposed End-to-End ML Pipeline

```text
1. Ingest       UCI CSV -> data/ (manual download, Week 2) + src loader with schema check
2. Profile      shape, dtypes, missingness, target balance, duplicates (notebooks/eda)
3. Split        stratified train/test (e.g. 80/20, seed 42); test locked until Week 6
4. Preprocess   ColumnTransformer: median/most-frequent impute; one-hot (low-card) /
                ordinal or target-free encoding (high-card); standard-scale numerics
                for linear models; all fitted INSIDE CV folds (Pipeline)
5. Engineer     justified only: approval rates, grade deltas sem2-sem1,
                failed-unit counts, age bands; each with leakage rationale
6. Train        baseline Pipelines: multinomial logistic, random forest,
                HistGradientBoosting (+ optional k-NN/SVM); class_weight options
7. Validate     stratified 5-fold CV on train: macro-F1 (primary), balanced
                accuracy, per-class P/R, confusion matrices, OvR ROC-AUC
8. Optimise     randomised/grid search on key hyperparameters; compare
                class_weight vs. resampling; ablation of feature groups (Week 5)
9. Interpret    coefficients (logistic), importances, permutation importance;
                error slices incl. Enrolled; fairness-qualitative review
10. Report      single final test evaluation; frozen artefacts (git-ignored
                models/) + full report in reports/ (Week 6)
```

**Planning decisions vs. future results:** the structure above is fixed in Week 1; every choice of columns, encoders, models and hyperparameters inside it is *future implementation* to be evidenced in Weeks 2–5. No performance is anticipated numerically.

## 10. Methodology

- **Paradigm:** supervised multiclass classification with an applied, experimental-comparison methodology: baseline-first, then controlled variations (one change at a time), each judged by the Section 16 protocol.
- **Validation design:** stratified split + stratified k-fold CV on train; scaling/encoding/imputation inside folds; seeds fixed and varied for stability checks; holdout test evaluated once in Week 6 to avoid test-set overfitting.
- **Imbalance strategy (planned comparison):** baseline (no correction) vs. `class_weight='balanced'` vs. resampling (e.g. SMOTE applied inside pipeline/fold only, if adopted) — judged on minority (*Enrolled*) recall and macro-F1, not accuracy.
- **Feature methodology:** EDA-driven, leakage-audited engineering; multicollinearity review (correlation/VIF outlook); ablation to test group value; no target leakage (no post-outcome columns, no test-fit transforms).
- **Interpretation:** model-appropriate (coefficients, importances, permutation), plus confusion/error analysis and qualitative fairness review; no causal language.
- **Reproducibility method:** modular `src/` (data, features, models, evaluate), notebooks for narrative only, `requirements.txt`, fixed seeds, run logs; see Section 18.
- **Ethics method:** advisory framing, limitation statements, subgroup error review, no deployment on real students within this project (Sections 7, 17).

## 11. Detailed 30–35 Hour Week 1 Timeline

Total **~32 hours**, single-developer, five working blocks. Only planning artefacts; no data download, no code beyond scaffold review, no experiments.

| Block | Hours | Tasks | Output |
|---|---|---|---|
| B1 — Orientation & brief | 4 | parse brief; fix three-class scope; confirm repo constraints (no commit/push by agent; docs-only) | scope note (this doc, Sections 4–7) |
| B2 — Literature & dataset | 8 | review [1–6] + UCI page [2]; record dataset justification; note *Enrolled*/imbalance implications | Sections 2, 8 |
| B3 — Design | 8 | draft pipeline (Section 9), methodology (Section 10), evaluation protocol (Section 16), deployment sketch (Section 17) | Sections 9–10, 16–17 |
| B4 — Planning & risk | 7 | 6-week roadmap, critical path, resources, risks, success criteria, reproducibility (Sections 11–15, 18–19) | Sections 11–15, 18–19 |
| B5 — Write-up & QA | 5 | draft Sections 1, 3, 20–21 + references; self-review against all 21 required items and rules (no invented results, academic tone) | complete `week1_project_plan.md` + README link |

**Time-box guardrails:** literature capped at 8 h (survey, not systematic review); document length capped to stay maintainable; any spillover defers to Week 2 EDA rather than extending Week 1.

## 12. Milestones and Deliverables

| # | Milestone | Due | Deliverable (in repo) | Acceptance |
|---|---|---|---|---|
| M1 | Scope frozen | W1 | Sections 4–7 of this plan | three classes fixed; ethics framing present |
| M2 | Dataset justified | W1 | Section 8 + `data/` ready (empty, ignored) | UCI source cited; no data committed |
| M3 | Pipeline + methods agreed | W1 | Sections 9–10, 16 | stratified, leakage-safe, imbalance-aware protocol written |
| M4 | Schedule + risks agreed | W1 | Sections 11–15 | 30–35 h plan, critical path, mitigations |
| M5 | Reproducibility + roadmap | W1 | Sections 18–19 + README link | another student could continue from docs |
| M6 | Clean scaffold | W1 | `README.md`, `requirements.txt`, 6 folders, `.gitignore` preserved | `git status` shows only intended files; no commit by agent |
| M7–M12 | (Future) W2–W6 | W2–W6 | EDA, baselines, validation, tuning, final report | per Sections 6, 19 |

Week 1 exit criterion: M1–M6 met and this document reviewed against the 21-item checklist (Section 21 order matches the brief).

## 13. Critical Path and Dependencies

**Critical path (Week 1):** brief → literature/dataset (B2) → pipeline/methods/evaluation (B3) → schedule/risks (B4) → write-up/QA (B5). B2 and B3 are the longest chain (~16 h); delay there directly delays B5.

**Cross-week critical path:** W1 plan → W2 clean, leakage-safe data → W3 runnable baselines → W4 trusted CV numbers → W5 justified candidate → W6 single holdout + report. The W2→W3 handoff is the project bottleneck: no modelling until the loader, schema check and `ColumnTransformer` are stable.

**Dependencies:**
- Sections 9–10 depend on Section 8 (dataset shape constrains encoders/splits).
- Section 16 depends on Sections 4–5 (questions determine metrics: macro-F1 primary, per-class for *Enrolled*).
- Section 17 depends on Sections 7, 16 (advisory deployment only makes sense with honest, limited metrics).
- Week 5 tuning depends on Week 4 error analysis (tune what fails, especially *Enrolled* recall).
- Week 6 report depends on frozen Week 5 candidate + locked test set (no re-tuning on test).

**Float:** reference formatting and README polish can slip within B5; literature depth can be trimmed (guardrail) without breaking the path.

## 14. Software, Libraries, Hardware and Computational Resources

**Software:** Python 3.10+ (planning assumption; version to be recorded in Week 2 environment), Jupyter for narrative EDA, pytest for `tests/`, Git for versioning (no commit/push by agent in Week 1).

**Libraries (already in `requirements.txt`, unpinned Week 1):** `numpy`, `pandas` (ingestion/profiling), `scikit-learn` (Pipeline, ColumnTransformer, models, model_selection, metrics), `matplotlib`, `seaborn` (EDA, confusion matrices, ROC curves), `jupyter`, `pytest`. Possible Week 5 additions only if needed: `imbalanced-learn` (fold-internal resampling) — not installed in Week 1, decision deferred to evidence.

**Hardware:** standard university laptop/desktop, CPU-only; no GPU, no cluster. n ~ 4.4k × ~37 features keeps 5-fold CV of tree ensembles comfortably within minutes on CPU — no special compute booking planned.

**Data/compute governance:** dataset stays in local `data/` (ignored); no cloud upload; no personal data beyond the published file; seeds and package versions logged so runs reproduce on equivalent hardware.

## 15. Risk Management and Mitigation

| Risk | Likelihood / Impact | Mitigation (planned) | Owner / When |
|---|---|---|---|
| Class imbalance; *Enrolled* minority poorly recalled | High / High | stratified splits + CV; macro-F1 primary; per-class P/R + confusion matrices; planned class_weight/resampling comparison (W5); never use accuracy alone | W2–W5 |
| Target leakage via semester features | Med / High | prediction-point audit in W2; exclude/lag post-outcome columns; all transforms inside Pipeline folds; ablation check | W2–W3 |
| Overfitting on small n / over-tuning on test | Med / High | lock test until W6; tune on CV only; report fold variance; single final test run | W3–W6 |
| *Enrolled* temporal ambiguity caps performance | Med / Med | frame as correlational snapshot; error analysis by class; report limits honestly; no causal claims | W4–W6 |
| Categorical cardinality / unseen levels | Med / Med | one-hot low-card, grouped/ordinal high-card; `handle_unknown='ignore'`; schema check in loader | W2 |
| Missingness / data-quality surprises | Med / Med | profile first; pipeline imputation; document assumptions; revisit plan if severe | W2 |
| Scope creep (deep learning, causal, live deploy) | Med / Med | deferred list (Section 3.2); time-boxes; advisory-only deployment framing | W1–W6 |
| Irreproducibility (notebook-only work, seed drift) | Low / High | `src/` modules, seeds, `requirements.txt`, run logs; tests for utilities | W2–W6 |
| Ethics/fairness harm if misused | Low / High | human-in-the-loop, confidence + limits shown, subgroup error review, no autonomous use; see Section 17 | W4–W6 |
| Time overrun (single developer) | Med / Med | 32 h W1 cap; baseline-first; optional models only on float | W1–W6 |

Residual risk accepted: even with mitigation, *Enrolled* recall may remain modest — reported as a limitation, not hidden by metric choice.

## 16. Evaluation and Success Criteria

**Decision rule (planning):** model comparison uses stratified 5-fold CV on the training split. Primary metric: **macro-F1** (equal weight to Dropout, Enrolled, Graduate — directly addresses imbalance). Secondary: **balanced accuracy**, per-class **precision/recall/F1**, **confusion matrices**, and **OvR ROC-AUC** (discrimination outlook). Accuracy is reported for context only and never used for selection.

**Class-imbalance handling in evaluation:** all splits/ folds stratified; *Enrolled* recall tracked explicitly as a first-class criterion; improvements must not collapse Graduate/Dropout precision — trade-offs shown via per-class tables, not a single number. Calibration outlook (predicted-probability reliability) reviewed qualitatively before any advisory-use claim.

**No-leakage contract:** imputers/encoders/scalers fit on training folds only (Pipeline); resampling (if used) inside folds only; test set locked until Week 6 single evaluation; seeds fixed (`42` proposed) with a small seed-sensitivity check in Week 5.

**Week 1 success (this deliverable):** all 21 sections present; no invented results; UCI source cited; three classes and imbalance treatment explicit; planning-vs-results language consistent; repo contains only intended files.

**Weeks 2–6 success gates:** W2 — loader + schema + leakage audit + preprocessing pipeline run on train; W3 — ≥3 baselines with CV tables; W4 — full evaluation + error analysis; W5 — tuned candidate with ablation and stability evidence; W6 — single holdout report with interpretation, limits and ethics reflection.

## 17. Deployment Planning

**Intended use (architectural planning only; no deployment in this project):** a batch, human-in-the-loop advisory aid — e.g., termly risk-score export with predicted class, class probabilities, top contributing features, and an explicit limitations banner — reviewed by academic advisors alongside attendance and pastoral context. No autonomous enrolment, funding or disciplinary decisions.

**Why batch, not real-time:** data arrive termly; decisions are deliberative; batch scoring simplifies governance, audit and fairness review. A REST API is technically feasible (serialised Pipeline + versioned artefact) but explicitly out of scope for six weeks.

**Requirements if ever operationalised (future work, not implemented):** versioned artefact + preprocessing bundle; input schema validation; confidence thresholds with a defer-to-human band (especially for *Enrolled* borderline cases); subgroup monitoring; periodic retraining with drift checks; access controls and data-protection compliance; full audit log. None of these are built in Weeks 1–6; this section records the design envelope so Week 6 limitations are concrete.

## 18. Documentation and Reproducibility Strategy

- **Single source of planning truth:** this file (`reports/week1_project_plan.md`); Week 6 report will reference it and record deviations with reasons.
- **Code vs. narrative split:** reusable logic in `src/` (`data_loader`, `preprocessing`, `features`, `models`, `evaluate` — names proposed, files created from Week 2); notebooks in `notebooks/` narrate and visualise but import from `src/`, never duplicate logic.
- **Environment:** `requirements.txt` (unpinned Week 1; pinned after Week 3 decision); Python version and seed log recorded from Week 2; `pytest` in `tests/` guards loader/preprocessing utilities.
- **Data discipline:** raw UCI file in `data/` only, never committed (`.gitignore`: `data/*.csv/*.xlsx/*.parquet`); processed artefacts regenerated; saved models git-ignored (`*.pkl`, `*.joblib`, `models/**` with `.gitkeep` exception preserved).
- **Experiment discipline:** one variable per comparison; CV tables with mean ± std; confusion matrices and per-class reports committed as figures/tables in `reports/`; final test run once, with frozen artefact hash noted.
- **Language discipline:** "planned/proposed" for future work, "observed/measured" only with dated evidence from Week 2 onward. No performance numbers in Week 1.

## 19. Six-Week Strategic Roadmap

- **Week 1 — Planning (this package):** problem, literature, dataset, pipeline, methods, schedule, risks, evaluation, deployment envelope, reproducibility. Exit: approved plan + clean scaffold.
- **Week 2 — Preprocessing & features:** download UCI file to `data/`; profile (shape, dtypes, missingness, balance, duplicates); stratified split; `ColumnTransformer` pipeline; leakage audit; justified engineered features; EDA notebook + `src/` modules + tests.
- **Week 3 — Implementation:** baseline Pipelines (logistic, forest, boosting; optional contrasts); fixed seeds; training curves/sanity checks; CV smoke test.
- **Week 4 — Evaluation & validation:** full stratified CV with Section 16 metrics; confusion matrices; OvR ROC-AUC; calibration outlook; error analysis incl. *Enrolled* slices and qualitative fairness review.
- **Week 5 — Optimisation:** hyperparameter search; class_weight vs. resampling comparison; feature-group ablation; seed stability; candidate selection with written justification.
- **Week 6 — Report & analysis:** single locked-test evaluation; interpretation; limitations; ethics reflection; comprehensive report + figures in `reports/`; reproducibility checklist; future work.

Go/no-go gates: W2 data quality → W3 baselines run → W4 trustworthy CV → W5 justified candidate → W6 honest holdout. Any gate failure revisits scope (drop optional models first), never evaluation integrity.

## 20. Expected Outcomes, Limitations and Future Work

**Expected outcomes (planned, not claimed):** by Week 6 the project is expected to deliver: (i) a runnable leakage-safe pipeline in `src/`; (ii) stratified-CV comparison of ≥3 baselines with macro-F1-led reporting; (iii) per-class analysis showing where *Enrolled* succeeds/fails; (iv) a tuned candidate with ablation and stability notes; (v) a single holdout evaluation with interpretation; (vi) an academic report recording methods, results, limits and ethics. No numerical targets are set in Week 1 — targets without data would be fabrication.

**Limitations (acknowledged in advance):** single dataset and institution context limit generalisation; n ~ 4.4k limits subgroup inference; *Enrolled* is time-dependent and may bound recall; tabular features omit attendance, engagement and pastoral signals; predictions are correlational, not causal; CPU-only budget restricts model breadth; six weeks restrict tuning depth.

**Future work (beyond six weeks):** multi-institution validation; temporal (cohort-wise) validation; fairness-aware thresholding with stakeholder review; calibration-focused study; batch advisory pilot with human-factors evaluation; drift monitoring and retraining policy. None undertaken here.

## 21. References

[1] V. Realinho, M. Vieira Martins, J. Machado and L. Baptista, "Predict students' dropout and academic success using data mining and machine learning," *Applied Sciences*, 2022. (Study behind the UCI dataset.)
[2] V. Realinho et al., "Predict Students' Dropout and Academic Success," UCI Machine Learning Repository. Dataset page and documentation; donor: Realinho / Vieira Martins / Machado / Baptista. Used as the sole data source for Weeks 2–6. **Not downloaded in Week 1.**
[3] S. A. Becker et al., NMC Horizon Report and learning-analytics literature on early-warning indicators combining academic and contextual features (survey context for Section 2).
[4] Literature on fairness and class imbalance in educational data mining: stratified evaluation, macro-averaged metrics and subgroup error review (basis for Sections 15–16).
[5] Benchmark discussions of the UCI Dropout dataset noting the minority *Enrolled* class and the need for per-class reporting (basis for imbalance treatment).
[6] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, 2011. (`Pipeline`, `ColumnTransformer`, `model_selection`, `metrics` — implementation basis from Week 2.)

> Week 1 integrity statement: this document contains planning decisions and literature-grounded rationale only. It reports no dataset statistics beyond published catalogue description, no EDA, no trained models, and no performance figures. All empirical content is deferred to Weeks 2–6 artefacts.






