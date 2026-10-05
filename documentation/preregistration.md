# Pre-registration

Written: 2 October 2026

I wrote this document before downloading, exploring, or modeling any data.
I list any change made after this commit in Section 8, with the date and the reason.

## 1. Research questions

1. Can a model that only uses information available before any lab test identify adults
   with undiagnosed CKD well enough to help decide who should be referred for kidney testing?
2. How much of the accuracy reported for CKD models on the UCI CKD dataset comes from
   features that are part of the diagnosis itself?

## 2. Reference standard for CKD

I follow the KDIGO definition: decreased kidney function (GFR below 60 mL/min/1.73m2)
or markers of kidney damage such as albuminuria (albumin-to-creatinine ratio of 30 mg/g
or more), present for more than 3 months [1].

NHANES label:

- CKD = 1 if eGFR < 60 or urine albumin-to-creatinine ratio (URDACT) >= 30 mg/g, otherwise 0.
- eGFR is calculated from serum creatinine (LBXSCR) with the CKD-EPI 2021 equation,
  which does not use race [2].
- NHANES measures each person only once, so the 3-month persistence rule cannot be checked.
  Some temporary albuminuria will be labeled as CKD. This is a limitation of the label.

The lab values above are used only to build the label. They are never model inputs.

## 3. Feature tiers

Rules used to assign each feature:

- Tier 1, definitional: a KDIGO criterion, or a direct lab measure of kidney filtration
  or urine abnormality used to make the diagnosis.
- Tier 2, consequential: changes because kidney disease is already present.
- Tier 3, antecedent: a risk factor or characteristic that can be collected at a rural
  health unit without a lab, using questions, a scale, a tape measure, and a blood
  pressure cuff.
- Other laboratory: a lab value that is neither definitional nor a clear consequence.
  It is still not available before a lab test, so it is removed before the Tier 3 only model.

### UCI CKD dataset (24 features)

| Feature | Description | Tier | Reason |
|---|---|---|---|
| sc | serum creatinine | 1 | used to calculate eGFR, the main KDIGO criterion |
| bu | blood urea | 1 | waste product that builds up when filtration drops |
| al | urine albumin (dipstick grade) | 1 | albuminuria is a KDIGO criterion |
| sg | urine specific gravity | 1 | measures the kidney's ability to concentrate urine |
| rbc | red blood cells in urine | 1 | urine sediment abnormality, listed by KDIGO as a marker of kidney damage |
| hemo | hemoglobin | 2 | CKD reduces erythropoietin, which causes anemia |
| pcv | packed cell volume | 2 | same reason as hemoglobin |
| rbcc | red blood cell count | 2 | same reason as hemoglobin |
| ane | anemia | 2 | same reason as hemoglobin |
| pot | potassium | 2 | builds up when the kidneys cannot excrete it |
| sod | sodium | 2 | sodium and water balance is disturbed in advanced CKD |
| pe | pedal edema | 2 | fluid retention from reduced kidney function |
| appet | appetite | 2 | poor appetite is a symptom of uremia |
| su | urine sugar | Other lab | mainly reflects high blood glucose |
| bgr | random blood glucose | Other lab | reflects diabetes, not kidney function |
| wbcc | white blood cell count | Other lab | not specific to kidney disease |
| pc | pus cells in urine | Other lab | mainly a sign of urinary tract infection |
| pcc | pus cell clumps in urine | Other lab | mainly a sign of urinary tract infection |
| ba | bacteria in urine | Other lab | mainly a sign of urinary tract infection |
| age | age in years | 3 | known without any test |
| bp | blood pressure | 3 | measured with a cuff |
| htn | hypertension | 3 | see note below |
| dm | diabetes | 3 | a leading cause of CKD |
| cad | coronary artery disease | 3 | shares risk factors with CKD, collected by history |

Note on hypertension: it is both a cause and a consequence of CKD. I keep it in Tier 3
because it can be measured with a cuff and screening guidelines already use it to decide
who should be tested.

Note on UCI Tier 3: UCI does not include sex, BMI, or smoking, so its Tier 3 set is
smaller than the NHANES Tier 3 set.

### NHANES 2017-March 2020 (Tier 3 only)

| Feature | NHANES variable(s) | File |
|---|---|---|
| Age | RIDAGEYR (top-coded at 80) | P_DEMO |
| Sex | RIAGENDR | P_DEMO |
| BMI | BMXBMI | P_BMX |
| Waist circumference | BMXWAIST | P_BMX |
| Systolic and diastolic BP | mean of BPXOSY1-3 and BPXODI1-3 | P_BPXO |
| Told has hypertension | BPQ020 | P_BPQ |
| Taking BP medication | BPQ050A | P_BPQ |
| Told has high cholesterol | BPQ080 | P_BPQ |
| Diabetes (yes / no / borderline) | DIQ010 | P_DIQ |
| Smoking (never / former / current) | SMQ020, SMQ040 | P_SMQ |
| Any cardiovascular disease | MCQ160B, MCQ160C, MCQ160D, MCQ160E, MCQ160F | P_MCQ |
| Family history of diabetes | MCQ300C | P_MCQ |

## 4. Who is included (NHANES)

1. Age 20 or older.
2. Examined at the mobile examination center (RIDSTATR = 2).
3. Not pregnant (RIDEXPRG not equal to 1), because pregnancy changes both eGFR and albuminuria.
4. Has both serum creatinine and urine albumin-to-creatinine ratio.
5. Has not been told they have weak or failing kidneys (KIQ022 not equal to 1) and has
   not had dialysis in the past 12 months (KIQ025 not equal to 1). The tool is meant for
   people who have never been diagnosed.

I will compare the people dropped by rule 4 with the people kept, to check whether the
missing labels are related to age, diabetes, or hypertension.

## 5. Never used as model inputs

| Variable | Reason |
|---|---|
| LBXSCR, URDACT, URXUMA, URXUCR | used to build the label |
| Any other lab value (biochemistry, HbA1c, glucose, blood count) | not available before a lab test |
| KIQ022, KIQ025 | these are the diagnosis itself, reported by the participant |
| RIDRETH3 (race and ethnicity) | the 2021 equation removed race, and at a Philippine health unit this feature would carry no information. Used only to report performance by subgroup. |

## 6. Analysis plan

- Split: 80 percent training, 20 percent test, stratified on the CKD label, random_state = 42.
  The test set is not looked at until the final model is chosen.
- Cross-validation: StratifiedKFold with 5 folds, shuffle = True, random_state = 42.
  The same folds are used for all three models.
- Preprocessing: every step (imputation, encoding, scaling) is inside a scikit-learn Pipeline
  and is fit only on the training folds. The same pipeline is used for all three models.
  The exact imputation choices are decided during EDA and explained in the EDA notebook.
- Models: Logistic Regression, Random Forest, and HistGradientBoostingClassifier,
  all with class_weight = "balanced".
- Primary metric: F2 score for the CKD class (beta = 2). A missed CKD case is more costly
  than an extra confirmatory blood test, so recall is weighted more than precision.
  Beta stays at 2 for the whole project.
- Decision threshold: treated as a hyperparameter and tuned to maximize F2 using
  cross-validation on the training data only.
- Supporting metrics, reported for all three models: accuracy, precision, recall,
  specificity, F1, confusion matrix, ROC-AUC, PR-AUC, calibration curve, and Brier score.
  Cross-validation results are reported as mean and standard deviation across folds.
- Baselines (not counted as one of the three models):
  1. Refer everyone.
  2. Rule based on common screening risk factors: refer if diabetes, hypertension,
     cardiovascular disease, or age 60 or older.
- Model selection: the model with the highest mean cross-validation F2 is selected.
  If the top two models differ by less than one standard deviation of the better model's
  fold scores, the simpler model is selected. Order from simplest: Logistic Regression,
  Random Forest, Gradient Boosting.
- Final test: the selected model is evaluated once on the test set.
- Subgroup check: test performance is also reported for the Non-Hispanic Asian subgroup
  as the closest available proxy for an Asian population. This is not the same as a
  Filipino population.
- UCI ablation: the same three models and the same procedure are run on four feature sets:
  all features; without Tier 1; without Tier 1 and Tier 2; Tier 3 only.

## 7. Expected results

- H1: On UCI with all features, the models will reach accuracy above 95 percent,
  similar to published results.
- H2: On UCI, performance will drop clearly once Tier 1 and Tier 2 are removed.
- H3: On NHANES, the Tier 3 models will beat the refer-everyone baseline on F2.
  Whether they beat the risk-factor rule is the main open question.

I will report the results whether or not these expectations hold.

## 8. Changes after this commit

No change was made to the research questions, label, tiers, features, inclusion rules, split,
metric, models, threshold tuning, or selection rule. Decisions that the plan left open, and
data problems found later, are recorded here.

- 2 October 2026, NHANES EDA: imputation is median (numeric) and most common answer
  (yes/no and categorical), with no "was missing" indicator columns. Reason: missing waist and
  BMI are linked to CKD in NHANES for a reason that would not apply in the app
  (`notebooks/01_nhanes_eda.ipynb`, section 13).
- 5 October 2026, hyperparameter grids: the grids were not part of the pre-registration. They are
  fixed in `src/modeling.make_models` and are the same for both datasets.
- 5 October 2026, UCI cleaning: three impossible lab values (one sodium of 4.5 mEq/L and two
  potassium values of 39 and 47 mEq/L) were set to missing (`notebooks/02_uci_ablation.ipynb`,
  section 2.1).
- 5 October 2026, app output: because `class_weight="balanced"` makes raw scores higher than the
  observed risk, the app shows the observed CKD rate among training patients with similar scores
  instead of the raw score as a percentage. This does not change the model or its evaluation.

## References

[1] Kidney Disease: Improving Global Outcomes (KDIGO) CKD Work Group, "KDIGO 2024 clinical
    practice guideline for the evaluation and management of chronic kidney disease,"
    Kidney Int., vol. 105, no. 4S, pp. S117-S314, 2024.
[2] L. A. Inker et al., "New creatinine- and cystatin C-based equations to estimate GFR
    without race," N. Engl. J. Med., vol. 385, no. 19, pp. 1737-1749, 2021.
