# Data dictionary

## data/processed/nhanes_cohort.csv

Created by `notebooks/01_nhanes_eda.ipynb` from the original files in `data/original/nhanes/`,
using the functions in `src/nhanes.py`. One row per participant. 7,363 rows.

### Identifiers and split

| Column | Type | Meaning |
|---|---|---|
| SEQN | integer | NHANES participant ID |
| split | text | `train` (5,890 rows) or `test` (1,473 rows). 80/20, stratified on `ckd`, random_state 42. |
| RIDRETH3 | number | Race and ethnicity. 1 Mexican American, 2 Other Hispanic, 3 Non-Hispanic White, 4 Non-Hispanic Black, 6 Non-Hispanic Asian, 7 Other or multiracial. Used only for the subgroup check, never as a model input. |

### Label columns (never model inputs)

| Column | Type | Unit | Meaning |
|---|---|---|---|
| egfr | number | mL/min/1.73m2 | eGFR from serum creatinine, age and sex, CKD-EPI 2021 equation without race |
| URDACT | number | mg/g | Urine albumin-to-creatinine ratio, as published by NHANES |
| low_egfr | 0 / 1 | | 1 if egfr < 60 |
| albuminuria | 0 / 1 | | 1 if URDACT >= 30 |
| ckd | 0 / 1 | | **Target.** 1 if low_egfr = 1 or albuminuria = 1 |

### Model features (Tier 3)

| Column | Type | Unit or coding | Meaning | NHANES source |
|---|---|---|---|---|
| age | number | years | Age at screening. 80 means 80 or older. | RIDAGEYR |
| bmi | number | kg/m2 | Body mass index | BMXBMI |
| waist | number | cm | Waist circumference | BMXWAIST |
| sbp | number | mmHg | Systolic blood pressure, mean of up to 3 readings | BPXOSY1-3 |
| dbp | number | mmHg | Diastolic blood pressure, mean of up to 3 readings | BPXODI1-3 |
| female | 0 / 1 | 1 = female | Sex | RIAGENDR |
| told_htn | 0 / 1 | 1 = yes | Ever told by a doctor they have high blood pressure | BPQ020 |
| bp_meds | 0 / 1 | 1 = yes | Currently taking prescribed blood pressure medicine | BPQ040A, BPQ050A |
| high_chol | 0 / 1 | 1 = yes | Ever told by a doctor they have high cholesterol | BPQ080 |
| any_cvd | 0 / 1 | 1 = yes | Ever told they had heart failure, coronary heart disease, angina, heart attack or stroke | MCQ160B-F |
| fam_diabetes | 0 / 1 | 1 = yes | A close relative has diabetes | MCQ300C |
| diabetes | text | no / borderline / yes | Ever told by a doctor they have diabetes | DIQ010 |
| smoking | text | never / former / current | Never means under 100 cigarettes in their life | SMQ020, SMQ040 |

An empty cell means the value is missing. Missing values are not filled in this file. They are
filled in inside the model pipeline so the fill values are learned from training data only.

### Processing notes

1. The 10 NHANES files were joined on SEQN onto the demographics file.
2. Answer codes 7 (refused) and 9 (don't know) were changed to missing.
3. Inclusion rules, in order: age 20 or older; examined at the mobile exam center; not pregnant;
   has both serum creatinine and urine ACR; not told of weak or failing kidneys and no dialysis.
   15,560 participants became 7,363.
4. Skip patterns: a blank "smokes now" for someone who never smoked 100 cigarettes is coded as a
   never smoker, and a blank "taking BP medicine" for someone never told of high blood pressure or
   never told to take medicine is coded as 0. These are known answers, not missing values.
5. `any_cvd` is 1 if any of the five heart and stroke questions is yes, 0 if all five are no,
   and missing otherwise.
6. Blood pressure is the mean of whichever of the three readings were taken.
7. No rows were removed as outliers. All numeric values are inside the plausible ranges in
   `PLAUSIBLE_RANGES` in `src/nhanes.py`.

The reasons for each step are in `notebooks/01_nhanes_eda.ipynb` and `documentation/preregistration.md`.
