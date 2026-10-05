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

## data/original/uci/chronic_kidney_disease.csv

The UCI Chronic Kidney Disease dataset, kept as downloaded. One row per hospital patient.
400 rows, 24 features and the label. Units are the ones given on the UCI dataset page.
It is only used for the tier ablation in `notebooks/02_uci_ablation.ipynb`. The cleaning is
done in memory by `src/uci.py`, so no processed copy of this file is saved.

| Column | Type | Unit or values | Meaning | Tier | Missing |
|---|---|---|---|---|---|
| age | number | years | Age | 3 | 9 |
| bp | number | mmHg | Blood pressure | 3 | 12 |
| sg | number | 1.005, 1.010, 1.015, 1.020, 1.025 | Urine specific gravity | 1 | 47 |
| al | number | 0 to 5 | Urine albumin (dipstick grade) | 1 | 46 |
| su | number | 0 to 5 | Urine sugar (dipstick grade) | Other lab | 49 |
| rbc | text | normal / abnormal | Red blood cells in urine | 1 | 152 |
| pc | text | normal / abnormal | Pus cells in urine | Other lab | 65 |
| pcc | text | present / notpresent | Pus cell clumps in urine | Other lab | 4 |
| ba | text | present / notpresent | Bacteria in urine | Other lab | 4 |
| bgr | number | mg/dL | Random blood glucose | Other lab | 44 |
| bu | number | mg/dL | Blood urea | 1 | 19 |
| sc | number | mg/dL | Serum creatinine | 1 | 17 |
| sod | number | mEq/L | Sodium | 2 | 87 |
| pot | number | mEq/L | Potassium | 2 | 88 |
| hemo | number | g/dL | Hemoglobin | 2 | 52 |
| pcv | number | % | Packed cell volume | 2 | 71 |
| wbcc | number | cells/cumm | White blood cell count | Other lab | 106 |
| rbcc | number | millions/cmm | Red blood cell count | 2 | 131 |
| htn | text | yes / no | Hypertension | 3 | 2 |
| dm | text | yes / no | Diabetes | 3 | 2 |
| cad | text | yes / no | Coronary artery disease | 3 | 2 |
| appet | text | good / poor | Appetite | 2 | 1 |
| pe | text | yes / no | Pedal edema | 2 | 1 |
| ane | text | yes / no | Anemia | 2 | 1 |
| class | text | ckd / notckd | **Target.** 250 ckd, 150 notckd | | 0 |

The reason for each tier is in Section 3 of `documentation/preregistration.md`.

### Processing notes

1. Stray tab characters were removed from text values (for example `ckd\t` and `\tno`).
2. Three impossible lab values were set to missing: one sodium of 4.5 mEq/L and two potassium
   values of 39 and 47 mEq/L. Allowed ranges are in `IMPOSSIBLE` in `src/uci.py`.
3. `class` was turned into a 0/1 column `ckd` (1 = ckd).
4. Other missing values are filled inside the model pipeline, fit on training folds only, with
   the same preprocessing as the NHANES models.
