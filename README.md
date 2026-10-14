# Pre-lab CKD Early Detection using Random Forest

Final project for my Machine Learning course. The goal is a screening model that estimates a person's risk of chronic kidney disease (CKD) using only information available before any lab test, and recommends whether they should be referred for kidney function testing.

## Background

CKD is underdiagnosed in the Philippines, and many patients only find out when the disease is already at end stage. A lot of published machine learning models for CKD report 99 to 100 percent accuracy, but most of them are trained on the UCI CKD dataset and use features like serum creatinine, blood urea, urine albumin and specific gravity. Those are the same measurements used to diagnose CKD, so the model is mostly rebuilding the diagnosis instead of predicting it early. At a rural health unit, for a patient who has never been tested, those values do not exist yet.

This project only uses information a health worker can collect without a lab: age, sex, blood pressure, BMI, waist circumference, diabetes and hypertension history, smoking, and heart disease history. The output is a risk score and a referral recommendation. It does not diagnose CKD.

## Approach

**Main dataset: NHANES 2017 to March 2020 (pre-pandemic), US CDC.**
CKD is labeled from lab results: eGFR below 60 mL/min/1.73m2 (CKD-EPI 2021 equation, without race) or urine albumin-to-creatinine ratio of 30 mg/g or higher. These lab values are only used to create the label. They are never given to the model as inputs.

**Second dataset: UCI Chronic Kidney Disease (400 patients).**
Used to show where the high accuracy in the literature comes from. I reproduce the reported result, then remove features in groups and measure how much performance drops.

Feature groups:

| Tier | Meaning | Examples |
|---|---|---|
| 1. Definitional | Part of how CKD is diagnosed | serum creatinine, blood urea, albumin, specific gravity |
| 2. Consequential | Effects of kidney disease that is already present | hemoglobin, packed cell volume, red blood cell count, pedal edema, appetite, anemia |
| 3. Antecedent | Known before any kidney test | age, sex, hypertension, diabetes, BMI, smoking, cardiovascular disease |

I wrote down and justified the tier assignments in `documentation/preregistration.md` before doing any modeling.

**Models compared:** Logistic Regression, Random Forest, and Gradient Boosting (scikit-learn). All three use the same train/test split, preprocessing pipeline, cross-validation folds and metric.

**Primary metric:** F2 score for the CKD class. Missing a CKD patient is much worse than sending a healthy person for one extra blood test, so recall is weighted more than precision.

## Results

| | F2 (primary) | Recall | Specificity | ROC-AUC |
|---|---|---|---|---|
| Random Forest, 5-fold CV (mean) | 0.580 | 0.819 | 0.554 | 0.770 |
| Random Forest, test set (run once) | 0.586 | 0.864 | 0.504 | 0.773 |
| Risk-factor rule, test set | 0.583 | 0.852 | 0.515 | - |
| Refer everyone, test set | 0.497 | 1.000 | 0.000 | - |

- All three models reached almost the same cross-validation F2 (0.578 to 0.580). Random Forest was selected by the pre-registered tie-break rule.
- With only pre-lab information, the model performs about the same as a simple rule (refer if diabetes, hypertension, heart disease or age 60+). On F2 they are tied (test difference 0.003, 95% CI -0.017 to 0.024).
- Its advantage is ranking. In a post-hoc analysis added after peer review (notebook 03, section 11), when only 300 of the 1,473 test participants could be tested, testing the highest scores first found 122 CKD cases, against 107 for a rule that counts the four risk factors (difference 15.2, 95% CI 3.9 to 25.3).
- On UCI, the same three algorithms reach about 99% accuracy with all features, but with Tier 3 features only, F2 is no better than referring everyone (0.89 to 0.90) and specificity falls to 0.21 to 0.44.

Full results are in the notebooks and in `paper/`.

## Folder structure

```
app/              Streamlit app (streamlit_app.py)
.streamlit/       App theme (config.toml)
data/original/    Original NHANES and UCI files, with SOURCE.md
data/processed/   nhanes_cohort.csv (cohort, label, features, train/test split)
notebooks/        01 NHANES EDA, 02 UCI ablation, 03 NHANES modeling
src/              Reusable code: data download, cohort and label, modeling, input validation
models/           Saved pipeline, model card and result tables
tests/            Automated tests (pytest)
documentation/    Pre-registration, data dictionary, technical documentation, figures, app screenshots
paper/            IMRaD paper (DOCX and PDF)
requirements.txt  Packages needed to run the app (pinned)
requirements-dev.txt  Extra packages for tests and rebuilding notebooks and documents
```

## Setup

Requires Python 3.12.

Windows (PowerShell):

```
git clone https://github.com/2nieGarcia/intro-ml-final.git
cd intro-ml-final
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

If activation fails with "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

macOS / Linux:

```
git clone https://github.com/2nieGarcia/intro-ml-final.git
cd intro-ml-final
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Use `requirements.txt` instead if you only need the app.

## Running

| Task | Command (from the project root) |
|---|---|
| Run the app | `streamlit run app/streamlit_app.py` |
| Run the tests | `python -m pytest` |
| Download the original data again | `python src/download_data.py` |
| Reproduce the results | run the notebooks in order: 01, 02, 03 |

Deployed app: https://intro-ml-final-k5xt73im3xwcggoc8x3mws.streamlit.app/

## Using the app

Enter one patient in the form, or upload a CSV for several (a template can be downloaded in the app). For a CSV, you can enter the number of tests available, and only that many patients are referred, highest scores first. The app shows a referral recommendation, the screening score against the referral threshold, and the share of training patients with a similar score who had CKD. The "Inputs and units" tab lists every field and its accepted range. Invalid or impossible inputs are rejected with a message instead of a prediction.

The full input format, application guide and troubleshooting notes are in `documentation/technical_documentation.md`.

## Limitations

- Trained on US adults (NHANES). Not validated on Filipino patients.
- The CKD label comes from a single lab measurement; the clinical definition needs 3 months.
- The model misses many younger adults with CKD and many albuminuria-only cases. A "no referral" result does not rule out CKD.
- It is a screening aid only. It does not diagnose CKD.

## Data sources

- National Center for Health Statistics, National Health and Nutrition Examination Survey (NHANES) 2017-March 2020 Pre-Pandemic Data. https://wwwn.cdc.gov/nchs/nhanes/ (public domain, US government work)
- L. Rubini, P. Soundarapandian, and P. Eswaran, "Chronic Kidney Disease," UCI Machine Learning Repository, 2015. https://doi.org/10.24432/C5G020 (CC BY 4.0)
