# Pre-lab CKD Risk Screening

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

## Folder structure

```
app/              Streamlit app
data/original/    Original datasets and source notes
data/processed/   Cleaned data
notebooks/        EDA and model development notebooks
src/              Reusable preprocessing and modeling code
models/           Saved final model pipeline
documentation/    Technical documentation and screenshots
paper/            Final paper (DOCX and PDF)
requirements.txt  Python dependencies
```

## Setup

Requires Python 3.12.

Windows (PowerShell):

```
git clone https://github.com/2nieGarcia/intro-ml-final.git
cd intro-ml-final
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If activation fails with "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

macOS / Linux:

```
git clone https://github.com/2nieGarcia/intro-ml-final.git
cd intro-ml-final
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status

Work in progress. Data download, EDA, modeling and the Streamlit app are not done yet. This README will be updated as each part is finished.

## Data sources

- National Center for Health Statistics, National Health and Nutrition Examination Survey (NHANES) 2017-March 2020 Pre-Pandemic Data. https://wwwn.cdc.gov/nchs/nhanes/ (public domain, US government work)
- L. Rubini, P. Soundarapandian, and P. Eswaran, "Chronic Kidney Disease," UCI Machine Learning Repository, 2015. https://doi.org/10.24432/C5G020 (CC BY 4.0)
