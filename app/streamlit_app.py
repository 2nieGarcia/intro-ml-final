"""Pre-lab CKD screening app.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

import screening  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "ckd_screening_pipeline.joblib"
CARD_PATH = PROJECT_ROOT / "models" / "model_card.json"

YES_NO_CHOICES = {"Yes": "yes", "No": "no", "Don't know": "dont_know"}
DIABETES_CHOICES = {"No": "no", "Borderline / prediabetes": "borderline", "Yes": "yes", "Don't know": "dont_know"}
SMOKING_CHOICES = {"Never (fewer than 100 cigarettes in life)": "never", "Former": "former",
                   "Current": "current", "Don't know": "dont_know"}

st.set_page_config(page_title="Pre-lab CKD screening", layout="centered")


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    card = json.loads(CARD_PATH.read_text())
    return model, card


try:
    model, card = load_model()
except Exception as error:  # the app cannot do anything without the model, so stop here with a clear message
    st.error(f"The model files could not be loaded from the models/ folder ({error}).")
    st.stop()


def show_result(result, notes):
    if result["refer"]:
        st.warning("**Refer for kidney testing.** Request a serum creatinine (for eGFR) and a urine "
                   "albumin-to-creatinine ratio.")
    else:
        st.success("**No referral indicated by this screen.** Screen again at the next routine visit, "
                   "or sooner if their health changes.")
    left, right = st.columns(2)
    left.metric("Screening score", f"{result['score']:.2f}", f"referral threshold {card['threshold']:.2f}",
                delta_color="off")
    right.metric("Similar patients with CKD", f"{result['observed_rate']:.0%}")
    st.caption("The screening score is not a probability. The second number is the share of patients in the "
               "training data with a similar score who were found to have CKD.")
    for note in notes:
        st.info(note)


st.title("Pre-lab CKD risk screening")
st.write(
    "For adults aged 20 and over who have **never been diagnosed with kidney disease**. It uses only "
    "information a health worker can collect without a laboratory: questions, a scale, a measuring tape "
    "and a blood pressure cuff. It recommends whether the person should be referred for kidney testing."
)
st.caption("This is a screening aid, not a diagnosis. CKD can only be confirmed with laboratory tests.")

with st.expander("About the model"):
    cv, test = card["cv_mean"], card["test"]
    st.markdown(
        f"- **Model:** {card['model']} with a referral threshold of {card['threshold']:.2f}\n"
        f"- **Trained on:** {card['training_rows']:,} US adults from NHANES 2017 to March 2020, "
        f"{card['training_prevalence']:.1%} of whom had CKD\n"
        f"- **CKD definition used for training:** eGFR below 60 (CKD-EPI 2021) or urine "
        f"albumin-to-creatinine ratio of 30 mg/g or more, from a single measurement\n"
        f"- **Cross-validation:** recall {cv['recall']:.2f}, specificity {cv['specificity']:.2f}, "
        f"F2 {cv['f2']:.2f}, ROC-AUC {cv['roc_auc']:.2f}\n"
        f"- **Test set:** recall {test['recall']:.2f}, specificity {test['specificity']:.2f}, "
        f"F2 {test['f2']:.2f}, ROC-AUC {test['roc_auc']:.2f}\n"
        "- **Limitations:** trained on a US population, not validated on Filipino patients. "
        "It will miss some patients, especially younger adults with reduced kidney function. "
        "A 'no referral' result does not rule out CKD."
    )

form_tab, csv_tab, help_tab = st.tabs(["One patient", "Upload a CSV", "Inputs and units"])

with form_tab:
    with st.form("patient"):
        left, right = st.columns(2)
        age = left.number_input("Age (years)", value=None, step=1, key="age")
        sex = right.selectbox("Sex", ["Female", "Male"], index=None, key="sex")
        weight = left.number_input("Weight (kg)", value=None, step=0.5, key="weight")
        height = right.number_input("Height (cm)", value=None, step=0.5, key="height")
        sbp = left.number_input("Systolic blood pressure (mmHg)", value=None, step=1, key="sbp")
        dbp = right.number_input("Diastolic blood pressure (mmHg)", value=None, step=1, key="dbp")
        waist = left.number_input("Waist circumference (cm), optional", value=None, step=0.5, key="waist")
        st.divider()
        left, right = st.columns(2)
        told_htn = left.selectbox("Ever told they have high blood pressure?", list(YES_NO_CHOICES), index=None, key="told_htn")
        bp_meds = right.selectbox("Taking blood pressure medicine now?", list(YES_NO_CHOICES), index=None, key="bp_meds")
        diabetes = left.selectbox("Ever told they have diabetes?", list(DIABETES_CHOICES), index=None, key="diabetes")
        any_cvd = right.selectbox("Ever had heart disease, heart failure or stroke?", list(YES_NO_CHOICES), index=None, key="any_cvd")
        high_chol = left.selectbox("Ever told they have high cholesterol?", list(YES_NO_CHOICES), index=None, key="high_chol")
        fam_diabetes = right.selectbox("Parent, brother or sister with diabetes?", list(YES_NO_CHOICES), index=None, key="fam_diabetes")
        smoking = left.selectbox("Smoking", list(SMOKING_CHOICES), index=None, key="smoking")
        submitted = st.form_submit_button("Screen this patient", type="primary")

    if submitted:
        record = {
            "age": age, "sex": sex.lower() if sex else None, "weight_kg": weight, "height_cm": height,
            "waist_cm": waist, "sbp": sbp, "dbp": dbp,
            "told_htn": YES_NO_CHOICES.get(told_htn), "bp_meds": YES_NO_CHOICES.get(bp_meds),
            "high_chol": YES_NO_CHOICES.get(high_chol), "any_cvd": YES_NO_CHOICES.get(any_cvd),
            "fam_diabetes": YES_NO_CHOICES.get(fam_diabetes), "diabetes": DIABETES_CHOICES.get(diabetes),
            "smoking": SMOKING_CHOICES.get(smoking),
        }
        features, errors, notes = screening.validate_patient(record)
        if errors:
            st.error("Please fix the following before screening:\n\n" + "\n".join(f"- {e}" for e in errors))
        else:
            result = screening.predict(model, card, features).iloc[0]
            show_result(result, notes)

with csv_tab:
    st.write("Screen several patients at once. Use the template so the column names and answers match.")
    template = pd.DataFrame([
        {"age": 58, "sex": "female", "weight_kg": 72, "height_cm": 155, "waist_cm": 95, "sbp": 150, "dbp": 88,
         "told_htn": "yes", "bp_meds": "yes", "high_chol": "no", "any_cvd": "no", "fam_diabetes": "yes",
         "diabetes": "yes", "smoking": "never"},
        {"age": 34, "sex": "male", "weight_kg": 68, "height_cm": 170, "waist_cm": "", "sbp": 118, "dbp": 76,
         "told_htn": "no", "bp_meds": "no", "high_chol": "dont_know", "any_cvd": "no", "fam_diabetes": "no",
         "diabetes": "no", "smoking": "current"},
    ])[screening.INPUT_COLUMNS]
    st.download_button("Download CSV template", template.to_csv(index=False), "screening_template.csv", "text/csv")

    upload = st.file_uploader("Upload a CSV file", type="csv")
    if upload is not None:
        try:
            table = pd.read_csv(upload)
            features, problems = screening.validate_csv(table)
        except Exception as error:
            st.error(f"The file could not be read: {error}")
        else:
            for row_number, errors in problems:
                st.error(f"Row {row_number} was skipped: " + " ".join(errors))
            if len(features):
                results = screening.predict(model, card, features)
                output = table.copy()
                output.index = range(1, len(output) + 1)
                output = output.join(results)
                output["recommendation"] = output["refer"].map({True: "refer for kidney testing", False: "no referral"})
                st.write(f"Screened {len(results)} of {len(table)} rows. {int(results['refer'].sum())} recommended for referral.")
                st.dataframe(output)
                st.download_button("Download results", output.to_csv(index_label="row"), "screening_results.csv", "text/csv")

with help_tab:
    st.markdown("""
| Input | Unit | Accepted range | Notes |
|---|---|---|---|
| Age | years | 20 to 120 | Ages above 80 are treated as 80, the oldest age in the training data |
| Sex | | female / male | |
| Weight | kg | 20 to 300 | Used with height to calculate BMI |
| Height | cm | 100 to 230 | |
| Systolic BP | mmHg | 60 to 260 | Use the average if several readings were taken |
| Diastolic BP | mmHg | 30 to 150 | Must be lower than systolic |
| Waist | cm | 40 to 200 | Optional. Measured at the top of the hip bone |
| Yes / no questions | | yes / no / don't know | "Don't know" is allowed for every question |
| Diabetes | | no / borderline / yes / don't know | Borderline includes prediabetes |
| Smoking | | never / former / current / don't know | Never means fewer than 100 cigarettes in their life |

**What the output means**

- **Refer for kidney testing:** the score is at or above the threshold chosen during training. The person should get a serum creatinine and a urine albumin-to-creatinine ratio.
- **No referral indicated:** the score is below the threshold. This does not rule out CKD.
- **Screening score:** the model's raw output between 0 and 1. It is higher than the true risk because the model was trained to pay extra attention to CKD cases, so it should not be read as a percentage.
- **Similar patients with CKD:** among people in the training data with a similar score, the share who had CKD.
""")
