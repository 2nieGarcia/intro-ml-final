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

# Small style additions that the theme in .streamlit/config.toml cannot set
STYLE = """
<style>
/* keep text lines a readable length */
.block-container p, .block-container li { max-width: 68ch; }
/* numbers line up when they change */
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
::selection { background: #CFE3DE; color: #1E2B2A; }

/* score bar under the result */
.score-bar { position: relative; height: 0.75rem; margin: 0.25rem 0 0.4rem;
             background: #E3E0D7; border-radius: 999px; }
.score-fill { position: absolute; inset: 0 auto 0 0; border-radius: 999px; }
.score-fill.refer { background: #A86A12; }
.score-fill.clear { background: #2E7D4F; }
.score-threshold { position: absolute; top: -0.3rem; bottom: -0.3rem; width: 2px; background: #1E2B2A; }
.score-labels { position: relative; height: 1.3rem; margin-bottom: 0.75rem; font-size: 0.85rem; color: #5A6664;
                font-variant-numeric: tabular-nums; }
.score-labels span { position: absolute; top: 0; white-space: nowrap; }
.score-labels .threshold-label { transform: translateX(-50%); color: #1E2B2A; }
</style>
"""

st.set_page_config(page_title="Pre-lab CKD screening", page_icon=":material/nephrology:", layout="centered")
st.markdown(STYLE, unsafe_allow_html=True)


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


def score_bar(score, threshold, refer):
    """Shows where the score sits between 0 and 1, with a line at the referral threshold."""
    fill_class = "refer" if refer else "clear"
    st.markdown(
        f'<div class="score-bar" role="img" aria-label="Score {score:.2f}, referral threshold {threshold:.2f}">'
        f'<div class="score-fill {fill_class}" style="width: {score:.0%}"></div>'
        f'<div class="score-threshold" style="left: {threshold:.0%}"></div></div>'
        f'<div class="score-labels"><span style="left: 0">0</span>'
        f'<span class="threshold-label" style="left: {threshold:.0%}">referral threshold {threshold:.2f}</span>'
        f'<span style="right: 0">1</span></div>',
        unsafe_allow_html=True,
    )


def show_result(result, notes):
    if result["refer"]:
        st.warning("**Refer for kidney testing.** Request a serum creatinine (for eGFR) and a urine "
                   "albumin-to-creatinine ratio.", icon=":material/assignment_late:")
    else:
        st.success("**No referral indicated by this screen.** Screen again at the next routine visit, "
                   "or sooner if their health changes.", icon=":material/task_alt:")
    left, right = st.columns(2)
    left.metric("Screening score", f"{result['score']:.2f}")
    right.metric("Similar patients with CKD", f"{result['observed_rate']:.0%}")
    score_bar(result["score"], card["threshold"], result["refer"])
    st.caption("The screening score is not a probability. The second number is the share of patients in the "
               "training data with a similar score who were found to have CKD.")
    for note in notes:
        st.info(note, icon=":material/info:")


st.title("Pre-lab CKD risk screening")
st.write(
    "For adults aged 20 and over who have **never been diagnosed with kidney disease**. It uses only "
    "information a health worker can collect without a laboratory: questions, a scale, a measuring tape "
    "and a blood pressure cuff. It recommends whether the person should be referred for kidney testing."
)
st.caption("This is a screening aid, not a diagnosis. CKD can only be confirmed with laboratory tests.")

with st.expander("About the model", icon=":material/model_training:"):
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

form_tab, csv_tab, help_tab = st.tabs([":material/person: One patient", ":material/upload_file: Upload a CSV",
                                       ":material/straighten: Inputs and units"])

with form_tab:
    with st.form("patient"):
        st.markdown("##### Measurements")
        left, right = st.columns(2)
        age = left.number_input("Age (years)", value=None, step=1, key="age")
        sex = right.selectbox("Sex", ["Female", "Male"], index=None, key="sex")
        weight = left.number_input("Weight (kg)", value=None, step=0.5, key="weight")
        height = right.number_input("Height (cm)", value=None, step=0.5, key="height")
        sbp = left.number_input("Systolic blood pressure (mmHg)", value=None, step=1, key="sbp")
        dbp = right.number_input("Diastolic blood pressure (mmHg)", value=None, step=1, key="dbp")
        waist = left.number_input("Waist circumference (cm), optional", value=None, step=0.5, key="waist")

        st.markdown("##### Health history")
        left, right = st.columns(2)
        told_htn = left.selectbox("Ever told they have high blood pressure?", list(YES_NO_CHOICES), index=None, key="told_htn")
        bp_meds = right.selectbox("Taking blood pressure medicine now?", list(YES_NO_CHOICES), index=None, key="bp_meds")
        diabetes = left.selectbox("Ever told they have diabetes?", list(DIABETES_CHOICES), index=None, key="diabetes")
        any_cvd = right.selectbox("Ever had heart disease, heart failure or stroke?", list(YES_NO_CHOICES), index=None, key="any_cvd")
        high_chol = left.selectbox("Ever told they have high cholesterol?", list(YES_NO_CHOICES), index=None, key="high_chol")
        fam_diabetes = right.selectbox("Parent, brother or sister with diabetes?", list(YES_NO_CHOICES), index=None, key="fam_diabetes")
        smoking = left.selectbox("Smoking", list(SMOKING_CHOICES), index=None, key="smoking")
        st.caption("Questions left empty count as \"Don't know\".")
        submitted = st.form_submit_button("Screen this patient", type="primary", icon=":material/arrow_forward:")

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
            st.error("Please fix the following before screening:\n\n" + "\n".join(f"- {e}" for e in errors),
                     icon=":material/error:")
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
    st.download_button("Download CSV template", template.to_csv(index=False), "screening_template.csv", "text/csv",
                       icon=":material/download:")

    upload = st.file_uploader("Upload a CSV file", type="csv")
    if upload is not None:
        try:
            table = pd.read_csv(upload)
            features, problems = screening.validate_csv(table)
        except Exception as error:
            st.error(f"The file could not be read: {error}", icon=":material/error:")
        else:
            if problems:
                st.error("These rows were skipped:\n\n"
                         + "\n".join(f"- Row {row_number}: " + " ".join(errors) for row_number, errors in problems),
                         icon=":material/error:")
            if len(features):
                results = screening.predict(model, card, features)
                results["recommendation"] = results["refer"].map({True: "refer for kidney testing",
                                                                  False: "no referral"})
                # results first, then what was uploaded, so the answer is visible without scrolling sideways
                inputs = table.copy()
                inputs.index = range(1, len(inputs) + 1)
                output = results[["recommendation", "score", "observed_rate"]].join(inputs, how="right")
                output["recommendation"] = output["recommendation"].fillna("skipped, see the message above")
                output["observed_rate"] = (output["observed_rate"] * 100).round()
                output = output.rename(columns={"observed_rate": "similar_with_ckd_percent"})

                screened, referred = len(results), int(results["refer"].sum())
                left, middle, right = st.columns(3)
                left.metric("Rows screened", f"{screened} of {len(table)}")
                middle.metric("Refer for testing", referred)
                right.metric("No referral", screened - referred)
                st.dataframe(output, column_config={
                    "recommendation": st.column_config.TextColumn("Recommendation"),
                    "score": st.column_config.NumberColumn("Score", format="%.2f"),
                    "similar_with_ckd_percent": st.column_config.NumberColumn("Similar patients with CKD",
                                                                              format="%.0f%%"),
                })
                st.download_button("Download results", output.to_csv(index_label="row"), "screening_results.csv",
                                   "text/csv", icon=":material/download:")

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
