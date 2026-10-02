# Original data sources

The files in this folder are kept exactly as downloaded. I do not edit them.
To download them again, run `python src/download_data.py` from the project root.

Downloaded: 2 October 2026

## NHANES 2017-March 2020 Pre-Pandemic (main dataset)

- Publisher: National Center for Health Statistics (NCHS), Centers for Disease Control and Prevention (CDC)
- Survey page: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2017-2020
- File location: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/
- License: public domain (work of the US federal government). NCHS asks that the data be cited.
- Collection method: national survey of the US civilian non-institutionalized population.
  Questionnaires are done by interview at home, and body measures, blood pressure, blood and
  urine samples are collected at a mobile examination center.
- Unit of analysis: one survey participant. Files are linked with the participant ID `SEQN`.
- Format: SAS transport (.xpt). Read with `pandas.read_sas(path, format="xport")`.
- Note from CDC: the 2019-2020 data collection stopped early because of COVID-19, so CDC combined
  it with 2017-2018 into one pre-pandemic file set. These files should be analyzed together,
  not split back into the two cycles.

| File | Contents | Rows | Use in this project |
|---|---|---|---|
| nhanes/P_DEMO.xpt | demographics | 15,560 | features and inclusion rules |
| nhanes/P_BMX.xpt | body measures | 14,300 | features |
| nhanes/P_BPXO.xpt | blood pressure, oscillometric | 11,656 | features |
| nhanes/P_BPQ.xpt | blood pressure and cholesterol questionnaire | 10,195 | features |
| nhanes/P_DIQ.xpt | diabetes questionnaire | 14,986 | features |
| nhanes/P_SMQ.xpt | smoking questionnaire | 11,137 | features |
| nhanes/P_MCQ.xpt | medical conditions questionnaire | 14,986 | features |
| nhanes/P_KIQ_U.xpt | kidney conditions questionnaire | 9,232 | exclusion rules only |
| nhanes/P_BIOPRO.xpt | standard biochemistry profile | 10,409 | serum creatinine for the label only |
| nhanes/P_ALB_CR.xpt | urine albumin and creatinine | 13,027 | albumin-to-creatinine ratio for the label only |

Row counts differ between files because each component has its own age range and not every
participant completed every component.

## UCI Chronic Kidney Disease (second dataset)

- Creators: L. Rubini, P. Soundarapandian, P. Eswaran (2015)
- Page: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
- DOI: https://doi.org/10.24432/C5G020
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Collection method: records of 400 patients collected from a hospital over a period of about
  two months. The UCI page does not name the hospital or give more detail.
- Unit of analysis: one patient.
- Downloaded with the `ucimlrepo` package (dataset id 336) and saved as received.

| File | Rows | Columns |
|---|---|---|
| uci/chronic_kidney_disease.csv | 400 | 24 features and 1 label (`class`) |

## Citations

National Center for Health Statistics, "National Health and Nutrition Examination Survey
2017-March 2020 Pre-Pandemic Data," Centers for Disease Control and Prevention, Hyattsville, MD, USA.

L. Rubini, P. Soundarapandian, and P. Eswaran, "Chronic Kidney Disease," UCI Machine Learning
Repository, 2015. doi: 10.24432/C5G020.
