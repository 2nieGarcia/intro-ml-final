"""Download the original datasets into data/original/.

Run from the project root:
    python src/download_data.py

Files that already exist are skipped, so it is safe to run more than once.
"""

from pathlib import Path
from urllib.request import Request, urlopen

from ucimlrepo import fetch_ucirepo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NHANES_DIR = PROJECT_ROOT / "data" / "original" / "nhanes"
UCI_DIR = PROJECT_ROOT / "data" / "original" / "uci"

# NHANES 2017-March 2020 pre-pandemic files (see documentation/preregistration.md)
NHANES_BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
NHANES_FILES = [
    "P_DEMO",    # demographics: age, sex, pregnancy, exam status
    "P_BMX",     # body measures: BMI, waist
    "P_BPXO",    # measured blood pressure
    "P_BPQ",     # blood pressure and cholesterol questionnaire
    "P_DIQ",     # diabetes questionnaire
    "P_SMQ",     # smoking questionnaire
    "P_MCQ",     # medical conditions: heart disease, stroke, family history
    "P_KIQ_U",   # kidney questionnaire (exclusions only)
    "P_BIOPRO",  # serum creatinine (label only)
    "P_ALB_CR",  # urine albumin-to-creatinine ratio (label only)
]

UCI_CKD_ID = 336


def download_nhanes():
    NHANES_DIR.mkdir(parents=True, exist_ok=True)
    for name in NHANES_FILES:
        target = NHANES_DIR / f"{name}.xpt"
        if target.exists():
            print(f"{name}: already downloaded")
            continue
        request = Request(NHANES_BASE_URL + name + ".xpt", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request) as response:
            content = response.read()
        # CDC returns an HTML page instead of the file when a name is wrong
        if content[:6] != b"HEADER":
            raise ValueError(f"{name} did not download as an XPT file")
        target.write_bytes(content)
        print(f"{name}: {len(content) / 1024:.0f} KB")


def download_uci():
    UCI_DIR.mkdir(parents=True, exist_ok=True)
    target = UCI_DIR / "chronic_kidney_disease.csv"
    if target.exists():
        print("UCI CKD: already downloaded")
        return
    dataset = fetch_ucirepo(id=UCI_CKD_ID)
    # saved exactly as received, with no cleaning
    dataset.data.original.to_csv(target, index=False)
    print(f"UCI CKD: {len(dataset.data.original)} rows")


if __name__ == "__main__":
    download_nhanes()
    download_uci()
