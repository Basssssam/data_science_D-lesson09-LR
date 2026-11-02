import json
from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 뇌졸중 예측 실습실")
st.write(
    "뇌졸중 관련 정보를 담은 데이터를 살펴보는 실습 공간입니다. "
    "먼저 데이터의 전체 규모와 각 열의 구성을 확인해 보세요."
)

# GitHub의 파일 페이지 주소가 아니라 CSV 원본 주소를 사용합니다.
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Basssssam/data_science_D-data/main/stroke.csv"
)

# 입력한 설명과 출처를 저장할 폴더 및 JSON 파일
RECORD_DIR = Path(__file__).parent / "내_기록"
RECORD_FILE = RECORD_DIR / "뇌졸중_데이터_기록.json"

COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]


# --------------------------------------------------
# 데이터와 기록 불러오기
# --------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="utf-8")

    # 교재에 제시된 열 순서와 이름을 확인합니다.
    missing_columns = [col for col in COLUMNS if col not in data.columns]
    if missing_columns:
        raise ValueError(
            "CSV 파일에서 다음 열을 찾을 수 없습니다: "
            + ", ".join(missing_columns)
        )

    return data[COLUMNS]


def load_records():
    default_records = {
        "meanings": {column: "" for column in COLUMNS},
        "source": "",
    }

    if not RECORD_FILE.exists():
        return default_records

    try:
        with RECORD_FILE.open("r", encoding="utf-8") as file:
            saved = json.load(file)

        meanings = default_records["meanings"].copy()
        meanings.update(saved.get("meanings", {}))

        return {
            "meanings": meanings,
            "source": saved.get("source", ""),
        }
    except (json.JSONDecodeError, OSError, AttributeError):
        return default_records


def save_records(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    records = {
        "meanings": meanings,
        "source": source,
