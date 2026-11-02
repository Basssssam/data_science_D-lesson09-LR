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
        }

    with RECORD_FILE.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=4)


try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.exception(error)
    st.stop()

saved_records = load_records()


# --------------------------------------------------
# 첫 화면: 데이터 요약 카드
# --------------------------------------------------
total_people = len(df)
column_count = len(df.columns)
stroke_count = int((df["stroke"] == 1).sum())
stroke_ratio = (stroke_count / total_people * 100) if total_people else 0

st.subheader("📊 데이터 한눈에 보기")

card1, card2, card3, card4 = st.columns(4)

card1.metric("전체 사람 수", f"{total_people:,}명")
card2.metric("열 개수", f"{column_count:,}개")
card3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
card4.metric("stroke가 1인 비율", f"{stroke_ratio:.2f}%")


# --------------------------------------------------
# 열 이름과 값의 종류 정리
# --------------------------------------------------
def describe_values(series):
    values = series.dropna().unique().tolist()
    count = len(values)

    # 값의 종류가 적은 열은 실제 값을 표시합니다.
    if count <= 8:
        display_values = []
        for value in values:
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            display_values.append(str(value))
        return ", ".join(display_values) if display_values else "값 없음"

    # 값이 많은 열은 자료형과 고유값 개수로 요약합니다.
    if pd.api.types.is_numeric_dtype(series):
        return f"숫자형 · 고유값 {count:,}개"

    preview = ", ".join(str(value) for value in values[:5])
    return f"범주형 · {preview} 등 (고유값 {count:,}개)"


st.subheader("🗂️ 열 이름과 데이터 구성")
st.caption(
    "‘우리말 뜻’ 칸을 직접 입력하거나 수정하면 내_기록 폴더의 JSON 파일에 저장됩니다."
)

table_rows = []
