import json
import os
import pandas as pd
import streamlit as st

# 1. 페이지 설정 (브라우저 탭 제목 및 아이콘)
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

# 앱 제목 및 소개
st.title("🧠 뇌졸중 예측 실습실")
st.markdown(
    "이 앱은 뇌졸중 예측 데이터를 탐색하고 실습하기 위한 공간입니다. 데이터의"
    " 기본 구조를 파악하고, 각 변수의 의미와 출처를 기록해 보세요."
)
st.markdown("---")

# 2. 데이터 로드
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data():
  return pd.read_csv(DATA_URL, encoding="utf-8")


try:
  df = load_data()
except Exception as e:
  st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
  st.stop()

# 3. 핵심 지표 카드 4개 (전체 사람 수, 열 개수, stroke=1 사람 수, 비율)
total_people = len(df)
num_columns = len(df.columns)
stroke_count = (df["stroke"] == 1).sum()
stroke_ratio = (stroke_count / total_people) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 사람 수", f"{total_people:,}명")
col2.metric("열 개수", f"{num_columns}개")
col3.metric("stroke=1 사람 수", f"{stroke_count:,}명")
col4.metric("stroke=1 비율", f"{stroke_ratio:.2f}%")

st.markdown("---")

# 4. 열 정보 및 우리말 뜻 입력 표 (st.data_editor 활용)
st.subheader("📋 데이터셋 열 정보 및 우리말 뜻")
st.markdown(
    "교재를 참고하여 **우리말 뜻** 칸을 직접 입력하고 아래의 **[저장하기]**"
    " 버튼을 눌러주세요. 입력한 내용은 `내_기록` 폴더에 저장되어"
    " 새로고침해도 유지됩니다."
)

# 내_기록 폴더 생성
os.makedirs("내_기록", exist_ok=True)
meanings_file = os.path.join("내_기록", "column_meanings.json")

# 기본 열 정보 데이터프레임 생성
col_info_list = []
for col in df.columns:
  dtype = str(df[col].dtype)
  if "int" in dtype or "float" in dtype:
    val_type = "숫자형 (Numeric)"
  else:
    val_type = "범주형/문자형 (Categorical)"

  missing_count = int(df[col].isnull().sum())
  col_info_list.append({
      "열 이름 (Column)": col,
      "우리말 뜻": "",
      "값의 종류": val_type,
      "빈 값 개수": missing_count,
  })

default_info_df = pd.DataFrame(col_info_list)

# 기존에 저장된 우리말 뜻이 있다면 불러와서 병합
if os.path.exists(meanings_file):
  try:
    with open(meanings_file, "r", encoding="utf-8") as f:
      saved_data = json.load(f)
      saved_df = pd.DataFrame(saved_data)
      if (
          "열 이름 (Column)" in saved_df.columns
          and "우리말 뜻" in saved_df.columns
      ):
        meaning_map = dict(
            zip(saved_df["열 이름 (Column)"], saved_df["우리말 뜻"])
        )
        default_info_df["우리말 뜻"] = default_info_df[
            "열 이름 (Column)"
        ].map(lambda x: meaning_map.get(x, ""))
  except Exception:
    pass

# 편집 가능한 표 생성
edited_df = st.data_editor(
    default_info_df,
    num_rows="fixed",
    use_container_width=True,
    key="column_meanings_editor",
)

if st.button("우리말 뜻 저장하기"):
  edited_df.to_json(
      meanings_file, orient="records", force_ascii=False, indent=4
  )
  st.success("우리말 뜻이 `내_기록/column_meanings.json`에 저장되었습니다!")

st.markdown("---")

# 5. 데이터 처음 다섯 줄 미리보기
st.subheader("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(), use_container_width=True)

st.markdown("---")

# 6. 데이터 출처 작성 영역
st.subheader("📖 데이터 출처")
source_file = os.path.join("내_기록", "data_source.json")

current_source = ""
if os.path.exists(source_file):
  try:
    with open(source_file, "r", encoding="utf-8") as f:
      source_data = json.load(f)
      current_source = source_data.get("source_text", "")
  except Exception:
    pass

source_text = st.text_area(
    "교재를 참고하여 데이터 출처를 작성해 주세요:",
    value=current_source,
    height=120,
    placeholder="여기에 데이터 출처 관련 설명을 적어보세요...",
)

if st.button("데이터 출처 저장하기"):
  with open(source_file, "w", encoding="utf-8") as f:
    json.dump({"source_text": source_text}, f, ensure_ascii=False, indent=4)
  st.success("데이터 출처가 `내_기록/data_source.json`에 저장되었습니다!")
