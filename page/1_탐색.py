import pandas as pd
import plotly.express as px
import streamlit as st

# 페이지 설정
st.set_page_config(page_title="데이터 탐색", page_icon="📊", layout="wide")

st.title("📊 뇌졸중 데이터 탐색 페이지")
st.markdown(
    "데이터셋을 바탕으로 주요 변수의 분포, 뇌졸중 여부에 따른 비교 및 위험 요인을"
    " 탐색합니다."
)

# 첫 화면과 동일한 데이터 주소 및 로드 함수
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data():
  return pd.read_csv(DATA_URL, encoding="utf-8")


try:
  df = load_data()
except Exception as e:
  st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
  st.stop()

# 주요 컬럼 수치형 변환 예외 처리
for col in [
    "age",
    "avg_glucose_level",
    "bmi",
    "stroke",
    "hypertension",
    "heart_disease",
]:
  if col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

stroke_col = "stroke" if "stroke" in df.columns else df.columns[-1]
df["stroke_label"] = df[stroke_col].map({0: "정상(0)", 1: "뇌졸중(1)"})

st.markdown("---")

# 1. 나이와 평균 혈당의 분포를 히스토그램 두 개로 나란히 보여 줘.
st.subheader("1. 나이 및 평균 혈당 분포")
col1, col2 = st.columns(2)

with col1:
  if "age" in df.columns:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이(Age) 분포",
        labels={"age": "나이", "count": "인원 수"},
        color_discrete_sequence=["#3366CC"],
    )
    st.plotly_chart(fig_age, use_container_width=True)
  else:
    st.warning("age 컬럼을 찾을 수 없습니다.")

with col2:
  if "avg_glucose_level" in df.columns:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당(Avg Glucose Level) 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "인원 수"},
        color_discrete_sequence=["#DC3912"],
    )
    st.plotly_chart(fig_glucose, use_container_width=True)
  else:
    st.warning("avg_glucose_level 컬럼을 찾을 수 없습니다.")

st.markdown("---")

# 2. 뇌졸중을 겪은 사람과 겪지 않은 사람으로 나누어, 나이와 평균 혈당을 상자그림 두 개로 비교해 줘. 두 그룹의 평균값도 표로 함께 적어 줘.
st.subheader("2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교")
col1, col2 = st.columns(2)

with col1:
  if "age" in df.columns:
    fig_box_age = px.box(
        df,
        x="stroke_label",
        y="age",
        color="stroke_label",
        title="뇌졸중 여부에 따른 나이 비교",
        labels={"stroke_label": "뇌졸중 여부", "age": "나이"},
    )
    st.plotly_chart(fig_box_age, use_container_width=True)

with col2:
  if "avg_glucose_level" in df.columns:
    fig_box_glucose = px.box(
        df,
        x="stroke_label",
        y="avg_glucose_level",
        color="stroke_label",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"},
    )
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값 표
st.markdown("##### 📋 그룹별 평균값 비교 표")
mean_cols = [c for c in ["age", "avg_glucose_level"] if c in df.columns]
if mean_cols:
  mean_table = df.groupby("stroke_label")[mean_cols].mean().reset_index()
  mean_table.columns = ["뇌졸중 여부"] + [
      "평균 나이" if c == "age" else "평균 혈당" for c in mean_cols
  ]
  st.dataframe(mean_table, use_container_width=True)

st.markdown("---")

# 3. 고혈압이 있는 사람과 없는 사람의 뇌졸중 비율을 막대그래프로 보여 줘. 심장병도 같은 방법으로 보여 줘.
st.subheader("3. 고혈압 및 심장병에 따른 뇌졸중 비율")
col1, col2 = st.columns(2)

hypertension_col = (
    "hypertension" if "hypertension" in df.columns else "고혈압"
)
heart_col = "heart_disease" if "heart_disease" in df.columns else "심장병"

with col1:
  if hypertension_col in df.columns:
    hyp_df = df.groupby(hypertension_col)[stroke_col].mean().reset_index()
    hyp_df["stroke_rate_pct"] = hyp_df[stroke_col] * 100
    hyp_df["label"] = hyp_df[hypertension_col].map({0: "없음(0)", 1: "있음(1)"})
    fig_hyp = px.bar(
        hyp_df,
        x="label",
        y="stroke_rate_pct",
        color="label",
        title="고혈압 여부에 따른 뇌졸중 비율",
        labels={"label": "고혈압 여부", "stroke_rate_pct": "뇌졸중 비율 (%)"},
        text="stroke_rate_pct",
    )
    fig_hyp.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig_hyp, use_container_width=True)
  else:
    st.info("고혈압 데이터 컬럼을 찾을 수 없습니다.")

with col2:
  if heart_col in df.columns:
    heart_df = df.groupby(heart_col)[stroke_col].mean().reset_index()
    heart_df["stroke_rate_pct"] = heart_df[stroke_col] * 100
    heart_df["label"] = heart_df[heart_col].map({0: "없음(0)", 1: "있음(1)"})
    fig_heart = px.bar(
        heart_df,
        x="label",
        y="stroke_rate_pct",
        color="label",
        title="심장병 여부에 따른 뇌졸중 비율",
        labels={"label": "심장병 여부", "stroke_rate_pct": "뇌졸중 비율 (%)"},
        text="stroke_rate_pct",
    )
    fig_heart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig_heart, use_container_width=True)
  else:
    st.info("심장병 데이터 컬럼을 찾을 수 없습니다.")

st.markdown("---")

# 4. 체질량지수(bmi)가 비어 있는 사람들만 따로 세어서, 그 사람들의 뇌졸중 비율과 전체 뇌졸중 비율을 한 표에 나란히 보여 줘.
st.subheader("4. 체질량지수(BMI) 결측치 그룹 vs 전체 뇌졸중 비율 비교")
bmi_col = "bmi" if "bmi" in df.columns else "BMI"

if bmi_col in df.columns:
  missing_bmi_df = df[df[bmi_col].isna()]
  missing_count = len(missing_bmi_df)
  missing_stroke_rate = (
      (missing_bmi_df[stroke_col].mean() * 100) if missing_count > 0 else 0
  )

  total_count = len(df)
  total_stroke_rate = df[stroke_col].mean() * 100

  comparison_data = {
      "구분": ["BMI 결측치 그룹", "전체 데이터"],
      "인원 수": [missing_count, total_count],
      "뇌졸중 비율 (%)": [
          round(missing_stroke_rate, 2),
          round(total_stroke_rate, 2),
      ],
  }
  comp_df = pd.DataFrame(comparison_data)
  st.dataframe(comp_df, use_container_width=True)
else:
  st.info("BMI 데이터 컬럼을 찾을 수 없습니다.")

st.markdown("---")

# 5. 흡연 상태의 값마다 사람 수를 세어 표로 보여 줘.
st.subheader("5. 흡연 상태(Smoking Status)별 인원 수")
smoking_col = "smoking_status" if "smoking_status" in df.columns else "흡연상태"

if smoking_col in df.columns:
  smoking_counts = (
      df[smoking_col]
      .value_counts(dropna=False)
      .reset_index()
  )
  smoking_counts.columns = ["흡연 상태", "인원 수"]
  st.dataframe(smoking_counts, use_container_width=True)
else:
  st.info("흡연 상태 데이터 컬럼을 찾을 수 없습니다.")
