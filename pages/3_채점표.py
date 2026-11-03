import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# 1. 페이지 설정
st.set_page_config(page_title="분류 모델 채점표", page_icon="📋", layout="wide")

st.title("📋 분류 모델 채점표 및 평가 실습")
st.markdown(
    "테스트 데이터를 바탕으로 로지스틱 회귀와 의사결정트리 모델의 혼동행렬과 평가지표를 상세히 비교합니다."
)

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

# 3. 속성 매핑 사전 (영문 <-> 우리말)
feature_map = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
inv_feature_map = {v: k for k, v in feature_map.items()}

st.markdown("---")

# 4. 입력 속성 선택 UI ('분류 모델' 페이지와 동일하게 기본값은 bmi 제외 넷)
st.subheader("⚙️ 1. 모델 입력 속성 선택")
all_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
default_features = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

selected_korean = st.multiselect(
    "학습에 사용할 속성을 선택하세요 (2개 이상 선택):",
    options=[feature_map[f] for f in all_features],
    default=[feature_map[f] for f in default_features],
)

selected_features = [inv_feature_map[k] for k in selected_korean]

if len(selected_features) < 2:
    st.warning(
        "⚠️ 속성을 2개 이상 선택해야 모델을 학습할 수 있습니다. 위에서 속성을 더 선택해 주세요."
    )
    st.stop()

# 5. 데이터 분할 (번호 순 정렬 후 10명씩 묶어 앞 3명 테스트용, 나머지 7명 훈련용)
df_sorted = (
    df.sort_values(by="id").reset_index(drop=True)
    if "id" in df.columns
    else df.reset_index(drop=True)
)

test_indices = []
train_indices = []

for i in range(len(df_sorted) // 10):
    chunk = range(i * 10, (i + 1) * 10)
    test_indices.extend(list(chunk[:3]))  # 앞 세 명 테스트용
    train_indices.extend(list(chunk[3:]))  # 나머지 일곱 명 훈련용

df_train = df_sorted.iloc[train_indices].copy()
df_test = df_sorted.iloc[test_indices].copy()

# 체질량지수(bmi) 결측치 처리 (고른 경우 훈련용 중앙값으로 채움)
if "bmi" in selected_features:
    bmi_median = df_train["bmi"].median()
    df_train["bmi"] = df_train["bmi"].fillna(bmi_median)
    df_test["bmi"] = df_test["bmi"].fillna(bmi_median)

# 6. 모델 학습
X_train = df_train[selected_features]
y_train = df_train["stroke"]
X_test = df_test[selected_features]
y_test = df_test["stroke"]

# 다수 클래스 기준 모델 (훈련용에서 많은 쪽으로만 답하는 모델)
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

# 로지스틱 회귀 모델
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# 의사결정트리 모델 (질문 3번 제한, 마디 인원 5명 미만 분할 중지, 난수 고정)
dt = DecisionTreeClassifier(
    max_depth=3, min_samples_split=5, random_state=42
)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

st.markdown("---")

# 7. 테스트용 사람 수 및 뇌졸중인 사람 수 표시
total_test_count = len(y_test)
actual_stroke_count = int(y_test.sum())
st.markdown(
    f"📌 **테스트용 총 사람 수**: {total_test_count}명 중 실제 뇌졸중인 사람"
    f" 수: **{actual_stroke_count}명**"
)

st.markdown("---")

# 8. 혼동행렬(Confusion Matrix) 2x2 표 표시 (행: 실제, 열: 예측)
st.subheader("📊 2. 모델별 혼동행렬 (실제 vs 예측)")


def get_cm_dict(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn, fp, fn, tp


col1, col2, col3 = st.columns(3)

models_eval = [
    ("로지스틱 회귀(확률로 답하는 모델)", lr_pred, col1),
    ("의사결정트리(질문으로 답하는 모델)", dt_pred, col2),
    ("다수 클래스 기준 모델 (입력을 보지 않는 모델)", dummy_pred, col3),
]

cm_results = {}

for name, pred, col in models_eval:
    tn, fp, fn, tp = get_cm_dict(y_test, pred)
    cm_results[name] = {"TN": tn, "FP": fp, "FN": fn, "TP": tp}

    # 표 구성 (모든 값을 문자열로 통일하여 PyArrow 타입 에러 방지)
    cm_df = pd.DataFrame(
        {
            "예측: 정상(0)": [f"TN: {tn}", f"FN: {fn}"],
            "예측: 뇌졸중(1)": [f"FP: {fp}", f"TP: {tp}"],
            "행 합계": [str(tn + fp), str(fn + tp)],
        },
        index=["실제: 정상(0)", "실제: 뇌졸중(1)"],
    )
    cm_df.loc["열 합계"] = [
        str(tn + fn),
        str(fp + tp),
        str(total_test_count),
    ]

    with col:
        st.markdown(f"##### {name}")
        st.dataframe(cm_df, use_container_width=True)

st.markdown("---")

# 9. 평가지표 (정확도, 재현율, 정밀도, F1) 비교 표 계산 및 출력
st.subheader("📋 3. 모델별 평가지표 비교 표")


def calc_metrics(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0

    # 재현율 분모: 실제 뇌졸중인 사람 수 (tp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 정밀도 분모: 뇌졸중이라 예측한 사람 수 (tp + fp)
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        if (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        precision_str = f"{precision * 100:.2f}%"
        f1_str = f"{f1 * 100:.2f}%"
    else:
        precision_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"

    return (
        f"{accuracy * 100:.2f}%",
        f"{recall * 100:.2f}%",
        precision_str,
        f1_str,
    )


metric_rows = []
for name, metrics in cm_results.items():
    tn, fp, fn, tp = (
        metrics["TN"],
        metrics["FP"],
        metrics["FN"],
        metrics["TP"],
    )
    acc, rec, prec, f1 = calc_metrics(tn, fp, fn, tp)
    metric_rows.append(
        {
            "모델명": name,
            "정확도(Accuracy)": acc,
            "재현율(Recall)": rec,
            "정밀도(Precision)": prec,
            "F1 스코어": f1,
        }
    )

metric_df = pd.DataFrame(metric_rows)
st.dataframe(metric_df, use_container_width=True)

st.markdown("---")

# 10. 놓친 환자(FN) 및 헛짚은 사람(FP) 상세 목록 표시
st.subheader("🔍 4. 오류 분석 상세 목록 (놓친 환자 vs 헛짚은 사람)")

selected_model_name = st.selectbox(
    "목록을 확인할 모델을 선택하세요:",
    [
        "로지스틱 회귀(확률로 답하는 모델)",
        "의사결정트리(질문으로 답하는 모델)",
        "다수 클래스 기준 모델 (입력을 보지 않는 모델)",
    ],
)

selected_pred = (
    lr_pred
    if "로지스틱" in selected_model_name
    else (dt_pred if "의사결정" in selected_model_name else dummy_pred)
)

eval_df = df_test.copy()
eval_df["pred"] = selected_pred
eval_df["stroke_actual"] = eval_df["stroke"]

# 놓친 환자 (실제 1인데 0으로 예측한 경우 -> FN)
fn_df = eval_df[
    (eval_df["stroke_actual"] == 1) & (eval_df["pred"] == 0)
].copy()
# 헛짚은 사람 (실제 0인데 1로 예측한 경우 -> FP)
fp_df = eval_df[
    (eval_df["stroke_actual"] == 0) & (eval_df["pred"] == 1)
].copy()

# 나이가 많은 순서로 정렬
fn_df = fn_df.sort_values(by="age", ascending=False).reset_index(drop=True)
fp_df = fp_df.sort_values(by="age", ascending=False).reset_index(drop=True)

fn_total_count = len(fn_df)
fp_total_count = len(fp_df)

col1, col2 = st.columns(2)

display_cols = ["id", "age", "avg_glucose_level", "hypertension", "heart_disease"]
display_names = ["번호", "나이", "평균 혈당", "고혈압 여부", "심장병 여부"]


def format_subset_df(sub_df):
    if sub_df.empty:
        return pd.DataFrame(columns=display_names)
    temp_df = sub_df[display_cols].copy()
    temp_df.columns = display_names
    return temp_df


with col1:
    # 에러 수정: 변수와 단어 사이 띄어쓰기({fn_total_count}명)
    st.markdown(
        f"##### 🔴 실제 뇌졸중인데 아니라고 예측한 사람 (놓친 환자 - FN) [총"
        f" {fn_total_count}명]"
    )
    limited_fn = fn_df.head(20)
    st.dataframe(format_subset_df(limited_fn), use_container_width=True)
    if fn_total_count > 20:
        st.caption("※ 항목이 20줄을 초과하여 앞 20줄만 표시합니다.")

with col2:
    # 에러 수정: 변수와 단어 사이 띄어쓰기({fp_total_count}명)
    st.markdown(
        f"##### 🔵 아니인데 뇌졸중이라고 예측한 사람 (헛짚은 사람 - FP) [총"
        f" {fp_total_count}명]"
    )
    limited_fp = fp_df.head(20)
    st.dataframe(format_subset_df(limited_fp), use_container_width=True)
    if fp_total_count > 20:
        st.caption("※ 항목이 20줄을 초과하여 앞 20줄만 표시합니다.")

st.markdown("---")

current_fn_val = cm_results[selected_model_name]["FN"]
current_fp_val = cm_results[selected_model_name]["FP"]
st.info(
    f"📌 **확인**: 선택하신 모델의 혼동행렬 상 **FN(놓친 환자)** 수는"
    f" **{current_fn_val}명**, **FP(헛짚은 사람)** 수는 **{current_fp_val}명**이며,"
    f" 위 상세 목록의 인원 수와 정확히 일치합니다."
)
