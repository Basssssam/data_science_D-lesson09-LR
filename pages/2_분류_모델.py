import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import streamlit as st

# 1. 페이지 설정
st.set_page_config(page_title="분류 모델 실습", page_icon="🤖", layout="wide")

st.title("🤖 뇌졸중 예측 분류 모델 실습")
st.markdown("데이터를 바탕으로 로지스틱 회귀와 의사결정트리 모델을 학습하고 성능을 비교합니다.")

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

# 4. 입력 속성 선택 UI
st.subheader("⚙️ 1. 모델 입력 속성 선택")
all_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
default_features = ["age", "avg_glucose_level", "hypertension", "heart_disease"]  # bmi 제외 넷

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

# 6. 체질량지수(bmi) 결측치 처리 (고른 경우 훈련용 중앙값으로 채움)
if "bmi" in selected_features:
    bmi_median = df_train["bmi"].median()
    df_train["bmi"] = df_train["bmi"].fillna(bmi_median)
    df_test["bmi"] = df_test["bmi"].fillna(bmi_median)

# 7. 모델 학습
X_train = df_train[selected_features]
y_train = df_train["stroke"]
X_test = df_test[selected_features]
y_test = df_test["stroke"]

# 다수 클래스 기준 모델 (입력을 보지 않는 모델)
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_train_acc = accuracy_score(y_train, dummy.predict(X_train))
dummy_test_acc = accuracy_score(y_test, dummy.predict(X_test))

# 로지스틱 회귀 모델
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
lr_train_acc = accuracy_score(y_train, lr.predict(X_train))
lr_test_acc = accuracy_score(y_test, lr.predict(X_test))

# 의사결정트리 모델 (질문 3번까지만, 마디에 사람이 5명 미만이면 더 나누지 않음, 난수 고정)
dt = DecisionTreeClassifier(
    max_depth=3, min_samples_split=5, random_state=42
)
dt.fit(X_train, y_train)
dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt.predict(X_test))

st.markdown("---")

# 8. 모델 정확도 비교 카드 표시
st.subheader("📊 2. 모델별 정확도 비교")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 로지스틱 회귀(확률로 답하는 모델)")
    st.metric(label="테스트 정확도", value=f"{lr_test_acc * 100:.2f}%")
    st.caption(
        f"훈련 정확도: {lr_train_acc * 100:.2f}% | 테스트 정확도:"
        f" {lr_test_acc * 100:.2f}%"
    )

with col2:
    st.markdown("##### 의사결정트리(질문으로 답하는 모델)")
    st.metric(label="테스트 정확도", value=f"{dt_test_acc * 100:.2f}%")
    st.caption(
        f"훈련 정확도: {dt_train_acc * 100:.2f}% | 테스트 정확도:"
        f" {dt_test_acc * 100:.2f}%"
    )

with col3:
    st.markdown("##### 다수 클래스 기준 모델(입력을 보지 않는 모델)")
    st.metric(label="테스트 정확도", value=f"{dummy_test_acc * 100:.2f}%")
    st.caption(
        f"훈련 정확도: {dummy_train_acc * 100:.2f}% | 테스트 정확도:"
        f" {dummy_test_acc * 100:.2f}%"
    )

st.markdown("---")

# 9. 2차원 산점도 및 결정 경계 시각화
st.subheader("📈 3. 2차원 산점도 및 결정 경계")

korean_selected = [feature_map[f] for f in selected_features]
col_x_kor = st.selectbox("가로축 속성 선택", korean_selected, index=0)
col_y_kor = st.selectbox(
    "세로축 속성 선택",
    korean_selected,
    index=min(1, len(korean_selected) - 1),
)

x_feat = inv_feature_map[col_x_kor]
y_feat = inv_feature_map[col_y_kor]

# 두 축이 아닌 속성은 테스트 데이터 중앙값으로 고정
other_feats = [
    f for f in selected_features if f != x_feat and f != y_feat
]
median_vals = {}
median_desc_list = []
for f in other_feats:
    med_val = df_test[f].median()
    median_vals[f] = med_val
    median_desc_list.append(f"**{feature_map[f]}**: {med_val}")

if median_desc_list:
    st.info(
        "💡 가로축과 세로축이 아닌 나머지 속성은 테스트 데이터의 중앙값으로"
        " 고정했습니다: "
        + ", ".join(median_desc_list)
    )

df_test["stroke_str"] = df_test["stroke"].map({0: "정상(0)", 1: "뇌졸중(1)"})

fig = go.Figure()

x_min, x_max = (
    df_test[x_feat].min() * 0.9 if df_test[x_feat].min() > 0 else df_test[x_feat].min() * 1.1,
    df_test[x_feat].max() * 1.1,
)
# 수정된 부분: y_min과 y_max 모두 올바르게 언패킹되도록 최솟값과 최댓값 지정
y_min, y_max = (
    df_test[y_feat].min() * 0.9 if df_test[y_feat].min() > 0 else df_test[y_feat].min() * 1.1,
    df_test[y_feat].max() * 1.1,
)

xx, yy = np.meshgrid(
    np.linspace(df_test[x_feat].min(), df_test[x_feat].max(), 200),
    np.linspace(df_test[y_feat].min(), df_test[y_feat].max(), 200),
)

grid_data = pd.DataFrame()
for f in selected_features:
    if f == x_feat:
        grid_data[f] = xx.ravel()
    elif f == y_feat:
        grid_data[f] = yy.ravel()
    else:
        grid_data[f] = median_vals[f]

# 의사결정트리 예측 영역 색칠 (배경)
dt_preds = dt.predict(grid_data)
Z_dt = dt_preds.reshape(xx.shape)

fig.add_trace(
    go.Contour(
        z=Z_dt,
        x=np.linspace(df_test[x_feat].min(), df_test[x_feat].max(), 200),
        y=np.linspace(df_test[y_feat].min(), df_test[y_feat].max(), 200),
        showscale=False,
        colorscale=[
            [0, "rgba(200, 220, 255, 0.25)"],
            [1, "rgba(255, 200, 200, 0.25)"],
        ],
        hoverinfo="skip",
    )
)

# 로지스틱 회귀 0.5 확률 결정 경계선
lr_probs = lr.predict_proba(grid_data)[:, 1]
Z_lr = lr_probs.reshape(xx.shape)

fig.add_trace(
    go.Contour(
        z=Z_lr,
        x=np.linspace(df_test[x_feat].min(), df_test[x_feat].max(), 200),
        y=np.linspace(df_test[y_feat].min(), df_test[y_feat].max(), 200),
        showscale=False,
        contours=dict(
            start=0.5, end=0.5, size=0, coloring="none", showlines=True
        ),
        line=dict(color="black", width=3, dash="solid"),
        name="로지스틱 경계선(p=0.5)",
    )
)

# 테스트 데이터 산점도 점 찍기
for label, color in [("정상(0)", "#3366CC"), ("뇌졸중(1)", "#DC3912")]:
    subset = df_test[df_test["stroke_str"] == label]
    fig.add_trace(
        go.Scatter(
            x=subset[x_feat],
            y=subset[y_feat],
            mode="markers",
            name=label,
            marker=dict(color=color, size=8, opacity=0.8),
        )
    )

fig.update_layout(
    title=f"{col_x_kor} vs {col_y_kor} 산점도 및 모델 결정 경계",
    xaxis_title=col_x_kor,
    yaxis_title=col_y_kor,
    legend_title="실제 뇌졸중 여부",
    template="plotly_white",
    height=550,
)

st.plotly_chart(fig, use_container_width=True)

# 결정 경계선 위치 확인 안내 글
if (Z_lr.min() <= 0.5) and (Z_lr.max() >= 0.5):
    st.success(
        "✅ 로지스틱 회귀 모델의 0.5 결정 경계선이 현재 그림 범위 내에"
        " 위치하고 있습니다."
    )
else:
    st.warning(
        "⚠️ 로지스틱 회귀 모델의 0.5 결정 경계선이 현재 그림 범위 밖(모든"
        " 데이터가 0.5보다 크거나 작음)에 있습니다."
    )

st.markdown("---")

# 10. 의사결정트리 규칙 시각화 (Graphviz)
st.subheader("🌳 4. 의사결정트리 질문 구조 시각화")

korean_feature_names = [feature_map[f] for f in selected_features]

dot_data = export_graphviz(
    dt,
    out_file=None,
    feature_names=korean_feature_names,
    class_names=["정상(0)", "뇌졸중(1)"],
    filled=True,
    rounded=True,
    special_chars=True,
    precision=2,
)

st.graphviz_chart(dot_data)

# 트리 요약 정보 계산
is_leaf = dt.tree_.children_left == dt.tree_.children_right
num_leaves = np.sum(is_leaf)
leaf_values = dt.tree_.value[is_leaf]
leaf_predictions = np.argmax(leaf_values, axis=1)
num_class_0_leaves = np.sum(leaf_predictions == 0)

used_features_indices = dt.tree_.feature
used_features_indices = used_features_indices[used_features_indices != -2]
used_feature_names = set(
    [korean_feature_names[i] for i in used_features_indices]
)

st.markdown(f"""
- **답을 내는 마디(리프 노드) 수**: 총 {num_leaves}개
- **그중 '정상(아님)'이라고 답하는 마디 수**: {num_class_0_leaves}개
- **고른 속성 가운데 이 나무가 실제로 물어본 속성**: {", ".join(used_feature_names) if used_feature_names else "없음"}
""")
