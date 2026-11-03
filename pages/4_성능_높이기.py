import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# 1. 페이지 설정
st.set_page_config(page_title="분류 모델 성능 높이기", page_icon="⚖️", layout="wide")

st.title("⚖️ 클래스 가중치 조정에 따른 분류 모델 성능 비교")
st.markdown(
    "뇌졸중 데이터의 불균형(희귀성) 문제를 해결하기 위해 **가중치 맞추기 전(balanced 없음)**과 **맞춘 뒤(class_weight='balanced')**의 결과를 비교합니다."
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

# 4. 입력 속성 선택 UI (기본값은 bmi 제외 넷)
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
    test_indices.extend(list(chunk[:3]))
    train_indices.extend(list(chunk[3:]))

df_train = df_sorted.iloc[train_indices].copy()
df_test = df_sorted.iloc[test_indices].copy()

# 체질량지수(bmi) 결측치 처리 (훈련용 중앙값으로 채움)
if "bmi" in selected_features:
    bmi_median = df_train["bmi"].median()
    df_train["bmi"] = df_train["bmi"].fillna(bmi_median)
    df_test["bmi"] = df_test["bmi"].fillna(bmi_median)

# 6. 모델 학습 (가중치 전 vs 후)
X_train = df_train[selected_features]
y_train = df_train["stroke"]
X_test = df_test[selected_features]
y_test = df_test["stroke"]

lr_before = LogisticRegression(random_state=42, max_iter=1000)
lr_before.fit(X_train, y_train)
lr_before_pred = lr_before.predict(X_test)

dt_before = DecisionTreeClassifier(
    max_depth=3, min_samples_split=5, random_state=42
)
dt_before.fit(X_train, y_train)
dt_before_pred = dt_before.predict(X_test)

lr_after = LogisticRegression(
    class_weight="balanced", random_state=42, max_iter=1000
)
lr_after.fit(X_train, y_train)
lr_after_pred = lr_after.predict(X_test)

dt_after = DecisionTreeClassifier(
    class_weight="balanced", max_depth=3, min_samples_split=5, random_state=42
)
dt_after.fit(X_train, y_train)
dt_after_pred = dt_after.predict(X_test)

st.markdown("---")


# 7. 혼동행렬 및 평가지표 계산 함수
def get_metrics_dict(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total if total > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0
        )
        prec_str = f"{precision * 100:.2f}%"
        f1_str = f"{f1 * 100:.2f}%"
    else:
        prec_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"

    return {
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Acc": f"{accuracy * 100:.2f}%",
        "Recall": f"{recall * 100:.2f}%",
        "Precision": prec_str,
        "F1": f1_str,
    }


total_test_count = len(y_test)

# 8. 모델별 가중치 전·후 비교 화면 구성
st.subheader("📊 2. 가중치 조정 전후 혼동행렬 및 평가지표 비교")

models_comparison = [
    ("로지스틱 회귀 (확률로 답하는 모델)", lr_before_pred, lr_after_pred),
    ("의사결정트리 (질문으로 답하는 모델)", dt_before_pred, dt_after_pred),
]

for model_name, pred_before, pred_after in models_comparison:
    st.markdown(f"#### 🤖 {model_name}")
    col1, col2 = st.columns(2)

    metrics_b = get_metrics_dict(y_test, pred_before)
    metrics_a = get_metrics_dict(y_test, pred_after)

    with col1:
        st.markdown("##### 📌 가중치 조정 전")
        cm_df_b = pd.DataFrame(
            {
                "예측: 정상(0)": [
                    f"TN: {metrics_b['TN']}",
                    f"FN: {metrics_b['FN']}",
                ],
                "예측: 뇌졸중(1)": [
                    f"FP: {metrics_b['FP']}",
                    f"TP: {metrics_b['TP']}",
                ],
                "행 합계": [
                    str(metrics_b['TN'] + metrics_b['FP']),
                    str(metrics_b['FN'] + metrics_b['TP']),
                ],
            },
            index=["실제: 정상(0)", "실제: 뇌졸중(1)"],
        )
        cm_df_b.loc["열 합계"] = [
            str(metrics_b['TN'] + metrics_b['FN']),
            str(metrics_b['FP'] + metrics_b['TP']),
            str(total_test_count),
        ]
        st.dataframe(cm_df_b, use_container_width=True)

        ind_df_b = pd.DataFrame(
            {
                "지표": ["정확도", "재현율", "정밀도", "F1 스코어"],
                "값": [
                    metrics_b["Acc"],
                    metrics_b["Recall"],
                    metrics_b["Precision"],
                    metrics_b["F1"],
                ],
            }
        )
        st.dataframe(ind_df_b, use_container_width=True)

    with col2:
        st.markdown("##### ⚖️ 가중치 조정 후 (balanced)")
        cm_df_a = pd.DataFrame(
            {
                "예측: 정상(0)": [
                    f"TN: {metrics_a['TN']}",
                    f"FN: {metrics_a['FN']}",
                ],
                "예측: 뇌졸중(1)": [
                    f"FP: {metrics_a['FP']}",
                    f"TP: {metrics_a['TP']}",
                ],
                "행 합계": [
                    str(metrics_a['TN'] + metrics_a['FP']),
                    str(metrics_a['FN'] + metrics_a['TP']),
                ],
            },
            index=["실제: 정상(0)", "실제: 뇌졸중(1)"],
        )
        cm_df_a.loc["열 합계"] = [
            str(metrics_a['TN'] + metrics_a['FN']),
            str(metrics_a['FP'] + metrics_a['TP']),
            str(total_test_count),
        ]
        st.dataframe(cm_df_a, use_container_width=True)

        ind_df_a = pd.DataFrame(
            {
                "지표": ["정확도", "재현율", "정밀도", "F1 스코어"],
                "값": [
                    metrics_a["Acc"],
                    metrics_a["Recall"],
                    metrics_a["Precision"],
                    metrics_a["F1"],
                ],
            }
        )
        st.dataframe(ind_df_a, use_container_width=True)

    st.info(
        f"💡 **변화 요약**: 찾아낸 환자 수(TP)는 **{metrics_b['TP']}명 ➔"
        f" {metrics_a['TP']}명**, 놓친 환자 수(FN)는 **{metrics_b['FN']}명 ➔"
        f" {metrics_a['FN']}명**, 헛짚은 사람 수(FP)는 **{metrics_b['FP']}명 ➔"
        f" {metrics_a['FP']}명**으로 변경되었습니다."
    )
    st.markdown("---")

# 9. 시각화 (2차원 또는 3차원 산점도 및 결정 경계 비교)
st.subheader("📈 3. 가중치 조정 전후 결정 경계 및 추세선 비교")

model_choice = st.selectbox(
    "시각화할 모델을 선택하세요:",
    ["로지스틱 회귀 (확률로 답하는 모델)", "의사결정트리 (질문으로 답하는 모델)"],
)
current_model_before = (
    lr_before if "로지스틱" in model_choice else dt_before
)
current_model_after = lr_after if "로지스틱" in model_choice else dt_after

korean_selected = [feature_map[f] for f in selected_features]

if len(selected_features) == 2:
    # 속성 2개일 때: 2차원 산점도, 결정 경계선 및 데이터 추세선(OLS) 표시
    col_x_kor, col_y_kor = korean_selected[0], korean_selected[1]
    x_feat, y_feat = selected_features[0], selected_features[1]

    df_test["stroke_str"] = df_test["stroke"].map({0: "정상(0)", 1: "뇌졸중(1)"})

    # Plotly Express를 이용해 산점도와 추세선(trendline="ols") 동시 생성
    fig = px.scatter(
        df_test,
        x=x_feat,
        y=y_feat,
        color="stroke_str",
        color_discrete_map={"정상(0)": "#3366CC", "뇌졸중(1)": "#DC3912"},
        trendline="ols",
        labels={x_feat: col_x_kor, y_feat: col_y_kor, "stroke_str": "실제 뇌졸중 여부"},
        title=f"{col_x_kor} vs {col_y_kor} : 산점도, 추세선 및 결정 경계 비교",
    )

    # 결정 경계선(Contour) 겹치기
    xx, yy = np.meshgrid(
        np.linspace(df_test[x_feat].min(), df_test[x_feat].max(), 100),
        np.linspace(df_test[y_feat].min(), df_test[y_feat].max(), 100),
    )
    grid_data = pd.DataFrame({x_feat: xx.ravel(), y_feat: yy.ravel()})

    Z_b = current_model_before.predict(grid_data).reshape(xx.shape)
    fig.add_trace(
        go.Contour(
            z=Z_b,
            x=np.linspace(df_test[x_feat].min(), df_test[x_feat].max(), 100),
            y=np.linspace(df_test[y_feat].min(), df_test[y_feat].max(), 100),
            showscale=False,
            contours=dict(start=0.5, end=0.5, size=0, coloring="none"),
            line=dict(color="blue", width=2, dash="dash"),
            name="조정 전 경계선",
        )
    )

    Z_a = current_model_after.predict(grid_data).reshape(xx.shape)
    fig.add_trace(
        go.Contour(
            z=Z_a,
            x=np.linspace(df_test[x_feat].min(), df_test[x_feat].max(), 100),
            y=np.linspace(df_test[y_feat].min(), df_test[y_feat].max(), 100),
            showscale=False,
            contours=dict(start=0.5, end=0.5, size=0, coloring="none"),
            line=dict(color="red", width=2, dash="solid"),
            name="조정 후 경계선(balanced)",
        )
    )

    fig.update_layout(
        xaxis_title=col_x_kor,
        yaxis_title=col_y_kor,
        template="plotly_white",
        height=550,
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # 속성 3개 이상일 때: 3차원 산점도 및 결정 경계면 표시
    st.markdown(
        "✨ 선택하신 속성이 3개 이상이므로, **처음 3개 속성**을 각각 X, Y, Z축으로"
        " 하여 3차원 산점도와 가중치 조정 전후의 **0.5 결정 경계면**을 겹쳐서"
        " 표시합니다."
    )
    x_feat, y_feat, z_feat = (
        selected_features[0],
        selected_features[1],
        selected_features[2],
    )
    col_x, col_y, col_z = (
        feature_map[x_feat],
        feature_map[y_feat],
        feature_map[z_feat],
    )

    other_feats = selected_features[3:]
    median_vals = {f: df_test[f].median() for f in other_feats}

    df_test["stroke_str"] = df_test["stroke"].map({0: "정상(0)", 1: "뇌졸중(1)"})
    fig = go.Figure()

    for label, color in [("정상(0)", "#3366CC"), ("뇌졸중(1)", "#DC3912")]:
        subset = df_test[df_test["stroke_str"] == label]
        fig.add_trace(
            go.Scatter3d(
                x=subset[x_feat],
                y=subset[y_feat],
                z=subset[z_feat],
                mode="markers",
                name=label,
                marker=dict(color=color, size=5, opacity=0.8),
            )
        )

    nx, ny, nz = 30, 30, 30
    x_vals = np.linspace(df_test[x_feat].min(), df_test[x_feat].max(), nx)
    y_vals = np.linspace(df_test[y_feat].min(), df_test[y_feat].max(), ny)
    z_vals = np.linspace(df_test[z_feat].min(), df_test[z_feat].max(), nz)
    GX, GY, GZ = np.meshgrid(x_vals, y_vals, z_vals, indexing="ij")

    grid_3d = pd.DataFrame(
        {x_feat: GX.ravel(), y_feat: GY.ravel(), z_feat: GZ.ravel()}
    )
    for f in other_feats:
        grid_3d[f] = median_vals[f]

    if hasattr(current_model_before, "predict_proba"):
        val_b = current_model_before.predict_proba(grid_3d)[:, 1].reshape(
            GX.shape
        )
    else:
        val_b = current_model_before.predict(grid_3d).astype(float).reshape(GX.shape)

    if hasattr(current_model_after, "predict_proba"):
        val_a = current_model_after.predict_proba(grid_3d)[:, 1].reshape(
            GX.shape
        )
    else:
        val_a = current_model_after.predict(grid_3d).astype(float).reshape(GX.shape)

    fig.add_trace(
        go.Isosurface(
            x=GX.flatten(),
            y=GY.flatten(),
            z=GZ.flatten(),
            value=val_b.flatten(),
            isomin=0.48,
            isomax=0.52,
            surface=dict(show=True, count=1, fill=0.3),
            colorscale=[[0, "blue"], [1, "blue"]],
            showscale=False,
            name="조정 전 경계(0.5)",
        )
    )

    fig.add_trace(
        go.Isosurface(
            x=GX.flatten(),
            y=GY.flatten(),
            z=GZ.flatten(),
            value=val_a.flatten(),
            isomin=0.48,
            isomax=0.52,
            surface=dict(show=True, count=1, fill=0.3),
            colorscale=[[0, "red"], [1, "red"]],
            showscale=False,
            name="조정 후 경계(balanced)",
        )
    )

    fig.update_layout(
        title=f"3차원 산점도 및 결정 경계면 비교 ({col_x} vs {col_y} vs {col_z})",
        scene=dict(
            xaxis_title=col_x, yaxis_title=col_y, zaxis_title=col_z
        ),
        template="plotly_white",
        height=700,
    )
    st.plotly_chart(fig, use_container_width=True)
