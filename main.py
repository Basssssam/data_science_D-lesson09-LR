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
for column in COLUMNS:
    table_rows.append(
        {
            "열 이름": column,
            "우리말 뜻": saved_records["meanings"].get(column, ""),
            "값의 종류": describe_values(df[column]),
            "빈 값 개수": int(df[column].isna().sum()),
        }
    )

table_df = pd.DataFrame(table_rows)

edited_table = st.data_editor(
    table_df,
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 참고해 각 열의 뜻을 직접 입력하세요.",
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d개",
        ),
    },
    key="meaning_editor",
)

# 표에서 입력한 우리말 뜻을 JSON에 반영합니다.
meanings_to_save = {
    row["열 이름"]: row["우리말 뜻"]
    for _, row in edited_table.iterrows()
}


# --------------------------------------------------
# 데이터 원본 미리보기
# --------------------------------------------------
st.subheader("🔎 데이터 미리보기")
st.write("원본 데이터의 처음 다섯 줄입니다.")
st.dataframe(df.head(5), use_container_width=True)


# --------------------------------------------------
# 데이터 출처 입력 및 JSON 저장
# --------------------------------------------------
st.subheader("🔗 데이터 출처")
source_text = st.text_area(
    "교재에 나온 데이터 출처를 입력하세요.",
    value=saved_records.get("source", ""),
    placeholder="교재를 참고해 데이터 출처를 입력하세요.",
    height=100,
    key="source_editor",
)

# 우리말 뜻과 출처를 함께 저장합니다.
try:
    save_records(meanings_to_save, source_text)
    st.caption("입력한 우리말 뜻과 데이터 출처는 내_기록/뇌졸중_데이터_기록.json에 자동 저장됩니다.")
