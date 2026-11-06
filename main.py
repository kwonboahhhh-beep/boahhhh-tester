import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DEFAULT_CSV = "ecommerce_orders_clean.csv"

st.set_page_config(page_title="주문 금액 예측기", page_icon="🛒", layout="centered")
st.title("🛒 주문 금액 예측기")
st.caption("가격(price)과 수량(quantity)으로 주문 금액을 예측하는 선형 회귀 모델")


@st.cache_data
def load_data(source):
    df = pd.read_csv(source)
    df = df.dropna(subset=["price", "quantity"]).copy()
    # 주문금액 = 가격 x 수량
    df["order_amount"] = df["price"] * df["quantity"]
    return df


# ---------- 데이터 불러오기 ----------
source = None
if os.path.exists(DEFAULT_CSV):
    source = DEFAULT_CSV
else:
    uploaded = st.file_uploader(
        f"'{DEFAULT_CSV}' 파일을 찾을 수 없습니다. CSV 파일을 업로드해 주세요.", type="csv"
    )
    if uploaded is not None:
        source = uploaded

if source is None:
    st.info("데이터 파일을 기다리는 중입니다.")
    st.stop()

df = load_data(source)

missing = {"price", "quantity"} - set(df.columns)
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}")
    st.stop()


# ---------- 모델 학습 ----------
@st.cache_resource
def train_model(data: pd.DataFrame):
    X = data[["price", "quantity"]]
    y = data["order_amount"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return model, r2_score(y_test, pred), mean_absolute_error(y_test, pred)


model, r2, mae = train_model(df)

# ---------- 슬라이더 범위: 데이터의 최솟값 ~ 최댓값 ----------
p_min, p_max = float(df["price"].min()), float(df["price"].max())
q_min, q_max = int(df["quantity"].min()), int(df["quantity"].max())
p_default = float(np.clip(df["price"].median(), p_min, p_max))
q_default = int(np.clip(df["quantity"].median(), q_min, q_max))
p_step = max(round((p_max - p_min) / 1000, 2), 0.01)

# ---------- 품목 추가/삭제 ----------
if "n_items" not in st.session_state:
    st.session_state.n_items = 1


def add_item():
    st.session_state.n_items += 1


def remove_item():
    if st.session_state.n_items > 1:
        i = st.session_state.n_items - 1
        # 삭제되는 품목의 슬라이더 상태 정리
        st.session_state.pop(f"price_{i}", None)
        st.session_state.pop(f"qty_{i}", None)
        st.session_state.n_items -= 1


st.subheader("품목 입력")
b1, b2, b3 = st.columns([1, 1, 4])
b1.button("➕", on_click=add_item, help="품목 추가", use_container_width=True)
b2.button(
    "➖",
    on_click=remove_item,
    help="마지막 품목 삭제",
    disabled=st.session_state.n_items <= 1,
    use_container_width=True,
)
b3.markdown(f"**현재 품목 수: {st.session_state.n_items}개**")

rows = []
for i in range(st.session_state.n_items):
    with st.container(border=True):
        st.markdown(f"**품목 {i + 1}**")
        price = st.slider(
            "가격",
            min_value=p_min,
            max_value=p_max,
            value=p_default,
            step=p_step,
            key=f"price_{i}",
        )
        quantity = st.slider(
            "수량",
            min_value=q_min,
            max_value=q_max,
            value=q_default,
            step=1,
            key=f"qty_{i}",
        )
        rows.append({"품목": f"품목 {i + 1}", "price": price, "quantity": quantity})

# ---------- 예측 ----------
items = pd.DataFrame(rows)
items["예측 주문금액"] = model.predict(items[["price", "quantity"]])
items["계산값(가격×수량)"] = items["price"] * items["quantity"]

st.subheader("예측 결과")
total_pred = items["예측 주문금액"].sum()
total_calc = items["계산값(가격×수량)"].sum()

c1, c2 = st.columns(2)
c1.metric("총 예측 주문 금액", f"{total_pred:,.2f}")
c2.metric(
    "총 계산값 (가격×수량)",
    f"{total_calc:,.2f}",
    delta=f"{total_pred - total_calc:,.2f} (오차)",
)

st.dataframe(
    items.rename(columns={"price": "가격", "quantity": "수량"}).set_index("품목"),
    use_container_width=True,
)

# ---------- 모델 정보 ----------
with st.expander("모델 성능 및 계수"):
    m1, m2 = st.columns(2)
    m1.metric("R² (테스트)", f"{r2:.4f}")
    m2.metric("MAE (테스트)", f"{mae:,.2f}")
    st.write(
        f"절편: {model.intercept_:,.4f} / 가격 계수: {model.coef_[0]:,.4f} / 수량 계수: {model.coef_[1]:,.4f}"
    )
    st.caption(
        "주문 금액은 가격×수량(곱셈)이라 순수 선형 모델로는 오차가 생길 수 있습니다."
    )

with st.expander("데이터 미리보기"):
    st.write(f"행 수: {len(df):,}")
    st.dataframe(df[["price", "quantity", "order_amount"]].head(50))
