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

# ---------- 입력 슬라이더 ----------
st.subheader("값 입력")
p_min, p_max = float(df["price"].min()), float(df["price"].max())
q_min, q_max = int(df["quantity"].min()), int(df["quantity"].max())

price = st.slider(
    "가격",
    min_value=p_min,
    max_value=p_max,
    value=float(df["price"].median()),
    step=max((p_max - p_min) / 1000, 0.01),
)
quantity = st.slider(
    "수량",
    min_value=q_min,
    max_value=q_max,
    value=int(df["quantity"].median()),
    step=1,
)

# ---------- 예측 ----------
input_df = pd.DataFrame({"price": [price], "quantity": [quantity]})
predicted = float(model.predict(input_df)[0])
actual_calc = price * quantity

st.subheader("예측 결과")
c1, c2 = st.columns(2)
c1.metric("예측 주문 금액", f"{predicted:,.2f}")
c2.metric("실제 계산값 (가격×수량)", f"{actual_calc:,.2f}", delta=f"{predicted - actual_calc:,.2f} (오차)")

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

# ---------- 시각화 ----------
with st.expander("수량별 예측 곡선"):
    qs = np.arange(q_min, q_max + 1)
    curve = pd.DataFrame({"price": price, "quantity": qs})
    curve["예측 주문금액"] = model.predict(curve[["price", "quantity"]])
    curve["실제(가격×수량)"] = price * qs
    st.line_chart(curve.set_index("quantity")[["예측 주문금액", "실제(가격×수량)"]])
