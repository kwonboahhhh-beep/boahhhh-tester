import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DEFAULT_CSV = "ecommerce_orders_clean.csv"
FEATURES = ["price", "quantity"]
OUTLIER_IQR_K = 1.0  # 이상치 제외 기준 (Q3 + k*IQR 초과 / Q1 - k*IQR 미만), 작을수록 엄격

st.set_page_config(page_title="주문 금액 예측기", page_icon="🛒", layout="centered")
st.title("🛒 주문 금액 예측기")
st.caption("가격(price)과 수량(quantity)으로 주문 금액을 예측하는 모델 (선형 회귀 vs 그래디언트 부스팅)")


def iqr_bounds(series: pd.Series, k: float):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


@st.cache_data
def load_data(source):
    """데이터 전처리 파이프라인. (정제된 df, 단계별 로그 표)를 반환한다."""
    df = pd.read_csv(source)
    log = [("원본 데이터", len(df), 0)]

    def record(name, before, after):
        log.append((name, after, before - after))

    # 1) 가격·수량을 숫자로 변환 (변환 불가 값은 결측 처리)
    for col in ["price", "quantity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2) 결측치 제거
    n = len(df)
    df = df.dropna(subset=["price", "quantity"])
    record("결측치 제거 (가격·수량)", n, len(df))

    # 3) 중복 행 제거
    n = len(df)
    df = df.drop_duplicates()
    if "order_id" in df.columns:
        df = df.drop_duplicates(subset="order_id")
    record("중복 제거", n, len(df))

    # 4) 비정상 값 제거 (가격·수량이 0 이하)
    n = len(df)
    df = df[(df["price"] > 0) & (df["quantity"] > 0)]
    record("비정상 값 제거 (가격·수량 ≤ 0)", n, len(df))

    # 5) 카테고리 문자열 공백 정리
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip()

    # 6) 주문금액 = 가격 x 수량
    df = df.copy()
    df["order_amount"] = df["price"] * df["quantity"]

    # 7) 이상치 제거 (IQR 기준): 가격, 수량, 주문금액
    for col, label in [("price", "가격"), ("quantity", "수량"), ("order_amount", "주문금액")]:
        n = len(df)
        low, high = iqr_bounds(df[col], OUTLIER_IQR_K)
        df = df[df[col].between(low, high)]
        record(f"이상치 제거 ({label}, {low:,.1f} ~ {high:,.1f})", n, len(df))

    log_df = pd.DataFrame(log, columns=["단계", "남은 행 수", "제거된 행 수"])
    return df.reset_index(drop=True), log_df


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

df, prep_log = load_data(source)

missing = {"price", "quantity"} - set(df.columns)
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}")
    st.stop()


# ---------- 모델 학습 (학습 50% / 테스트 50%) ----------
@st.cache_resource
def train_models(data: pd.DataFrame):
    X = data[FEATURES]
    y = data["order_amount"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    # 기준 모델: 선형 회귀
    base = LinearRegression().fit(X_train, y_train)
    base_pred = base.predict(X_test)

    # 개선 모델: 그래디언트 부스팅 (가격×수량 같은 비선형 관계를 학습)
    improved = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.1, max_depth=3, random_state=42
    ).fit(X_train, y_train)
    pred = improved.predict(X_test)

    comparison = pd.DataFrame(
        {
            "모델": ["기준: 선형 회귀", "개선: 그래디언트 부스팅"],
            "R²": [r2_score(y_test, base_pred), r2_score(y_test, pred)],
            "MAE": [
                mean_absolute_error(y_test, base_pred),
                mean_absolute_error(y_test, pred),
            ],
        }
    ).set_index("모델")

    test_result = pd.DataFrame(
        {
            "price": X_test["price"].values,
            "quantity": X_test["quantity"].values,
            "actual": y_test,
            "predicted": pred,
        }
    )
    return (
        improved,
        r2_score(y_test, pred),
        mean_absolute_error(y_test, pred),
        len(X_train),
        len(X_test),
        test_result,
        comparison,
    )


model, r2, mae, n_train, n_test, test_result, comparison = train_models(df)

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
items["예측 주문금액"] = model.predict(items[FEATURES])
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

# ---------- 전처리 결과 ----------
with st.expander("🧹 데이터 전처리 결과", expanded=False):
    st.dataframe(prep_log.set_index("단계"), use_container_width=True)
    st.caption(
        f"원본 {prep_log.iloc[0, 1]:,}행 → 전처리 후 {len(df):,}행 "
        f"(이상치 기준: Q1−{OUTLIER_IQR_K:g}×IQR ~ Q3+{OUTLIER_IQR_K:g}×IQR)"
    )

# ---------- 모델 정보 ----------
with st.expander("모델 성능 (테스트 데이터 기준)", expanded=True):
    st.write(f"데이터 분할: 학습용 {n_train:,}행(50%) / 테스트용 {n_test:,}행(50%)")
    m1, m2 = st.columns(2)
    m1.metric("R² (테스트)", f"{r2:.4f}")
    m2.metric("MAE (테스트)", f"{mae:,.2f}")
    st.dataframe(comparison.style.format({"R²": "{:.4f}", "MAE": "{:,.2f}"}))
    st.caption(
        "주문 금액은 가격×수량(곱셈)이라 선형 회귀는 오차가 큽니다. "
        "그래디언트 부스팅은 비선형 관계를 학습해 오차를 줄이며, "
        "예측·그래프·성능 평가에는 개선 모델이 사용됩니다."
    )

# ---------- 그래프 ----------
st.subheader("📊 데이터 분석 그래프")

# 1) 카테고리별 총 주문 금액
st.markdown("**1. 카테고리별 총 주문 금액**")
if "category" in df.columns:
    cat_total = (
        df.groupby("category", as_index=False)["order_amount"]
        .sum()
        .sort_values("order_amount", ascending=False)
    )
    fig_cat = px.bar(
        cat_total,
        x="category",
        y="order_amount",
        color="category",
        text="order_amount",
        labels={"category": "카테고리", "order_amount": "총 주문 금액"},
    )
    fig_cat.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_cat.update_layout(showlegend=False, yaxis_tickformat=",")
    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.warning("'category' 컬럼이 없어 카테고리별 그래프를 그릴 수 없습니다.")

# 2) 테스트 데이터: 실제 주문 금액 vs 예측값
st.markdown("**2. 테스트 데이터: 실제 주문 금액 vs 예측값**")
fig_sc = px.scatter(
    test_result,
    x="actual",
    y="predicted",
    opacity=0.45,
    hover_data={"price": ":,.2f", "quantity": True, "actual": ":,.2f", "predicted": ":,.2f"},
    labels={
        "actual": "실제 주문 금액",
        "predicted": "예측 주문 금액",
        "price": "가격",
        "quantity": "수량",
    },
)
# 완벽한 예측선 (y = x)
axis_min = float(min(test_result["actual"].min(), test_result["predicted"].min()))
axis_max = float(max(test_result["actual"].max(), test_result["predicted"].max()))
fig_sc.add_shape(
    type="line",
    x0=axis_min,
    y0=axis_min,
    x1=axis_max,
    y1=axis_max,
    line=dict(color="red", dash="dash"),
)
fig_sc.add_annotation(
    x=axis_max,
    y=axis_max,
    text="완벽한 예측 (y = x)",
    showarrow=False,
    xanchor="right",
    yanchor="bottom",
    font=dict(color="red"),
)
fig_sc.update_layout(xaxis_tickformat=",", yaxis_tickformat=",")
st.plotly_chart(fig_sc, use_container_width=True)
st.caption(
    f"테스트 데이터 {n_test:,}건 · 점이 빨간 점선에 가까울수록 예측이 정확합니다. "
    f"(R² {r2:.4f}, MAE {mae:,.2f})"
)

# 3) 가격 구간(5개)별 평균 수량
st.markdown("**3. 가격 구간별 평균 수량**")
price_bins = pd.cut(df["price"], bins=5)  # 가격 범위를 5개 구간으로 균등 분할
bin_labels = [f"{iv.left:,.0f} ~ {iv.right:,.0f}" for iv in price_bins.cat.categories]
price_bins = price_bins.cat.rename_categories(bin_labels)
qty_by_price = (
    df.assign(price_bin=price_bins)
    .groupby("price_bin", observed=False)["quantity"]
    .agg(avg_quantity="mean", count="size")
    .reset_index()
)
fig_bin = px.bar(
    qty_by_price,
    x="price_bin",
    y="avg_quantity",
    color="price_bin",
    text="avg_quantity",
    hover_data={"count": ":,", "avg_quantity": ":.3f", "price_bin": False},
    labels={
        "price_bin": "가격 구간",
        "avg_quantity": "평균 수량",
        "count": "주문 건수",
    },
)
fig_bin.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig_bin.update_layout(showlegend=False)
# 세로축을 0부터 시작하지 않고 값이 몰린 구간만 확대해 차이를 크게 보이게 함
_y_min = float(qty_by_price["avg_quantity"].min())
_y_max = float(qty_by_price["avg_quantity"].max())
_pad = max((_y_max - _y_min) * 0.3, 0.01)
fig_bin.update_yaxes(range=[_y_min - _pad, _y_max + _pad])
st.plotly_chart(fig_bin, use_container_width=True)
st.caption("⚠️ 차이를 잘 보이게 하려고 세로축을 0이 아닌 값에서 시작했습니다. 막대 길이 비율은 실제 비율과 다릅니다.")

_hi = qty_by_price.loc[qty_by_price["avg_quantity"].idxmax()]
_lo = qty_by_price.loc[qty_by_price["avg_quantity"].idxmin()]
_gap = (_hi["avg_quantity"] - _lo["avg_quantity"]) / _lo["avg_quantity"] * 100
st.caption(
    f"💡 평균 수량은 가격 {_hi['price_bin']} 구간이 {_hi['avg_quantity']:.2f}개로 가장 많고, "
    f"{_lo['price_bin']} 구간이 {_lo['avg_quantity']:.2f}개로 가장 적으며 "
    f"차이는 {_gap:.1f}%입니다."
)

# 4) 고객 유형별 평균 주문 금액
st.markdown("**4. 고객 유형별 평균 주문 금액**")
if "customer_segment" in df.columns:
    seg_avg = (
        df.groupby("customer_segment", as_index=False)
        .agg(avg_amount=("order_amount", "mean"), count=("order_amount", "size"))
        .sort_values("avg_amount", ascending=False)
    )
    fig_seg = px.bar(
        seg_avg,
        x="customer_segment",
        y="avg_amount",
        color="customer_segment",
        text="avg_amount",
        hover_data={"count": ":,", "avg_amount": ":,.2f", "customer_segment": False},
        labels={
            "customer_segment": "고객 유형",
            "avg_amount": "평균 주문 금액",
            "count": "주문 건수",
        },
    )
    fig_seg.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
    fig_seg.update_layout(showlegend=False, yaxis_tickformat=",")
    fig_seg.update_yaxes(rangemode="tozero")
    st.plotly_chart(fig_seg, use_container_width=True)
else:
    st.warning("'customer_segment' 컬럼이 없어 고객 유형별 그래프를 그릴 수 없습니다.")

# ---------- 모델 성능 평가 (테스트 데이터에서 무작위 100개) ----------
st.subheader("🧪 모델 성능 평가")
SAMPLE_N = 100

if "eval_seed" not in st.session_state:
    st.session_state.eval_seed = 0
if "show_full_table" not in st.session_state:
    st.session_state.show_full_table = False


def redraw_sample():
    st.session_state.eval_seed += 1


def toggle_full_table():
    st.session_state.show_full_table = not st.session_state.show_full_table


n_sample = min(SAMPLE_N, len(test_result))
sample = test_result.sample(
    n=n_sample, random_state=st.session_state.eval_seed
).reset_index(drop=True)

eval_df = pd.DataFrame(
    {
        "실제값": sample["actual"],
        "예측값": sample["predicted"],
    }
)
eval_df["오차"] = eval_df["실제값"] - eval_df["예측값"]  # 실제 - 예측
eval_df["절댓값"] = eval_df["오차"].abs()  # 절대 오차
eval_df["제곱"] = eval_df["오차"] ** 2  # 제곱 오차
eval_df.index = eval_df.index + 1
eval_df.index.name = "번호"

s_mae = mean_absolute_error(eval_df["실제값"], eval_df["예측값"])
s_mse = mean_squared_error(eval_df["실제값"], eval_df["예측값"])
s_r2 = r2_score(eval_df["실제값"], eval_df["예측값"])

e1, e2, e3 = st.columns(3)
e1.metric("MAE (평균 절대 오차)", f"{s_mae:,.2f}")
e2.metric("MSE (평균 제곱 오차)", f"{s_mse:,.2f}")
e3.metric("R²", f"{s_r2:.4f}")

fmt = {c: "{:,.2f}" for c in eval_df.columns}

t1, t2, t3 = st.columns([1, 1, 3])
t1.button("🔄 다시 뽑기", on_click=redraw_sample, use_container_width=True)
t2.button(
    "📂 전체 접기" if st.session_state.show_full_table else "📂 전체 보기",
    on_click=toggle_full_table,
    use_container_width=True,
)
t3.markdown(
    f"**테스트 데이터에서 무작위 {n_sample}개** (뽑기 #{st.session_state.eval_seed + 1})"
)

if st.session_state.show_full_table:
    st.caption(f"전체 {n_sample}행")
    st.dataframe(eval_df.style.format(fmt), use_container_width=True, height=400)
else:
    st.caption(f"앞 10행만 표시 (전체 {n_sample}행)")
    st.dataframe(eval_df.head(10).style.format(fmt), use_container_width=True)
st.caption("오차 = 실제값 − 예측값 · 절댓값 = |오차| · 제곱 = 오차²")

with st.expander("데이터 미리보기"):
    st.write(f"행 수: {len(df):,}")
    st.dataframe(df[["price", "quantity", "order_amount"]].head(50))
