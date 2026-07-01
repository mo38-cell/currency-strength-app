import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Currency Strength",
    page_icon="💱",
    layout="centered"
)

currencies = ["USD", "JPY", "EUR", "AUD", "GBP"]

pair_defs = [
    {"symbol": "USDJPY=X", "pair": "USDJPY", "base": "USD", "quote": "JPY"},
    {"symbol": "EURUSD=X", "pair": "EURUSD", "base": "EUR", "quote": "USD"},
    {"symbol": "AUDUSD=X", "pair": "AUDUSD", "base": "AUD", "quote": "USD"},
    {"symbol": "GBPUSD=X", "pair": "GBPUSD", "base": "GBP", "quote": "USD"},
    {"symbol": "EURJPY=X", "pair": "EURJPY", "base": "EUR", "quote": "JPY"},
    {"symbol": "AUDJPY=X", "pair": "AUDJPY", "base": "AUD", "quote": "JPY"},
    {"symbol": "GBPJPY=X", "pair": "GBPJPY", "base": "GBP", "quote": "JPY"},
    {"symbol": "EURAUD=X", "pair": "EURAUD", "base": "EUR", "quote": "AUD"},
    {"symbol": "EURGBP=X", "pair": "EURGBP", "base": "EUR", "quote": "GBP"},
    {"symbol": "GBPAUD=X", "pair": "GBPAUD", "base": "GBP", "quote": "AUD"},
]

kijun_lookback = 5

st.markdown(
    """
    <style>
    .main-title {
        font-size: 1.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 0.9rem;
        text-align: center;
        opacity: 0.75;
        margin-bottom: 1.0rem;
    }
    .watch-card {
        border-radius: 18px;
        padding: 18px;
        margin: 12px 0px;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.08);
        text-align: center;
    }
    .watch-pair {
        font-size: 2.1rem;
        font-weight: 900;
        margin-bottom: 4px;
    }
    .watch-direction {
        font-size: 1.2rem;
        font-weight: 700;
    }
    .currency-card {
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 8px;
        border: 1px solid rgba(128,128,128,0.20);
        background: rgba(128,128,128,0.06);
    }
    .currency-line {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .currency-rank {
        font-size: 1.15rem;
        font-weight: 800;
    }
    .currency-score {
        font-size: 1.05rem;
        font-weight: 700;
    }
    .small-note {
        font-size: 0.8rem;
        opacity: 0.7;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data(ttl=900)
def get_data(symbols, period, interval):
    data = yf.download(
        tickers=symbols,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=True
    )

    if isinstance(data.columns, pd.MultiIndex):
        high = data["High"]
        low = data["Low"]
        close = data["Close"]
    else:
        high = data[["High"]]
        low = data[["Low"]]
        close = data[["Close"]]

    return high, low, close


def ichimoku_score(df):
    tenkan = (
        df["High"].rolling(9).max()
        + df["Low"].rolling(9).min()
    ) / 2

    kijun = (
        df["High"].rolling(26).max()
        + df["Low"].rolling(26).min()
    ) / 2

    senkou_a = ((tenkan + kijun) / 2).shift(26)

    senkou_b = (
        df["High"].rolling(52).max()
        + df["Low"].rolling(52).min()
    ) / 2
    senkou_b = senkou_b.shift(26)

    latest_close = df["Close"].iloc[-1]
    latest_tenkan = tenkan.iloc[-1]
    latest_kijun = kijun.iloc[-1]
    latest_senkou_a = senkou_a.iloc[-1]
    latest_senkou_b = senkou_b.iloc[-1]

    needed = [
        latest_tenkan,
        latest_kijun,
        latest_senkou_a,
        latest_senkou_b,
    ]

    if any(pd.isna(x) for x in needed):
        return None

    cloud_upper = max(latest_senkou_a, latest_senkou_b)
    cloud_lower = min(latest_senkou_a, latest_senkou_b)

    score = 0
    details = []

    if latest_close > latest_kijun:
        score += 1
        details.append("終値＞基準線 +1")
    elif latest_close < latest_kijun:
        score -= 1
        details.append("終値＜基準線 -1")

    if latest_tenkan > latest_kijun:
        score += 1
        details.append("転換線＞基準線 +1")
    elif latest_tenkan < latest_kijun:
        score -= 1
        details.append("転換線＜基準線 -1")

    if latest_close > cloud_upper:
        score += 2
        details.append("終値＞雲上限 +2")
    elif latest_close < cloud_lower:
        score -= 2
        details.append("終値＜雲下限 -2")
    else:
        details.append("終値は雲の中 0")

    if len(kijun.dropna()) > kijun_lookback:
        kijun_now = kijun.iloc[-1]
        kijun_past = kijun.iloc[-1 - kijun_lookback]

        if not pd.isna(kijun_past):
            if kijun_now > kijun_past:
                score += 1
                details.append("基準線上向き +1")
            elif kijun_now < kijun_past:
                score -= 1
                details.append("基準線下向き -1")
            else:
                details.append("基準線横ばい 0")

    return score, " / ".join(details)


def calculate_strength(high, low, close):
    strength = {ccy: 0.0 for ccy in currencies}
    counts = {ccy: 0 for ccy in currencies}
    pair_rows = []

    for p in pair_defs:
        symbol = p["symbol"]
        pair = p["pair"]
        base = p["base"]
        quote = p["quote"]

        if symbol not in close.columns:
            continue

        df = pd.concat(
            [
                high[symbol],
                low[symbol],
                close[symbol],
            ],
            axis=1
        ).dropna()

        df.columns = ["High", "Low", "Close"]

        if len(df) < 80:
            continue

        result = ichimoku_score(df)

        if result is None:
            continue

        score, details = result

        strength[base] += score
        strength[quote] -= score
        counts[base] += 1
        counts[quote] += 1

        pair_rows.append({
            "Pair": pair,
            "Base": base,
            "Quote": quote,
            "Ichimoku Score": score,
            "Details": details
        })

    strength_avg = {}

    for ccy in currencies:
        if counts[ccy] > 0:
            strength_avg[ccy] = strength[ccy] / counts[ccy]
        else:
            strength_avg[ccy] = np.nan

    ranking = pd.Series(strength_avg).dropna().sort_values(ascending=False)
    pair_df = pd.DataFrame(pair_rows)

    if not pair_df.empty:
        pair_df = pair_df.sort_values("Ichimoku Score", ascending=False)

    return ranking, pair_df


def find_watch_pair(strongest, weakest):
    for p in pair_defs:
        if p["base"] == strongest and p["quote"] == weakest:
            return p["pair"], "買い目線"

        if p["base"] == weakest and p["quote"] == strongest:
            return p["pair"], "売り目線"

    return f"{strongest}{weakest}", "監視"


def strength_label(score):
    if score >= 2.0:
        return "かなり強い"
    elif score >= 1.0:
        return "強い"
    elif score > -1.0:
        return "中立"
    elif score > -2.0:
        return "弱い"
    else:
        return "かなり弱い"


st.markdown(
    '<div class="main-title">💱 Currency Strength</div>',
    unsafe_allow_html=True
)

timeframe = st.radio("時間足", ["1時間足", "15分足"], horizontal=True)

if timeframe == "1時間足":
    interval = "1h"
    period = "30d"
    timeframe_note = "1時間足 / 一目均衡表スコア"
else:
    interval = "15m"
    period = "5d"
    timeframe_note = "15分足 / 一目均衡表スコア"

st.markdown(
    f'<div class="sub-title">{timeframe_note}</div>',
    unsafe_allow_html=True
)

refresh_col, time_col = st.columns([1, 2])

with refresh_col:
    if st.button("更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with time_col:
    st.caption(f"Checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

symbols = [p["symbol"] for p in pair_defs]

try:
    high, low, close = get_data(symbols, period, interval)
    ranking, pair_df = calculate_strength(high, low, close)

    if ranking.empty:
        st.error("データを取得できませんでした。時間をおいて再度試してください。")
        st.stop()

    strongest = ranking.index[0]
    weakest = ranking.index[-1]
    watch_pair, direction = find_watch_pair(strongest, weakest)

    st.markdown(
        f"""
        <div class="watch-card">
            <div class="small-note">Main Watch Pair</div>
            <div class="watch-pair">{watch_pair}</div>
            <div class="watch-direction">{direction}</div>
            <div class="small-note">{strongest} strongest / {weakest} weakest</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Strongest", strongest, f"{ranking.iloc[0]:.2f}")

    with col2:
        st.metric("Weakest", weakest, f"{ranking.iloc[-1]:.2f}")

    st.subheader("通貨ランキング")

    for i, (ccy, score) in enumerate(ranking.items(), start=1):
        label = strength_label(score)

        st.markdown(
            f"""
            <div class="currency-card">
                <div class="currency-line">
                    <div class="currency-rank">{i}. {ccy}</div>
                    <div class="currency-score">{score:.2f}</div>
                </div>
                <div class="small-note">{label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("通貨強弱チャート")

    chart_df = ranking.rename("Strength").to_frame()
    st.bar_chart(chart_df)

    st.subheader("ペア別 一目スコア")

    if not pair_df.empty:
        display_pair_df = pair_df[["Pair", "Ichimoku Score"]].copy()
        display_pair_df["Ichimoku Score"] = display_pair_df["Ichimoku Score"].round(2)

        st.dataframe(
            display_pair_df,
            hide_index=True,
            use_container_width=True
        )

        with st.expander("ペア別スコア内訳"):
            detail_df = pair_df[["Pair", "Ichimoku Score", "Details"]].copy()
            st.dataframe(
                detail_df,
                hide_index=True,
                use_container_width=True
            )

    valid_close = close.dropna(how="all")

    if not valid_close.empty:
        last_time = valid_close.index[-1]

        try:
            if last_time.tzinfo is not None:
                last_time = last_time.tz_convert("Asia/Tokyo")
        except Exception:
            pass

        st.caption(f"Last data: {last_time.strftime('%Y-%m-%d %H:%M')}")

    st.caption(
        "これは売買指示ではなく、一目均衡表スコアに基づく通貨強弱の可視化です。"
    )

except Exception as e:
    st.error("エラーが発生しました。")
    st.write(e)
