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

ma_window = 20
slope_lookback = 5

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
        close = data["Close"]
    else:
        close = data[["Close"]]
    return close

def calculate_strength(close):
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

        series = close[symbol].dropna()
        if len(series) < ma_window + slope_lookback + 1:
            continue

        ma20 = series.rolling(ma_window).mean().dropna()
        if len(ma20) < slope_lookback + 1:
            continue

        slope = np.log(ma20.iloc[-1] / ma20.iloc[-1 - slope_lookback]) * 100

        strength[base] += slope
        strength[quote] -= slope
        counts[base] += 1
        counts[quote] += 1

        pair_rows.append({
            "Pair": pair,
            "Base": base,
            "Quote": quote,
            "20MA Slope": slope
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
        pair_df = pair_df.sort_values("20MA Slope", ascending=False)

    return ranking, pair_df

def find_watch_pair(strongest, weakest):
    for p in pair_defs:
        if p["base"] == strongest and p["quote"] == weakest:
            return p["pair"], "買い目線"
        if p["base"] == weakest and p["quote"] == strongest:
            return p["pair"], "売り目線"
    return f"{strongest}{weakest}", "監視"

def strength_label(score):
    if score >= 0.08:
        return "かなり強い"
    elif score >= 0.03:
        return "強い"
    elif score > -0.03:
        return "中立"
    elif score > -0.08:
        return "弱い"
    else:
        return "かなり弱い"

st.markdown('<div class="main-title">💱 Currency Strength</div>', unsafe_allow_html=True)

timeframe = st.radio("時間足", ["1時間足", "15分足"], horizontal=True)

if timeframe == "1時間足":
    interval = "1h"
    period = "30d"
    timeframe_note = "1時間足 / 20MA / 5本前比較"
else:
    interval = "15m"
    period = "30d"
    timeframe_note = "15分足 / 20MA / 5本前比較"

st.markdown(f'<div class="sub-title">{timeframe_note}</div>', unsafe_allow_html=True)

# 更新ボタンを上部に配置
refresh_col, time_col = st.columns([1, 2])

with refresh_col:
    if st.button("更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with time_col:
    st.caption(f"Checked: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

symbols = [p["symbol"] for p in pair_defs]

try:
    close = get_data(symbols, period, interval)
    ranking, pair_df = calculate_strength(close)

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
        st.metric("Strongest", strongest, f"{ranking.iloc[0]:.4f}")
    with col2:
        st.metric("Weakest", weakest, f"{ranking.iloc[-1]:.4f}")

    st.subheader("通貨ランキング")
    for i, (ccy, score) in enumerate(ranking.items(), start=1):
        label = strength_label(score)
        st.markdown(
            f"""
            <div class="currency-card">
                <div class="currency-line">
                    <div class="currency-rank">{i}. {ccy}</div>
                    <div class="currency-score">{score:.4f}</div>
                </div>
                <div class="small-note">{label}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("通貨強弱チャート")
    chart_df = ranking.rename("Strength").to_frame()
    st.bar_chart(chart_df)

    st.subheader("ペア別 20MA傾き")
    if not pair_df.empty:
        display_pair_df = pair_df[["Pair", "20MA Slope"]].copy()
        display_pair_df["20MA Slope"] = display_pair_df["20MA Slope"].round(4)
        st.dataframe(display_pair_df, hide_index=True, use_container_width=True)

    valid_close = close.dropna(how="all")
    if not valid_close.empty:
        last_time = valid_close.index[-1]
        try:
            if last_time.tzinfo is not None:
                last_time = last_time.tz_convert("Asia/Tokyo")
        except Exception:
            pass
        st.caption(f"Last data: {last_time.strftime('%Y-%m-%d %H:%M')}")

    st.caption("これは売買指示ではなく、20MAの傾きに基づく通貨強弱の可視化です。")

except Exception as e:
    st.error("エラーが発生しました。")
    st.write(e)
