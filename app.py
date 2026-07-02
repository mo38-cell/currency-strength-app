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

TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
TENKAN_SLOPE_LOOKBACK = 3
STOCH_PERIOD = 14
STOCH_UPPER = 93
STOCH_LOWER = 7

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        font-size: 0.9rem;
        opacity: 0.7;
        margin-bottom: 1rem;
    }
    .watch {
        text-align: center;
        border: 1px solid rgba(128,128,128,0.25);
        background: rgba(128,128,128,0.08);
        border-radius: 18px;
        padding: 18px;
        margin: 12px 0 18px 0;
    }
    .watch-label {
        font-size: 0.85rem;
        opacity: 0.7;
    }
    .watch-pair {
        font-size: 2.2rem;
        font-weight: 900;
        margin: 4px 0;
    }
    .watch-dir {
        font-size: 1.25rem;
        font-weight: 700;
    }
    .rank-card {
        border: 1px solid rgba(128,128,128,0.20);
        background: rgba(128,128,128,0.06);
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .rank-line {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .rank-name {
        font-size: 1.15rem;
        font-weight: 800;
    }
    .rank-score {
        font-size: 1.05rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data(ttl=900)
def get_data(symbols, period, interval, resample_4h=False):
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

    if resample_4h:
        high = high.resample("4h").max()
        low = low.resample("4h").min()
        close = close.resample("4h").last()

    return high, low, close


def make_ohlc(high, low, close, symbol):
    if symbol not in close.columns:
        return None

    df = pd.concat(
        [high[symbol], low[symbol], close[symbol]],
        axis=1
    ).dropna()

    df.columns = ["High", "Low", "Close"]

    if len(df) < KIJUN_PERIOD + TENKAN_SLOPE_LOOKBACK + 1:
        return None

    return df


def ichimoku_score(df):
    tenkan = (
        df["High"].rolling(TENKAN_PERIOD).max()
        + df["Low"].rolling(TENKAN_PERIOD).min()
    ) / 2

    kijun = (
        df["High"].rolling(KIJUN_PERIOD).max()
        + df["Low"].rolling(KIJUN_PERIOD).min()
    ) / 2

    latest_close = df["Close"].iloc[-1]
    latest_tenkan = tenkan.iloc[-1]
    latest_kijun = kijun.iloc[-1]

    if pd.isna(latest_tenkan) or pd.isna(latest_kijun):
        return None

    score = 0

    if latest_close > latest_tenkan:
        score += 1
    elif latest_close < latest_tenkan:
        score -= 1

    tenkan_now = tenkan.iloc[-1]
    tenkan_past = tenkan.iloc[-1 - TENKAN_SLOPE_LOOKBACK]

    if not pd.isna(tenkan_past):
        if tenkan_now > tenkan_past:
            score += 1
        elif tenkan_now < tenkan_past:
            score -= 1

    if latest_close > latest_kijun:
        score += 1
    elif latest_close < latest_kijun:
        score -= 1

    return score


def stochastic_k(df):
    lowest_low = df["Low"].rolling(STOCH_PERIOD).min()
    highest_high = df["High"].rolling(STOCH_PERIOD).max()
    denominator = highest_high - lowest_low

    k = (df["Close"] - lowest_low) / denominator * 100
    k = k.replace([np.inf, -np.inf], np.nan)

    return k.iloc[-1]


def calculate_strength(high, low, close):
    strength = {ccy: 0.0 for ccy in currencies}
    counts = {ccy: 0 for ccy in currencies}
    pair_rows = []

    for p in pair_defs:
        df = make_ohlc(high, low, close, p["symbol"])

        if df is None:
            continue

        score = ichimoku_score(df)

        if score is None:
            continue

        stoch = stochastic_k(df)

        strength[p["base"]] += score
        strength[p["quote"]] -= score
        counts[p["base"]] += 1
        counts[p["quote"]] += 1

        pair_rows.append({
            "Pair": p["pair"],
            "Base": p["base"],
            "Quote": p["quote"],
            "Score": score,
            "Stoch": stoch
        })

    ranking = pd.Series({
        ccy: strength[ccy] / counts[ccy]
        for ccy in currencies
        if counts[ccy] > 0
    }).sort_values(ascending=False)

    pair_df = pd.DataFrame(pair_rows)

    return ranking, pair_df


def find_watch_pair(strongest, weakest):
    for p in pair_defs:
        if p["base"] == strongest and p["quote"] == weakest:
            return p["pair"], "買い目線"

        if p["base"] == weakest and p["quote"] == strongest:
            return p["pair"], "売り目線"

    return f"{strongest}{weakest}", "監視"


def get_warning_mark(pair_df, watch_pair):
    if pair_df.empty:
        return ""

    row = pair_df[pair_df["Pair"] == watch_pair]

    if row.empty:
        return ""

    stoch = row["Stoch"].iloc[0]

    if pd.isna(stoch):
        return ""

    if stoch >= STOCH_UPPER or stoch <= STOCH_LOWER:
        return " ⚠️"

    return ""


st.markdown('<div class="title">💱 Currency Strength</div>', unsafe_allow_html=True)

timeframe = st.radio(
    "時間足",
    ["5分足", "1時間足", "4時間足"],
    horizontal=True
)

if timeframe == "5分足":
    interval = "5m"
    period = "5d"
    resample_4h = False
    note = "5分足"

elif timeframe == "1時間足":
    interval = "1h"
    period = "60d"
    resample_4h = False
    note = "1時間足"

else:
    interval = "1h"
    period = "180d"
    resample_4h = True
    note = "4時間足"

st.markdown(f'<div class="subtitle">{note}</div>', unsafe_allow_html=True)

col_refresh, col_time = st.columns([1, 2])

with col_refresh:
    if st.button("更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

with col_time:
    st.caption(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

symbols = [p["symbol"] for p in pair_defs]

try:
    high, low, close = get_data(symbols, period, interval, resample_4h)
    ranking, pair_df = calculate_strength(high, low, close)

    if ranking.empty:
        st.error("データを取得できませんでした。")
        st.stop()

    strongest = ranking.index[0]
    weakest = ranking.index[-1]
    watch_pair, direction = find_watch_pair(strongest, weakest)
    warning = get_warning_mark(pair_df, watch_pair)

    st.markdown(
        f"""
        <div class="watch">
            <div class="watch-label">Main Watch Pair</div>
            <div class="watch-pair">{watch_pair}{warning}</div>
            <div class="watch-dir">{direction}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("通貨ランキング")

    for i, (ccy, score) in enumerate(ranking.items(), start=1):
        st.markdown(
            f"""
            <div class="rank-card">
                <div class="rank-line">
                    <div class="rank-name">{i}. {ccy}</div>
                    <div class="rank-score">{score:.2f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    valid_close = close.dropna(how="all")

    if not valid_close.empty:
        last_time = valid_close.index[-1]

        try:
            if last_time.tzinfo is not None:
                last_time = last_time.tz_convert("Asia/Tokyo")
        except Exception:
            pass

        st.caption(f"Last Data: {last_time.strftime('%Y-%m-%d %H:%M')}")

except Exception as e:
    st.error("エラーが発生しました。")
    st.write(e)
