import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo


# =========================================================
# Page settings
# =========================================================

st.set_page_config(
    page_title="Currency Strength",
    page_icon="💱",
    layout="centered"
)


# =========================================================
# Currency settings
# =========================================================

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


# =========================================================
# Indicators
# =========================================================

KIJUN_PERIOD = 26

STOCH_K_PERIOD = 14
STOCH_SLOWING = 5
STOCH_D_PERIOD = 3

STOCH_UPPER = 93
STOCH_LOWER = 7


# =========================================================
# Download
# =========================================================

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

    if data.empty:
        return None, None, None

    if isinstance(data.columns, pd.MultiIndex):

        high = data["High"].copy()
        low = data["Low"].copy()
        close = data["Close"].copy()

    else:

        high = data[["High"]].copy()
        low = data[["Low"]].copy()
        close = data[["Close"]].copy()

    if resample_4h:

        high = high.resample("4h").max()
        low = low.resample("4h").min()
        close = close.resample("4h").last()

    return high, low, close


# =========================================================
# OHLC
# =========================================================

def make_ohlc(high, low, close, symbol):

    if high is None or low is None or close is None:
        return None

    if symbol not in close.columns:
        return None

    df = pd.concat(
        [
            high[symbol],
            low[symbol],
            close[symbol]
        ],
        axis=1
    ).dropna()

    df.columns = [
        "High",
        "Low",
        "Close"
    ]

    min_len = max(
        KIJUN_PERIOD,
        STOCH_K_PERIOD
        + STOCH_SLOWING
        + STOCH_D_PERIOD
    )

    if len(df) < min_len:
        return None

    return df


# =========================================================
# Kijun
# =========================================================

def kijun_score(df):

    kijun = (
        df["High"]
        .rolling(KIJUN_PERIOD)
        .max()
        +
        df["Low"]
        .rolling(KIJUN_PERIOD)
        .min()
    ) / 2

    latest_close = df["Close"].iloc[-1]
    latest_kijun = kijun.iloc[-1]

    if pd.isna(latest_kijun):
        return None

    if latest_close > latest_kijun:
        return 1

    if latest_close < latest_kijun:
        return -1

    return 0


# =========================================================
# Slow Stochastic
# =========================================================

def stochastic_slow(df):

    lowest_low = (
        df["Low"]
        .rolling(STOCH_K_PERIOD)
        .min()
    )

    highest_high = (
        df["High"]
        .rolling(STOCH_K_PERIOD)
        .max()
    )

    denominator = highest_high - lowest_low

    fast_k = (
        (df["Close"] - lowest_low)
        / denominator
        * 100
    )

    fast_k = fast_k.replace(
        [np.inf, -np.inf],
        np.nan
    )

    slow_k = (
        fast_k
        .rolling(STOCH_SLOWING)
        .mean()
    )

    slow_d = (
        slow_k
        .rolling(STOCH_D_PERIOD)
        .mean()
    )

    return (
        slow_k.iloc[-1],
        slow_d.iloc[-1]
    )


# =========================================================
# Currency strength
# =========================================================

def calculate_strength(high, low, close):

    strength = {
        ccy: 0
        for ccy in currencies
    }

    counts = {
        ccy: 0
        for ccy in currencies
    }

    pair_rows = []

    for p in pair_defs:

        df = make_ohlc(
            high,
            low,
            close,
            p["symbol"]
        )

        if df is None:
            continue

        score = kijun_score(df)

        if score is None:
            continue

        stoch_k, stoch_d = stochastic_slow(df)

        # 基準線のみで通貨強弱を評価
        strength[p["base"]] += score
        strength[p["quote"]] -= score

        counts[p["base"]] += 1
        counts[p["quote"]] += 1

        pair_rows.append(
            {
                "Pair": p["pair"],
                "Base": p["base"],
                "Quote": p["quote"],
                "Score": score,
                "StochK": stoch_k,
                "StochD": stoch_d
            }
        )

    ranking = pd.Series(
        {
            ccy: strength[ccy] / counts[ccy]
            for ccy in currencies
            if counts[ccy] > 0
        }
    ).sort_values(
        ascending=False,
        kind="stable"
    )

    pair_df = pd.DataFrame(pair_rows)

    return ranking, pair_df


# =========================================================
# Watch pair
# =========================================================

def find_watch_pair(strongest, weakest):

    for p in pair_defs:

        if (
            p["base"] == strongest
            and p["quote"] == weakest
        ):

            return (
                p["pair"],
                "買い目線"
            )

        if (
            p["base"] == weakest
            and p["quote"] == strongest
        ):

            return (
                p["pair"],
                "売り目線"
            )

    return (
        f"{strongest}{weakest}",
        "監視"
    )


# =========================================================
# Stochastic warning
# =========================================================

def get_stoch_info(
    pair_df,
    watch_pair,
    direction
):

    if pair_df.empty:
        return None, ""

    row = pair_df[
        pair_df["Pair"] == watch_pair
    ]

    if row.empty:
        return None, ""

    stoch_k = row["StochK"].iloc[0]

    if pd.isna(stoch_k):
        return None, ""

    warning = ""

    # 買い目線なのに買われすぎ
    if (
        direction == "買い目線"
        and stoch_k >= STOCH_UPPER
    ):

        warning = "⚠️ 買われすぎ"

    # 売り目線なのに売られすぎ
    elif (
        direction == "売り目線"
        and stoch_k <= STOCH_LOWER
    ):

        warning = "⚠️ 売られすぎ"

    return stoch_k, warning


# =========================================================
# Header
# =========================================================

st.title("💱 Currency Strength")


# =========================================================
# Timeframe
# =========================================================

timeframe = st.radio(
    "時間足",
    [
        "5分足",
        "1時間足",
        "4時間足"
    ],
    horizontal=True
)


if timeframe == "5分足":

    interval = "5m"
    period = "5d"
    resample_4h = False

elif timeframe == "1時間足":

    interval = "1h"
    period = "60d"
    resample_4h = False

else:

    interval = "1h"
    period = "180d"
    resample_4h = True


# =========================================================
# Refresh
# =========================================================

col1, col2 = st.columns(
    [1, 2]
)

with col1:

    if st.button(
        "更新",
        use_container_width=True
    ):

        st.cache_data.clear()
        st.rerun()


with col2:

    now_jst = datetime.now(
        ZoneInfo("Asia/Tokyo")
    )

    st.caption(
        now_jst.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


# =========================================================
# Main
# =========================================================

symbols = [
    p["symbol"]
    for p in pair_defs
]


try:

    high, low, close = get_data(
        symbols,
        period,
        interval,
        resample_4h
    )


    if (
        high is None
        or low is None
        or close is None
    ):

        st.error(
            "データを取得できませんでした。"
        )

        st.stop()


    ranking, pair_df = calculate_strength(
        high,
        low,
        close
    )


    if ranking.empty:

        st.error(
            "通貨強弱を計算できませんでした。"
        )

        st.stop()


    # =====================================================
    # Strongest / Weakest
    # =====================================================

    strongest = ranking.index[0]
    weakest = ranking.index[-1]


    watch_pair, direction = find_watch_pair(
        strongest,
        weakest
    )


    stoch_k, warning = get_stoch_info(
        pair_df,
        watch_pair,
        direction
    )


    # =====================================================
    # Main Watch Pair
    # =====================================================

    st.subheader(
        "Main Watch Pair"
    )


    with st.container(
        border=True
    ):

        st.markdown(
            f"<h1 style='text-align:center;'>"
            f"{watch_pair}"
            f"</h1>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"<h3 style='text-align:center;'>"
            f"{direction}"
            f"</h3>",
            unsafe_allow_html=True
        )


        if stoch_k is not None:

            st.markdown(
                f"<p style='text-align:center;'>"
                f"Slow Stoch %K: {stoch_k:.1f}"
                f"</p>",
                unsafe_allow_html=True
            )


        if warning:

            st.warning(
                warning
            )


    # =====================================================
    # Ranking
    # =====================================================

    st.subheader(
        "通貨ランキング"
    )


    for i, (
        ccy,
        score
    ) in enumerate(
        ranking.items(),
        start=1
    ):

        left, right = st.columns(
            [3, 1]
        )


        with left:

            st.markdown(
                f"### {i}. {ccy}"
            )


        with right:

            st.metric(
                label="Score",
                value=f"{score:.2f}",
                label_visibility="collapsed"
            )


        st.divider()


    # =====================================================
    # Last Data
    # =====================================================

    valid_close = close.dropna(
        how="all"
    )


    if not valid_close.empty:

        last_time = (
            valid_close.index[-1]
        )


        try:

            if last_time.tzinfo is not None:

                last_time = (
                    last_time
                    .tz_convert(
                        "Asia/Tokyo"
                    )
                )

        except Exception:

            pass


        st.caption(
            "Last Data: "
            +
            last_time.strftime(
                "%Y-%m-%d %H:%M"
            )
        )


except Exception as e:

    st.error(
        "エラーが発生しました。"
    )

    st.exception(e)
