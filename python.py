import streamlit as st
import pandas as pd
import numpy as np
import requests
import pytz
from datetime import datetime

# =========================
# CẤU HÌNH
# =========================
st.set_page_config(page_title="AI Vàng Thế Giới - Pro", layout="wide")
st.title("🤖 AI chuyên gia phân tích Vàng (XAUUSD)")
st.caption("Tự động lấy dữ liệu từ Binance → Kraken → Yahoo Finance (realtime).")

# =========================
# HÀM CHỈ BÁO
# =========================
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def sma(series, length): return series.rolling(length).mean()

def rsi(close, length=14):
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).rolling(length).mean()
    roll_down = pd.Series(loss, index=close.index).rolling(length).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def atr(df, length=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# =========================
# NGUỒN DỮ LIỆU
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_binance(symbol="XAUUSDT", interval="1h", limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 451:
            raise ValueError("Binance bị chặn hoặc không hỗ trợ XAUUSDT.")
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "OpenTime", "Open", "High", "Low", "Close", "Volume",
            "CloseTime", "QuoteAssetVolume", "NumTrades", "TBB", "TBQ", "Ignore"
        ])
        df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")
        df = df.astype(float)
        df.set_index("OpenTime", inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.warning(f"⚠️ Không lấy được từ Binance: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kraken(pair="XAU/USD", interval=60):  # interval=60 phút
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": pair, "interval": interval}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = list(r.json()["result"].values())[0]
        df = pd.DataFrame(data, columns=[
            "time", "Open", "High", "Low", "Close", "vwap", "volume", "count"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.astype(float)
        df.set_index("time", inplace=True)
        df.rename(columns={"volume": "Volume"}, inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.warning(f"⚠️ Kraken lỗi: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_yahoo():
    import yfinance as yf
    try:
        df = yf.download("XAUUSD=X", period="90d", interval="1h", progress=False)
        if not df.empty:
            df.rename(columns=str.capitalize, inplace=True)
        return df
    except Exception as e:
        st.warning(f"⚠️ Yahoo lỗi: {e}")
        return pd.DataFrame()

# =========================
# PHÂN TÍCH
# =========================
def evaluate(df, frame):
    res = {
        "frame": frame, "rsi": None, "ma20": None, "ma50": None,
        "macd_cross": False, "vol_spike": False, "trend": "-", "suggest": "HOLD"
    }
    if df.empty:
        res["trend"] = "Không có dữ liệu"
        return res

    df["RSI"] = rsi(df["Close"])
    df["MA20"], df["MA50"] = sma(df["Close"], 20), sma(df["Close"], 50)
    m, s, h = macd(df["Close"])
    df["MACD"], df["SIGNAL"] = m, s
    last, prev = df.iloc[-1], df.iloc[-2]
    res["rsi"], res["ma20"], res["ma50"] = last["RSI"], last["MA20"], last["MA50"]

    res["macd_cross"] = prev["MACD"] <= prev["SIGNAL"] and last["MACD"] > last["SIGNAL"]
    vol_avg = df["Volume"].rolling(20).mean()
    res["vol_spike"] = last["Volume"] > 1.5 * vol_avg.iloc[-1]
    res["trend"] = "Tăng" if last["Close"] > last["MA20"] > last["MA50"] else "Giảm"

    if res["macd_cross"] and res["trend"] == "Tăng" and res["vol_spike"]:
        res["suggest"] = "BUY"
    elif res["rsi"] > 70:
        res["suggest"] = "SELL"
    return res

def build_table(res):
    def f(v): return "-" if v is None or pd.isna(v) else f"{v:.2f}"
    data = [
        ["1", "Xu hướng", res["trend"], "Có" if res["trend"] == "Tăng" else "Không", res["suggest"]],
        ["2", "RSI", f(res["rsi"]), "Có" if (res["rsi"] < 30 or res["rsi"] > 50) else "Không", res["suggest"]],
        ["3", "MACD", "Cắt lên" if res["macd_cross"] else "Chưa", "Có" if res["macd_cross"] else "Không", res["suggest"]],
        ["4", "Giá > MA20/50", f"{f(res['ma20'])} / {f(res['ma50'])}",
         "Có" if res["trend"] == "Tăng" else "Không", res["suggest"]],
        ["5", "Volume", "Tăng mạnh" if res["vol_spike"] else "Bình thường", 
         "Có" if res["vol_spike"] else "Không", res["suggest"]],
    ]
    return pd.DataFrame(data, columns=["STT", "Chỉ tiêu", "Giá trị", "Đáp ứng tín hiệu tăng?", "Khuyến nghị"])

# =========================
# GIAO DIỆN APP
# =========================
st.sidebar.header("Cấu hình")
symbol = st.sidebar.text_input("Nhập mã:", "XAUUSDT")

st.subheader("💰 Giá thời gian thực")
df_bin = fetch_binance(symbol, "1m", 20)
if df_bin.empty:
    df_bin = fetch_kraken("XAU/USD", 1)
if df_bin.empty:
    df_bin = fetch_yahoo()

if not df_bin.empty:
    last = df_bin.iloc[-1]
    prev = df_bin.iloc[-2] if len(df_bin) > 1 else last
    price = last["Close"]
    delta = price - prev["Close"]
    st.metric("Giá hiện tại", f"{price:.2f}", f"{delta:+.2f}")
else:
    st.warning("Không lấy được dữ liệu thời gian thực từ bất kỳ nguồn nào.")

if st.button("🔍 Phân tích"):
    with st.spinner("Đang phân tích..."):
        for frame, interval in [("1H", 60), ("4H", 240), ("1D", 1440)]:
            df = fetch_kraken("XAU/USD", interval)
            if df.empty:
                df = fetch_yahoo()
            res = evaluate(df, frame)
            st.markdown(f"### ⏱ Khung {frame}")
            st.dataframe(build_table(res), use_container_width=True)
