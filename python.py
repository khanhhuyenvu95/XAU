import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pytz
from datetime import datetime

# =========================
# CẤU HÌNH GIAO DIỆN
# =========================
st.set_page_config(page_title="AI Analyst Pro - XAUUSD", layout="wide")
st.title("🤖 AI chuyên gia phân tích vàng (XAUUSD - Pro v2)")
st.caption(
    "Phân tích kỹ thuật XAUUSD theo dữ liệu thực từ Yahoo Finance. "
    "Hiển thị biểu đồ nến, RSI, MACD, Volume và khuyến nghị đầu tư thông minh (Buy/Sell + TP/SL)."
)

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
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df, length=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# =========================
# LẤY DỮ LIỆU YFINANCE
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_yahoo(symbol="XAUUSD=X", interval="1h", period="90d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            st.warning("⚠️ Không thể lấy dữ liệu XAUUSD từ Yahoo Finance.")
        df.rename(columns=str.capitalize, inplace=True)
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu từ Yahoo Finance: {e}")
        return pd.DataFrame()

# =========================
# PHÂN TÍCH KỸ THUẬT
# =========================
def evaluate(df, frame):
    res = {"frame": frame, "trend": "-", "rsi": None, "ma20": None, "ma50": None,
            "macd_cross": False, "vol_spike": False, "suggest": "HOLD", "tp": None, "sl": None}
    if df.empty:
        res["trend"] = "Không có dữ liệu"
        return res

    df["RSI"] = rsi(df["Close"])
    df["MA20"], df["MA50"] = sma(df["Close"], 20), sma(df["Close"], 50)
    m, s, h = macd(df["Close"])
    df["MACD"], df["SIGNAL"], df["HIST"] = m, s, h
    df["ATR"] = atr(df)

    last, prev = df.iloc[-1], df.iloc[-2]
    res["rsi"], res["ma20"], res["ma50"] = last["RSI"], last["MA20"], last["MA50"]

    res["macd_cross"] = prev["MACD"] <= prev["SIGNAL"] and last["MACD"] > last["SIGNAL"]
    res["vol_spike"] = last["Volume"] > 1.5 * df["Volume"].rolling(20).mean().iloc[-1]
    res["trend"] = "Tăng" if last["Close"] > last["MA20"] > last["MA50"] else "Giảm"

    if res["macd_cross"] and res["trend"] == "Tăng" and res["vol_spike"]:
        res["suggest"] = "BUY"
    elif res["rsi"] > 70:
        res["suggest"] = "SELL"

    atr_val = last["ATR"] if not pd.isna(last["ATR"]) else 0
    if res["suggest"] == "BUY":
        res["tp"] = round(last["Close"] + 1.5 * atr_val, 2)
        res["sl"] = round(last["Close"] - 1.0 * atr_val, 2)
    elif res["suggest"] == "SELL":
        res["tp"] = round(last["Close"] - 1.5 * atr_val, 2)
        res["sl"] = round(last["Close"] + 1.0 * atr_val, 2)
    return res

# =========================
# VẼ BIỂU ĐỒ PLOTLY
# =========================
def plot_chart(df, frame):
    fig = go.Figure()

    # Nến
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Giá", increasing_line_color="green", decreasing_line_color="red"
    ))

    # MA20 & MA50
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", line=dict(color="orange", width=1.5), name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", line=dict(color="blue", width=1.5), name="MA50"))

    fig.update_layout(
        title=f"Biểu đồ XAUUSD ({frame})",
        xaxis_title="Thời gian",
        yaxis_title="Giá (USD)",
        template="plotly_white",
        height=600,
        showlegend=True
    )
    return fig

# =========================
# GIAO DIỆN APP
# =========================
st.sidebar.header("Cấu hình")
frame = st.sidebar.selectbox("Khung thời gian:", ["1h", "4h", "1d"])
symbol = "XAUUSD=X"

st.subheader("💰 Giá thời gian thực")
df = fetch_yahoo(symbol, frame, "90d")

if not df.empty:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    price = last["Close"]
    delta = price - prev["Close"]
    st.metric("Giá hiện tại", f"{price:.2f}", f"{delta:+.2f}")
else:
    st.warning("Không thể tải giá từ Yahoo Finance.")

if st.button("🔍 Phân tích"):
    with st.spinner("Đang tính toán..."):
        result = evaluate(df, frame.upper())
        st.markdown(f"### 🧩 Kết quả phân tích ({frame.upper()})")
        st.dataframe(pd.DataFrame([
            ["Xu hướng", result["trend"]],
            ["RSI(14)", f"{result['rsi']:.2f}" if result['rsi'] else "-"],
            ["Giá > MA20/50", "Có" if result["trend"] == "Tăng" else "Không"],
            ["MACD", "Cắt lên" if result["macd_cross"] else "Chưa cắt"],
            ["Volume", "Tăng mạnh" if result["vol_spike"] else "Bình thường"],
            ["Khuyến nghị", result["suggest"]],
            ["TP", result["tp"] if result["tp"] else "-"],
            ["SL", result["sl"] if result["sl"] else "-"]
        ], columns=["Chỉ tiêu", "Giá trị"]), use_container_width=True)

        st.plotly_chart(plot_chart(df, frame.upper()), use_container_width=True)

st.caption("⚠️ Dữ liệu cập nhật từ Yahoo Finance (5–10 phút trễ). Không phải lời khuyên đầu tư.")
