import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# =========================
# CẤU HÌNH ỨNG DỤNG
# =========================
st.set_page_config(page_title="Gold Analyst Pro v4", layout="wide")
st.title("🏆 Gold Analyst Pro v4 – AI chuyên gia phân tích vàng (XAU/USD)")
st.caption(
    "Realtime từ GoldAPI.io + Dữ liệu lịch sử từ Yahoo Finance. "
    "Hiển thị 3 biểu đồ chuyên sâu (Nến, RSI, MACD) và khuyến nghị đầu tư thông minh."
)

# =========================
# API KEY (GoldAPI.io)
# =========================
GOLD_API_KEY = "goldapi-hoaacsmgyc540m-io"

# =========================
# HÀM CHỈ BÁO KỸ THUẬT
# =========================
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def sma(series, n): return series.rolling(n).mean()

def rsi(close, n=14):
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain, index=close.index).rolling(n).mean()
    roll_down = pd.Series(loss, index=close.index).rolling(n).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df, n=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# =========================
# LẤY DỮ LIỆU GOLDAPI
# =========================
@st.cache_data(ttl=60)
def fetch_goldapi():
    url = "https://www.goldapi.io/api/XAU/USD"
    headers = {"x-access-token": GOLD_API_KEY, "Content-Type": "application/json"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            st.warning(f"⚠️ GoldAPI lỗi: {r.status_code} - {r.text}")
            return None
        d = r.json()
        return {
            "price": d.get("price"),
            "ask": d.get("ask"),
            "bid": d.get("bid"),
            "timestamp": datetime.fromtimestamp(d.get("timestamp"))
        }
    except Exception as e:
        st.error(f"Lỗi kết nối GoldAPI: {e}")
        return None

# =========================
# DỮ LIỆU LỊCH SỬ (Yahoo)
# =========================
@st.cache_data(ttl=600)
def fetch_history():
    df = yf.download("XAUUSD=X", period="90d", interval="1h", progress=False)
    df.rename(columns=str.capitalize, inplace=True)
    return df

# =========================
# PHÂN TÍCH
# =========================
def evaluate(df):
    res = {"trend": "-", "rsi": None, "ma20": None, "ma50": None,
            "macd_cross": False, "vol_spike": False,
            "suggest": "HOLD", "tp": None, "sl": None}
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
# BIỂU ĐỒ PLOTLY (3 vùng)
# =========================
def plot_full_chart(df):
    # Vùng 1: Nến + MA
    candle = go.Figure()
    candle.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Giá", increasing_line_color="green",
        decreasing_line_color="red"
    ))
    candle.add_trace(go.Scatter(x=df.index, y=df["MA20"],
                                line=dict(color="orange", width=1.5), name="MA20"))
    candle.add_trace(go.Scatter(x=df.index, y=df["MA50"],
                                line=dict(color="blue", width=1.5), name="MA50"))
    candle.update_layout(title="Biểu đồ giá XAU/USD", xaxis_rangeslider_visible=False, height=400)

    # Vùng 2: RSI
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"],
                                 line=dict(color="purple", width=1.5), name="RSI(14)"))
    rsi_fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.2, line_width=0)
    rsi_fig.update_layout(title="Chỉ báo RSI", height=200)

    # Vùng 3: MACD
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="orange"), name="MACD"))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["SIGNAL"], line=dict(color="blue"), name="Signal"))
    macd_fig.add_trace(go.Bar(x=df.index, y=df["HIST"], name="Histogram", marker_color="gray"))
    macd_fig.update_layout(title="Chỉ báo MACD", height=200)

    return candle, rsi_fig, macd_fig

# =========================
# GIAO DIỆN STREAMLIT
# =========================
st.subheader("💰 Giá vàng thời gian thực")
gold_data = fetch_goldapi()

if gold_data:
    st.metric("Giá hiện tại (XAU/USD)", f"{gold_data['price']:.2f}")
    st.write(f"🕒 Cập nhật lúc: {gold_data['timestamp']}")
else:
    st.warning("Không thể tải dữ liệu realtime từ GoldAPI.io")

if st.button("🔍 Phân tích chuyên sâu"):
    with st.spinner("Đang phân tích dữ liệu..."):
        df = fetch_history()
        if not df.empty:
            result = evaluate(df)
            st.markdown("### 📈 Kết quả phân tích (1H)")
            st.dataframe(pd.DataFrame([
                ["Xu hướng", result["trend"]],
                ["RSI(14)", f"{result['rsi']:.2f}" if result["rsi"] else "-"],
                ["Giá > MA20/50", "Có" if result["trend"] == "Tăng" else "Không"],
                ["MACD", "Cắt lên" if result["macd_cross"] else "Chưa"],
                ["Volume", "Tăng mạnh" if result["vol_spike"] else "Bình thường"],
                ["Khuyến nghị", result["suggest"]],
                ["Take Profit", result["tp"] if result["tp"] else "-"],
                ["Cut Loss", result["sl"] if result["sl"] else "-"]
            ], columns=["Chỉ tiêu", "Giá trị"]), use_container_width=True)

            candle, rsi_fig, macd_fig = plot_full_chart(df)
            st.plotly_chart(candle, use_container_width=True)
            st.plotly_chart(rsi_fig, use_container_width=True)
            st.plotly_chart(macd_fig, use_container_width=True)
        else:
            st.warning("Không thể tải dữ liệu lịch sử từ Yahoo Finance.")

st.caption("⚠️ Dữ liệu realtime từ GoldAPI.io; lịch sử từ Yahoo Finance. Không phải lời khuyên đầu tư.")
