import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
import time

# =========================
# CẤU HÌNH ỨNG DỤNG
# =========================
st.set_page_config(page_title="Gold Analyst Pro v7", layout="wide")
st.title("🏆 Gold Analyst Pro v7 – AI phân tích vàng (Realtime Finnhub + History Yahoo)")
st.caption(
    "Giá realtime từ Finnhub.io (Free Tier) + Dữ liệu lịch sử từ Yahoo Finance. "
    "Phân tích RSI, MACD, MA20/50, Volume, và khuyến nghị BUY/SELL tự động."
)

# =========================
# API KEY CỦA BẠN (FINNHUB)
# =========================
FINNHUB_KEY = "d3qnebhr01quv7kbllqgd3qnebhr01quv7kbllr0"

# =========================
# CÁC HÀM CHỈ BÁO
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
def macd(close, f=12, s=26, sig=9):
    macd_line = ema(close, f) - ema(close, s)
    signal_line = ema(macd_line, sig)
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
# LẤY GIÁ REALTIME (FINNHUB)
# =========================
def fetch_realtime():
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol=XAUUSD&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()
        price = d.get("c", 0.0)
        t = d.get("t", 0)
        return {"price": price, "time": datetime.fromtimestamp(t) if t > 0 else datetime.now()}
    except Exception as e:
        st.error(f"Lỗi dữ liệu realtime Finnhub: {e}")
        return None

# =========================
# LẤY DỮ LIỆU LỊCH SỬ (YAHOO FINANCE)
# =========================
def fetch_history(interval="1h", period="90d"):
    try:
        df = yf.download("XAUUSD=X", interval=interval, period=period, progress=False)
        df.rename(columns=str.capitalize, inplace=True)
        return df
    except Exception as e:
        st.error(f"Lỗi dữ liệu Yahoo Finance: {e}")
        return pd.DataFrame()

# =========================
# PHÂN TÍCH
# =========================
def analyze(df):
    res = {"trend":"-", "rsi":None, "ma20":None, "ma50":None,
            "macd_cross":False, "vol_spike":False,
            "suggest":"HOLD", "tp":None, "sl":None}
    if df.empty: return res
    df["RSI"]=rsi(df.Close)
    df["MA20"],df["MA50"]=sma(df.Close,20),sma(df.Close,50)
    m,s,h=macd(df.Close); df["MACD"],df["SIGNAL"],df["HIST"]=m,s,h
    df["ATR"]=atr(df)
    last,prev=df.iloc[-1],df.iloc[-2]
    res["rsi"],res["ma20"],res["ma50"]=last.RSI,last.MA20,last.MA50
    res["macd_cross"]=prev.MACD<=prev.SIGNAL and last.MACD>last.SIGNAL
    res["trend"]="Tăng" if last.Close>last.MA20>last.MA50 else "Giảm"
    if res["macd_cross"] and res["trend"]=="Tăng": res["suggest"]="BUY"
    elif res["rsi"]>70: res["suggest"]="SELL"
    atr_val=last.ATR if not pd.isna(last.ATR) else 0
    if res["suggest"]=="BUY":
        res["tp"]=round(last.Close+1.5*atr_val,2)
        res["sl"]=round(last.Close-1.0*atr_val,2)
    elif res["suggest"]=="SELL":
        res["tp"]=round(last.Close-1.5*atr_val,2)
        res["sl"]=round(last.Close+1.0*atr_val,2)
    return res, df

# =========================
# VẼ BIỂU ĐỒ
# =========================
def plot_charts(df):
    candle=go.Figure()
    candle.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,
                                    low=df.Low,close=df.Close,name="Giá",
                                    increasing_line_color="green",
                                    decreasing_line_color="red"))
    candle.add_trace(go.Scatter(x=df.index,y=df.MA20,line=dict(color="orange"),name="MA20"))
    candle.add_trace(go.Scatter(x=df.index,y=df.MA50,line=dict(color="blue"),name="MA50"))
    candle.update_layout(title="Biểu đồ giá XAU/USD",xaxis_rangeslider_visible=False,height=400)

    rsi_fig=go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index,y=df.RSI,line=dict(color="purple"),name="RSI"))
    rsi_fig.add_hrect(y0=30,y1=70,fillcolor="gray",opacity=0.2,line_width=0)
    rsi_fig.update_layout(title="Chỉ báo RSI(14)",height=200)

    macd_fig=go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index,y=df.MACD,line=dict(color="orange"),name="MACD"))
    macd_fig.add_trace(go.Scatter(x=df.index,y=df.SIGNAL,line=dict(color="blue"),name="Signal"))
    macd_fig.add_trace(go.Bar(x=df.index,y=df.HIST,name="Histogram",marker_color="gray"))
    macd_fig.update_layout(title="Chỉ báo MACD",height=200)
    return candle, rsi_fig, macd_fig

# =========================
# GIAO DIỆN STREAMLIT
# =========================
st.subheader("💰 Giá vàng thời gian thực")
realtime = fetch_realtime()
if realtime:
    st.metric("Giá hiện tại (XAU/USD)", f"{realtime['price']:.2f}")
    st.write(f"🕒 Cập nhật lúc: {realtime['time']}")
else:
    st.warning("Không thể lấy dữ liệu realtime từ Finnhub.io.")

if st.button("🔍 Phân tích chuyên sâu"):
    with st.spinner("Đang tải dữ liệu lịch sử & phân tích..."):
        df = fetch_history()
        if not df.empty:
            res, df = analyze(df)
            st.markdown("### 📊 Kết quả phân tích (1H)")
            st.dataframe(pd.DataFrame([
                ["Xu hướng", res["trend"]],
                ["RSI(14)", f"{res['rsi']:.2f}" if res["rsi"] else "-"],
                ["Giá > MA20/50", "Có" if res["trend"]=="Tăng" else "Không"],
                ["MACD", "Cắt lên" if res["macd_cross"] else "Chưa"],
                ["Khuyến nghị", res["suggest"]],
                ["Take Profit", res["tp"] if res["tp"] else "-"],
                ["Cut Loss", res["sl"] if res["sl"] else "-"]
            ], columns=["Chỉ tiêu","Giá trị"]), use_container_width=True)

            candle, rsi_fig, macd_fig = plot_charts(df)
            st.plotly_chart(candle, use_container_width=True)
            st.plotly_chart(rsi_fig, use_container_width=True)
            st.plotly_chart(macd_fig, use_container_width=True)
        else:
            st.warning("Không có dữ liệu lịch sử để phân tích.")

st.caption("⚠️ Dữ liệu realtime từ Finnhub.io; lịch sử từ Yahoo Finance. Không phải lời khuyên đầu tư.")
