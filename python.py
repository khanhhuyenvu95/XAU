import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# =========================
# CẤU HÌNH
# =========================
st.set_page_config(page_title="Gold Analyst Pro v8.2", layout="wide")
st.title("🏆 Gold Analyst Pro v8.2 – AI phân tích vàng (Yahoo Realtime + History Fallback)")
st.caption("Lấy dữ liệu realtime & lịch sử từ Yahoo Finance. Tự động chuyển nguồn nếu lỗi. Không cần API key.")

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
# HÀM LẤY DỮ LIỆU (CÓ FALLBACK)
# =========================
def fetch_data(symbol="XAUUSD=X", interval="1h", period="90d"):
    try:
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        if df.empty:
            st.warning(f"⚠️ Không có dữ liệu cho {symbol}. Thử fallback sang GC=F (Gold Futures).")
            df = yf.download("GC=F", interval=interval, period=period, progress=False)
        df.rename(columns=str.capitalize, inplace=True)
        return df
    except Exception as e:
        st.error(f"Lỗi Yahoo Finance: {e}")
        return pd.DataFrame()

# =========================
# PHÂN TÍCH
# =========================
def analyze(df):
    res = {"trend":"-", "rsi":None, "ma20":None, "ma50":None,
            "macd_cross":False, "suggest":"HOLD", "tp":None, "sl":None}
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
# BIỂU ĐỒ
# =========================
def plot_charts(df):
    candle=go.Figure()
    candle.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,
                                    low=df.Low,close=df.Close,name="Giá",
                                    increasing_line_color="green",
                                    decreasing_line_color="red"))
    candle.add_trace(go.Scatter(x=df.index,y=df.MA20,line=dict(color="orange"),name="MA20"))
    candle.add_trace(go.Scatter(x=df.index,y=df.MA50,line=dict(color="blue"),name="MA50"))
    candle.update_layout(title="Biểu đồ giá vàng",xaxis_rangeslider_visible=False,height=400)

    rsi_fig=go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index,y=df.RSI,line=dict(color="purple"),name="RSI"))
    rsi_fig.add_hrect(y0=30,y1=70,fillcolor="gray",opacity=0.2,line_width=0)
    rsi_fig.update_layout(title="RSI(14)",height=200)

    macd_fig=go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index,y=df.MACD,line=dict(color="orange"),name="MACD"))
    macd_fig.add_trace(go.Scatter(x=df.index,y=df.SIGNAL,line=dict(color="blue"),name="Signal"))
    macd_fig.add_trace(go.Bar(x=df.index,y=df.HIST,name="Histogram",marker_color="gray"))
    macd_fig.update_layout(title="MACD",height=200)
    return candle, rsi_fig, macd_fig

# =========================
# GIAO DIỆN APP
# =========================
st_autorefresh(interval=30 * 1000, key="refresh_data")
interval_map = {"1 Giờ": "1h", "4 Giờ": "4h", "1 Ngày": "1d"}
selected = st.selectbox("⏱️ Chọn khung thời gian:", list(interval_map.keys()))

# Lấy realtime
st.subheader("💰 Giá vàng thời gian thực (Yahoo Finance)")
try:
    df_live = yf.download("XAUUSD=X", period="1d", interval="1m", progress=False)
    if df_live.empty:
        df_live = yf.download("GC=F", period="1d", interval="1m", progress=False)
    if not df_live.empty:
        last_price = df_live["Close"].iloc[-1]
        st.metric("Giá hiện tại (XAU/USD)", f"{last_price:.2f}")
        st.write(f"🕒 Cập nhật lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("Không thể tải dữ liệu realtime từ Yahoo hoặc GC=F.")
except Exception as e:
    st.error(f"Lỗi realtime Yahoo: {e}")

# Phân tích
if st.button("🔍 Phân tích chuyên sâu"):
    with st.spinner("Đang phân tích..."):
        df = fetch_data(interval=interval_map[selected])
        if not df.empty:
            res, df = analyze(df)
            st.markdown(f"### 📊 Phân tích kỹ thuật ({selected})")
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

st.caption("⚠️ Dữ liệu realtime & lịch sử từ Yahoo Finance. Tự động fallback sang GC=F nếu cần.")
