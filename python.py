import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import time

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Gold Analyst Pro v6", layout="wide")
st.title("🏆 Gold Analyst Pro v6 – AI chuyên gia phân tích vàng (Finnhub.io)")
st.caption(
    "Realtime & dữ liệu lịch sử từ Finnhub.io. Phân tích RSI, MACD, MA20/50, Volume, "
    "và khuyến nghị đầu tư thông minh. Tự động cập nhật mỗi 30 giây."
)

# =========================
# API KEY CỦA BẠN
# =========================
FINNHUB_KEY = "d3qnebhr01quv7kbllqgd3qnebhr01quv7kbllr0"

# =========================
# CHỈ BÁO KỸ THUẬT
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
# LẤY GIÁ REALTIME
# =========================
def fetch_realtime():
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol=XAUUSD&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        d = r.json()
        return {"price": d["c"], "time": datetime.fromtimestamp(d["t"])}
    except Exception as e:
        st.error(f"Lỗi dữ liệu realtime Finnhub: {e}")
        return None

# =========================
# LẤY DỮ LIỆU LỊCH SỬ
# =========================
def fetch_history(resolution="60"):
    try:
        # resolution: 1, 5, 15, 30, 60, D, W, M
        now = int(time.time())
        frm = now - 90 * 24 * 3600
        url = f"https://finnhub.io/api/v1/forex/candle?symbol=OANDA:XAU_USD&resolution={resolution}&from={frm}&to={now}&token={FINNHUB_KEY}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("s") != "ok":
            st.warning("⚠️ Không lấy được dữ liệu lịch sử từ Finnhub.")
            return pd.DataFrame()
        df = pd.DataFrame({
            "Time": pd.to_datetime(data["t"], unit="s"),
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data["v"]
        })
        df.set_index("Time", inplace=True)
        return df
    except Exception as e:
        st.error(f"Lỗi dữ liệu lịch sử Finnhub: {e}")
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
    candle.update_layout(title="Biểu đồ giá XAU/USD (1H)",xaxis_rangeslider_visible=False,height=400)

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
# AUTO REFRESH
# =========================
placeholder = st.empty()
interval = 30  # 30s refresh

while True:
    with placeholder.container():
        st.subheader("💰 Giá vàng thời gian thực")
        rt = fetch_realtime()
        if rt:
            st.metric("Giá hiện tại (XAU/USD)", f"{rt['price']:.2f}")
            st.write(f"🕒 Cập nhật lúc: {rt['time']}")
        else:
            st.warning("Không thể lấy dữ liệu realtime.")

        df = fetch_history()
        if not df.empty:
            res, df = analyze(df)
            st.markdown("### 📊 Phân tích kỹ thuật (1H)")
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

        st.caption("⚠️ Dữ liệu realtime & lịch sử từ Finnhub.io. Không phải lời khuyên đầu tư.")
        st.info(f"⏳ Trang sẽ tự động cập nhật sau {interval} giây.")
    time.sleep(interval)
