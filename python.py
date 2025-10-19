import streamlit as st
import pandas as pd
import numpy as np
import requests
import pytz
from datetime import datetime

# =========================
# CẤU HÌNH GIAO DIỆN
# =========================
st.set_page_config(page_title="XAUUSD AI Analyst PRO", layout="wide")
st.title("🤖 AI chuyên gia phân tích Vàng (XAUUSD - Realtime Binance)")
st.caption(
    "AI đóng vai chuyên gia đầu tư vàng với hơn 30 năm kinh nghiệm. "
    "Phân tích realtime giá XAUUSDT từ Binance, chỉ báo RSI, MACD, MA20/50, Volume, "
    "và khuyến nghị Buy/Sell + TP/SL theo thời gian thực."
)

# =========================
# HÀM TIỆN ÍCH
# =========================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def sma(series, length):
    return series.rolling(window=length, min_periods=length).mean()

def rsi(close, length=14):
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).rolling(length).mean()
    roll_down = pd.Series(loss, index=close.index).rolling(length).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df, length=14):
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# =========================
# BINANCE API - GET DATA
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_binance(symbol="XAUUSDT", interval="1h", limit=500):
    """
    Lấy dữ liệu giá vàng XAUUSDT từ Binance Futures (realtime)
    interval: 1m, 5m, 1h, 4h, 1d
    """
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "OpenTime", "Open", "High", "Low", "Close", "Volume",
            "CloseTime", "QuoteAssetVolume", "NumTrades", "TBBaseVol", "TBQuoteVol", "Ignore"
        ])
        df = df.astype(float)
        df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")
        df.set_index("OpenTime", inplace=True)
        df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"}, inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        st.error(f"Lỗi Binance API: {e}")
        return pd.DataFrame()

# =========================
# PHÂN TÍCH KHUNG
# =========================
def evaluate_frame(df, name):
    res = {
        "frame": name, "close": None, "rsi": None, "ma20": None, "ma50": None,
        "macd_cross_up": False, "bullish_div_rsi": False,
        "price_above_ma2050": False, "volume_spike": False,
        "trend_note": "-", "suggest": "HOLD", "tp": None, "sl": None
    }

    if df.empty:
        res["trend_note"] = "Không có dữ liệu"
        return res

    df['RSI14'] = rsi(df['Close'])
    macd_line, signal_line, hist = macd(df['Close'])
    df['MACD'], df['SIGNAL'], df['HIST'] = macd_line, signal_line, hist
    df['MA20'], df['MA50'] = sma(df['Close'], 20), sma(df['Close'], 50)
    df['ATR14'] = atr(df, 14)

    last = df.iloc[-1]
    res["close"] = float(last['Close'])
    res["rsi"] = float(last['RSI14']) if not pd.isna(last['RSI14']) else None
    res["ma20"] = float(last['MA20']) if not pd.isna(last['MA20']) else None
    res["ma50"] = float(last['MA50']) if not pd.isna(last['MA50']) else None
    res["price_above_ma2050"] = (res["ma20"] and res["ma50"]) and last['Close'] > last['MA20'] and last['Close'] > last['MA50']

    # MACD cắt lên
    if len(df) >= 2:
        prev = df.iloc[-2]
        if prev['MACD'] <= prev['SIGNAL'] and last['MACD'] > last['SIGNAL']:
            res["macd_cross_up"] = True

    # RSI phân kỳ tăng
    if len(df) > 40:
        w = df.iloc[-40:]
        p1, p2 = w['Close'].iloc[-20], w['Close'].iloc[-1]
        r1, r2 = w['RSI14'].iloc[-20], w['RSI14'].iloc[-1]
        if (p2 < p1) and (r2 > r1):
            res["bullish_div_rsi"] = True

    # Volume spike
    vol_avg = df['Volume'].rolling(20).mean()
    if df['Volume'].iloc[-1] > 1.5 * vol_avg.iloc[-1]:
        res["volume_spike"] = True

    # Xu hướng
    if res["macd_cross_up"] and res["price_above_ma2050"]:
        res["trend_note"] = "Bứt phá tăng"
    elif res["bullish_div_rsi"]:
        res["trend_note"] = "Đảo chiều tăng"
    else:
        res["trend_note"] = "Trung tính"

    # Kết luận
    if res["bullish_div_rsi"] and res["macd_cross_up"] and res["price_above_ma2050"] and res["volume_spike"]:
        res["suggest"] = "BUY"
    elif res["rsi"] and res["rsi"] > 70 and not res["price_above_ma2050"]:
        res["suggest"] = "SELL"

    atr_val = last['ATR14'] if not pd.isna(last['ATR14']) else 0
    if res["suggest"] == "BUY":
        res["tp"] = round(last['Close'] + 1.5 * atr_val, 2)
        res["sl"] = round(last['Close'] - 1.0 * atr_val, 2)
    elif res["suggest"] == "SELL":
        res["tp"] = round(last['Close'] - 1.5 * atr_val, 2)
        res["sl"] = round(last['Close'] + 1.0 * atr_val, 2)

    return res

def build_table(res):
    def fmt(val, digits=2):
        if val is None or pd.isna(val):
            return "-"
        try:
            return f"{val:.{digits}f}"
        except Exception:
            return str(val)

    data = [
        ["1", "Xu hướng (Trend)", res.get("trend_note", "-"),
         "Có" if "tăng" in str(res.get("trend_note", "")).lower() else "Không",
         res.get("suggest", "HOLD")],
        ["2", "RSI(14)", fmt(res.get("rsi")),
         "Có" if ((res.get("rsi") and (res['rsi'] < 30 or res['rsi'] > 50 or res.get("bullish_div_rsi")))
                  or res.get("bullish_div_rsi")) else "Không",
         res.get("suggest", "HOLD")],
        ["3", "MACD(12,26,9)",
         "MACD cắt lên" if res.get("macd_cross_up") else "Chưa cắt lên",
         "Có" if res.get("macd_cross_up") else "Không",
         res.get("suggest", "HOLD")],
        ["4", "Giá vượt MA20/50",
         f"MA20={fmt(res.get('ma20'))} / MA50={fmt(res.get('ma50'))}",
         "Có" if res.get("price_above_ma2050") else "Không",
         res.get("suggest", "HOLD")],
        ["5", "Volume xác nhận",
         "Tăng mạnh" if res.get("volume_spike") else "Bình thường",
         "Có" if res.get("volume_spike") else "Không",
         res.get("suggest", "HOLD")]
    ]
    return pd.DataFrame(data, columns=["STT", "Chỉ tiêu", "Giá trị", "Đáp ứng tín hiệu tăng?", "Khuyến nghị"])

# =========================
# GIAO DIỆN APP
# =========================
with st.sidebar:
    st.markdown("### Cấu hình")
    symbol = st.text_input("Nhập mã Binance (VD: XAUUSDT):", "XAUUSDT")
    st.caption("Hỗ trợ: XAUUSDT, BTCUSDT, ETHUSDT ...")

# Giá realtime
st.subheader("💰 Giá thời gian thực")
df_now = fetch_binance(symbol, interval="1m", limit=20)
if not df_now.empty:
    last = df_now.iloc[-1]
    prev = df_now.iloc[-2]
    price = last['Close']
    delta = price - prev['Close']
    st.metric(f"{symbol}", f"{price:.2f}", f"{delta:+.2f}")
else:
    st.warning("Không thể tải dữ liệu realtime từ Binance.")

# Phân tích
if st.button("🔍 Phân tích"):
    with st.spinner("Đang phân tích khung 1h / 4h / 1D..."):
        df_1h = fetch_binance(symbol, interval="1h", limit=500)
        df_4h = fetch_binance(symbol, interval="4h", limit=500)
        df_1d = fetch_binance(symbol, interval="1d", limit=500)

        results = [evaluate_frame(df_1h, "1H"),
                   evaluate_frame(df_4h, "4H"),
                   evaluate_frame(df_1d, "1D")]

        for r in results:
            st.markdown(f"### ⏱ Khung {r['frame']}")
            st.dataframe(build_table(r), use_container_width=True)

        st.divider()
        st.subheader("📈 Khuyến nghị tổng hợp")
        best = next((r for r in results if r["suggest"] in ("BUY", "SELL")), results[-1])
        st.metric("Tín hiệu", best["suggest"])
        st.metric("Khung thời gian", best["frame"])
        st.metric("Take Profit", f"{best['tp']}" if best["tp"] else "-")
        st.metric("Cut Loss", f"{best['sl']}" if best["sl"] else "-")
        st.info("RSI phân kỳ tăng + MACD cắt lên + Giá > MA20/50 + Volume tăng ⇒ BUY.")

st.caption("⚠️ Thông tin chỉ tham khảo, không phải khuyến nghị đầu tư.")
