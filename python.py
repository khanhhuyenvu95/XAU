import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz

# =========================
# CẤU HÌNH GIAO DIỆN
# =========================
st.set_page_config(page_title="XAUUSD AI Analyst", layout="wide")
st.title("🤖 AI chuyên gia phân tích Vàng (XAUUSD)")
st.caption(
    "AI đóng vai chuyên gia đầu tư vàng với 30 năm kinh nghiệm. "
    "Phân tích RSI, MACD, MA20/50, Volume, xu hướng đảo chiều hoặc bứt phá, "
    "và gợi ý Buy/Sell + TP/SL theo thời gian thực. "
)

# =========================
# HÀM TÍNH CHỈ BÁO
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length, min_periods=length).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).rolling(length).mean()
    roll_down = pd.Series(loss, index=close.index).rolling(length).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(length).mean()

# =========================
# TẢI DỮ LIỆU
# =========================
@st.cache_data(show_spinner=False)
def fetch_yf(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            st.warning(f"⚠️ Không có dữ liệu cho {symbol} ({interval})")
        return df
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu {symbol}: {e}")
        return pd.DataFrame()

def resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample từ 1H sang 4H, kiểm tra đủ cột OHLCV"""
    if df_1h is None or df_1h.empty:
        return pd.DataFrame()
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df_1h.columns)
    if missing:
        print(f"[CẢNH BÁO] Thiếu cột {missing} trong dữ liệu => bỏ qua resample.")
        return pd.DataFrame()
    try:
        rule = "4H"
        agg = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }
        return df_1h.resample(rule).apply(agg).dropna()
    except Exception as e:
        print(f"[LỖI RESAMPLE] {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_price_and_frames(symbol_spot: str = "XAUUSD=X"):
    """Tải dữ liệu spot và futures"""
    df_1h = fetch_yf(symbol_spot, period="90d", interval="1h")
    df_1d = fetch_yf(symbol_spot, period="2y", interval="1d")

    # Thêm cột Volume nếu thiếu
    for df in [df_1h, df_1d]:
        if not df.empty and "Volume" not in df.columns:
            df["Volume"] = 0

    df_4h = resample_4h(df_1h)

    # Lấy volume proxy từ Gold Futures
    df_fut_1h = fetch_yf("GC=F", period="90d", interval="1h")
    df_fut_4h = resample_4h(df_fut_1h)
    df_fut_1d = fetch_yf("GC=F", period="2y", interval="1d")

    for df in [df_fut_1h, df_fut_1d, df_fut_4h]:
        if not df.empty and "Volume" not in df.columns:
            df["Volume"] = 0

    return df_1h, df_4h, df_1d, df_fut_1h, df_fut_4h, df_fut_1d

# =========================
# PHÂN TÍCH KHUNG THỜI GIAN
# =========================
def evaluate_frame(df: pd.DataFrame, df_vol_proxy: pd.DataFrame, name: str):
    """Tính toán chỉ báo & gợi ý khung"""
    res = {"frame": name, "close": None, "rsi": None, "ma20": None, "ma50": None,
            "macd_cross_up": False, "bullish_div_rsi": False,
            "price_above_ma2050": False, "volume_spike": False,
            "trend_note": "", "suggest": "HOLD", "tp": None, "sl": None}

    if df.empty:
        res["trend_note"] = "Không có dữ liệu"
        return res

    if 'Volume' not in df.columns or df['Volume'].sum() == 0:
        if df_vol_proxy is not None and not df_vol_proxy.empty:
            df['Volume'] = df_vol_proxy['Volume'].reindex(df.index, method='nearest').fillna(0)
        else:
            df['Volume'] = 0

    df['RSI14'] = rsi(df['Close'])
    macd_line, signal_line, hist = macd(df['Close'])
    df['MACD'], df['SIGNAL'], df['HIST'] = macd_line, signal_line, hist
    df['MA20'], df['MA50'] = sma(df['Close'], 20), sma(df['Close'], 50)
    df['ATR14'] = atr(df, 14)

    last = df.iloc[-1]
    res["close"] = float(last['Close'])
    res["rsi"] = float(last['RSI14'])
    res["ma20"] = float(last['MA20'])
    res["ma50"] = float(last['MA50'])
    res["price_above_ma2050"] = last['Close'] > last['MA20'] and last['Close'] > last['MA50']

    # MACD cắt lên
    if len(df) >= 2:
        prev = df.iloc[-2]
        if prev['MACD'] <= prev['SIGNAL'] and last['MACD'] > last['SIGNAL']:
            res["macd_cross_up"] = True

    # RSI phân kỳ tăng (đơn giản)
    if len(df) > 30:
        w = df.iloc[-40:]
        p1, p2 = w['Close'].iloc[-40//2], w['Close'].iloc[-1]
        r1, r2 = w['RSI14'].iloc[-40//2], w['RSI14'].iloc[-1]
        if (p2 < p1) and (r2 > r1):
            res["bullish_div_rsi"] = True

    # Volume spike
    if 'Volume' in df.columns:
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

    # Khuyến nghị
    if res["bullish_div_rsi"] and res["macd_cross_up"] and res["price_above_ma2050"] and res["volume_spike"]:
        res["suggest"] = "BUY"
    elif res["rsi"] > 70 and not res["price_above_ma2050"]:
        res["suggest"] = "SELL"
    else:
        res["suggest"] = "HOLD"

    atr_val = last['ATR14']
    if res["suggest"] == "BUY":
        res["tp"] = round(last['Close'] + 1.5 * atr_val, 2)
        res["sl"] = round(last['Close'] - 1.0 * atr_val, 2)
    elif res["suggest"] == "SELL":
        res["tp"] = round(last['Close'] - 1.5 * atr_val, 2)
        res["sl"] = round(last['Close'] + 1.0 * atr_val, 2)

    return res

def build_table(res):
    data = [
        ["1", "Xu hướng (Trend)", res["trend_note"], 
         "Có" if "tăng" in res["trend_note"].lower() else "Không", res["suggest"]],
        ["2", "RSI(14)", f"{res['rsi']:.2f}", 
         "Có" if (res["rsi"] < 30 or res["rsi"] > 50 or res["bullish_div_rsi"]) else "Không", res["suggest"]],
        ["3", "MACD(12,26,9)", 
         "MACD cắt lên" if res["macd_cross_up"] else "Chưa cắt lên",
         "Có" if res["macd_cross_up"] else "Không", res["suggest"]],
        ["4", "Giá vượt MA20/50", 
         f"MA20={res['ma20']:.2f} / MA50={res['ma50']:.2f}",
         "Có" if res["price_above_ma2050"] else "Không", res["suggest"]],
        ["5", "Volume xác nhận", 
         "Tăng mạnh" if res["volume_spike"] else "Bình thường", 
         "Có" if res["volume_spike"] else "Không", res["suggest"]]
    ]
    return pd.DataFrame(data, columns=["STT", "Chỉ tiêu", "Giá trị", "Đáp ứng tín hiệu tăng?", "Khuyến nghị"])

# =========================
# GIAO DIỆN APP
# =========================
with st.sidebar:
    symbol = st.text_input("Nhập mã (VD: XAUUSD=X hoặc GC=F):", "XAUUSD=X")
    st.caption("Mặc định XAUUSD=X (Spot Gold).")

st.subheader("🎯 Giá thời gian thực")
df_price = fetch_yf(symbol, period="5d", interval="1h")
if not df_price.empty:
    last = df_price.iloc[-1]
    prev = df_price.iloc[-2]
    price = last['Close']
    delta = price - prev['Close']
    tz = pytz.timezone("Asia/Bangkok")
    st.metric(f"{symbol}", f"{price:.2f}", f"{delta:+.2f}")
    st.write("Cập nhật:", last.name.tz_localize("UTC").tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S %Z"))
else:
    st.warning("Không thể tải giá gần nhất.")

if st.button("🔍 Tìm kiếm & Phân tích"):
    with st.spinner("Đang phân tích..."):
        df_1h, df_4h, df_1d, fut_1h, fut_4h, fut_1d = fetch_price_and_frames(symbol)
        r1h = evaluate_frame(df_1h, fut_1h, "1H")
        r4h = evaluate_frame(df_4h, fut_4h, "4H")
        r1d = evaluate_frame(df_1d, fut_1d, "1D")

        for r in [r1h, r4h, r1d]:
            st.markdown(f"### ⏱ Khung {r['frame']}")
            st.dataframe(build_table(r), use_container_width=True)

        st.divider()
        st.subheader("📈 Khuyến nghị tổng hợp")
        best = next((r for r in [r1d, r4h, r1h] if r["suggest"] in ("BUY", "SELL")), r1d)
        st.metric("Tín hiệu", best["suggest"])
        st.metric("Khung thời gian", best["frame"])
        st.metric("Take Profit", f"{best['tp']}" if best["tp"] else "-")
        st.metric("Cut Loss", f"{best['sl']}" if best["sl"] else "-")
        st.info("Công thức kết luận: RSI phân kỳ tăng + MACD cắt lên + Giá > MA20/50 + Volume tăng ⇒ BUY.")

st.caption("⚠️ Thông tin chỉ mang tính tham khảo, không phải khuyến nghị đầu tư.")
