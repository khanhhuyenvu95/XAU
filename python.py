import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz

# =========================
# CẤU HÌNH TRANG & HEADER
# =========================
st.set_page_config(page_title="XAUUSD AI Analyst • Streamlit", layout="wide")
st.title("🤖 Chuyên gia AI phân tích XAUUSD (1h • 4h • 1D)")
st.caption(
    "AI vào vai chuyên gia đầu tư Vàng với 30+ năm kinh nghiệm: phân tích RSI, MACD, MA20/50, Volume, "
    "phát hiện đảo chiều/bứt phá, và gợi ý Buy/Sell + TP/SL. "
    "Nguồn dữ liệu tài chính đáng tin cậy (yfinance/Yahoo Finance) – phù hợp để tham chiếu với TradingView."
)

# =========================
# TIỆN ÍCH TÍNH CHỈ BÁO
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
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

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

def resample_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    # Resample 1H -> 4H OHLCV
    rule = '4H'
    agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    return df_1h.resample(rule).apply(agg).dropna()

def latest_price_row(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    return df.iloc[-1]

# =========================
# TẢI DỮ LIỆU (CACHE)
# =========================
@st.cache_data(show_spinner=False)
def fetch_yf(symbol: str, period: str, interval: str) -> pd.DataFrame:
    return yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)

@st.cache_data(show_spinner=False)
def fetch_price_and_frames(symbol_spot: str = "XAUUSD=X"):
    """
    Trả về:
      - df_1h (period 90d)
      - df_4h (resample từ 1h)
      - df_1d (period 2y)
      - df_fut (GC=F) dùng khi Volume spot bị thiếu
    """
    df_1h = fetch_yf(symbol_spot, period="90d", interval="1h")
    df_1d = fetch_yf(symbol_spot, period="2y", interval="1d")

    # 4H từ 1H
    df_4h = resample_4h(df_1h.copy()) if not df_1h.empty else pd.DataFrame()

    # Lấy volume proxy từ gold futures (GC=F) nếu volume XAUUSD thiếu
    df_fut_1h = fetch_yf("GC=F", period="90d", interval="1h")
    df_fut_4h = resample_4h(df_fut_1h.copy()) if not df_fut_1h.empty else pd.DataFrame()
    df_fut_1d = fetch_yf("GC=F", period="2y", interval="1d")

    return df_1h, df_4h, df_1d, df_fut_1h, df_fut_4h, df_fut_1d

# =========================
# PHÂN TÍCH THEO KHUNG
# =========================
def evaluate_frame(df: pd.DataFrame, df_vol_proxy: pd.DataFrame, name: str):
    """
    Tính RSI(14), MACD(12,26,9), MA20/50, ATR(14), phát hiện:
     - MACD cắt lên (và dưới trục 0 → mạnh hơn)
     - RSI phân kỳ tăng (heuristic đơn giản)
     - Bứt phá MA20/50 (Close > cả MA20 & MA50)
     - Volume spike & xác nhận
    Trả về dict kết quả + khuyến nghị khung đó.
    """
    result = {
        "frame": name,
        "close": None,
        "rsi": None,
        "macd_cross_up": False,
        "macd_below_zero_then_up": False,
        "ma20": None,
        "ma50": None,
        "price_above_ma2050": False,
        "bullish_div_rsi": False,
        "volume_spike": False,
        "trend_note": "",    # 'đảo chiều' / 'bứt phá' / 'trung tính'
        "suggest": "",       # BUY / SELL / HOLD
        "tp": None,
        "sl": None
    }

    if df is None or df.empty:
        result["trend_note"] = "Không có dữ liệu"
        result["suggest"] = "HOLD"
        return result

    df = df.copy()
    # Bù volume nếu thiếu
    if 'Volume' in df.columns and df['Volume'].dropna().sum() == 0 and df_vol_proxy is not None and not df_vol_proxy.empty:
        # canh theo index gần nhất
        vol_proxy = df_vol_proxy['Volume'].reindex(df.index, method='nearest')
        df['Volume'] = vol_proxy

    # Tính chỉ báo
    df['RSI14'] = rsi(df['Close'])
    macd_line, signal_line, hist = macd(df['Close'])
    df['MACD'] = macd_line
    df['SIGNAL'] = signal_line
    df['HIST'] = hist
    df['MA20'] = sma(df['Close'], 20)
    df['MA50'] = sma(df['Close'], 50)
    df['ATR14'] = atr(df, 14)

    last = df.iloc[-1]
    result["close"] = float(last['Close'])
    result["rsi"] = float(last['RSI14']) if pd.notna(last['RSI14']) else None
    result["ma20"] = float(last['MA20']) if pd.notna(last['MA20']) else None
    result["ma50"] = float(last['MA50']) if pd.notna(last['MA50']) else None
    result["price_above_ma2050"] = (
        pd.notna(last['MA20']) and pd.notna(last['MA50']) and
        last['Close'] > last['MA20'] and last['Close'] > last['MA50']
    )

    # MACD cắt lên
    if len(df) >= 2:
        prev = df.iloc[-2]
        cross_up = (prev['MACD'] <= prev['SIGNAL']) and (last['MACD'] > last['SIGNAL'])
        result["macd_cross_up"] = bool(cross_up)
        result["macd_below_zero_then_up"] = bool(cross_up and (prev['MACD'] < 0 and prev['SIGNAL'] < 0))

    # RSI phân kỳ tăng (heuristic: LL price & HL RSI trong ~40 nến gần nhất)
    lookback = min(40, len(df) - 1)
    if lookback > 10:
        window = df.iloc[-lookback:]
        # tìm hai đáy cục bộ
        price = window['Close']
        rsi_ser = window['RSI14']
        # đáy 1: min đầu nửa đầu; đáy 2: min nửa sau
        left = window.iloc[:lookback//2]
        right = window.iloc[lookback//2:]
        if not left.empty and not right.empty:
            p1_idx = left['Close'].idxmin()
            p2_idx = right['Close'].idxmin()
            if p1_idx in rsi_ser.index and p2_idx in rsi_ser.index:
                p1, p2 = price.loc[p1_idx], price.loc[p2_idx]
                r1, r2 = rsi_ser.loc[p1_idx], rsi_ser.loc[p2_idx]
                if pd.notna(r1) and pd.notna(r2):
                    # Bullish divergence: price makes lower low, RSI makes higher low
                    result["bullish_div_rsi"] = bool((p2 < p1) and (r2 > r1))

    # Volume spike: volume > 1.5 * SMA20(volume) và nến tăng
    if 'Volume' in df.columns and df['Volume'].notna().any():
        vol_sma20 = df['Volume'].rolling(20).mean()
        if pd.notna(vol_sma20.iloc[-1]):
            result["volume_spike"] = bool((df['Volume'].iloc[-1] > 1.5 * vol_sma20.iloc[-1]) and
                                          (df['Close'].iloc[-1] > df['Open'].iloc[-1]))

    # Phân loại xu hướng: đảo chiều hay bứt phá
    if result["macd_cross_up"] and result["macd_below_zero_then_up"]:
        result["trend_note"] = "Đảo chiều tăng (MACD cắt lên dưới trục 0)"
    elif result["price_above_ma2050"] and result["volume_spike"]:
        result["trend_note"] = "Bứt phá (vượt MA20/50 kèm Volume)"
    elif result["price_above_ma2050"]:
        result["trend_note"] = "Xu hướng tăng (trên MA20/50)"
    else:
        result["trend_note"] = "Trung tính / yếu"

    # Tín hiệu tổng hợp BUY/SELL (theo yêu cầu)
    bullish_combo = (
        bool(result["bullish_div_rsi"]) and
        bool(result["macd_cross_up"]) and
        bool(result["price_above_ma2050"]) and
        bool(result["volume_spike"])
    )
    bearish_combo = (
        (result["rsi"] is not None and result["rsi"] > 70) and
        (not result["price_above_ma2050"]) and
        (not result["macd_cross_up"])
    )

    atr_val = float(last['ATR14']) if pd.notna(last['ATR14']) else None
    if bullish_combo:
        result["suggest"] = "BUY"
        if atr_val:
            entry = float(last['Close'])
            result["tp"] = round(entry + 1.5 * atr_val, 2)
            result["sl"] = round(entry - 1.0 * atr_val, 2)
    elif bearish_combo:
        result["suggest"] = "SELL"
        if atr_val:
            entry = float(last['Close'])
            result["tp"] = round(entry - 1.5 * atr_val, 2)
            result["sl"] = round(entry + 1.0 * atr_val, 2)
    else:
        result["suggest"] = "HOLD"

    return result

def build_indicator_table(res):
    """
    Trả về DataFrame gồm:
    STT | Chỉ tiêu | Giá trị | Đáp ứng tín hiệu tăng? | Khuyến nghị
    """
    rows = []
    stt = 1

    # 1. Xu hướng (đảo chiều / bứt phá / …)
    rows.append([
        stt, "Xu hướng (Trend)", res["trend_note"], 
        "Có" if (("bứt phá" in res["trend_note"].lower()) or ("đảo chiều" in res["trend_note"].lower()) or ("tăng" in res["trend_note"].lower() and "trung" not in res["trend_note"].lower())) else "Không",
        res["suggest"]
    ])
    stt += 1

    # 2. RSI
    rsi_val = "-" if res["rsi"] is None else f"{res['rsi']:.2f}"
    rsi_flag = "Có" if (res["rsi"] is not None and (res["rsi"] < 30 or res["bullish_div_rsi"] or res["rsi"] >= 50)) else "Không"
    rows.append([
        stt, "RSI(14)", rsi_val + (" • Phân kỳ tăng" if res["bullish_div_rsi"] else ""),
        rsi_flag, res["suggest"]
    ])
    stt += 1

    # 3. MACD
    macd_note = "MACD cắt lên" if res["macd_cross_up"] else "Chưa cắt lên"
    if res["macd_below_zero_then_up"]:
        macd_note += " (dưới 0 ⇒ tín hiệu mạnh)"
    rows.append([
        stt, "MACD(12,26,9)", macd_note,
        "Có" if res["macd_cross_up"] else "Không",
        res["suggest"]
    ])
    stt += 1

    # 4. MA20/50
    ma_val = f"MA20: {res['ma20']:.2f} • MA50: {res['ma50']:.2f}" if (res["ma20"] and res["ma50"]) else "-"
    rows.append([
        stt, "Giá vượt MA20/50", ma_val,
        "Có" if res["price_above_ma2050"] else "Không",
        res["suggest"]
    ])
    stt += 1

    # 5. Volume
    rows.append([
        stt, "Volume xác nhận", "Spike & nến tăng" if res["volume_spike"] else "Bình thường / thiếu dữ liệu",
        "Có" if res["volume_spike"] else "Không",
        res["suggest"]
    ])

    df = pd.DataFrame(rows, columns=["STT", "Chỉ tiêu", "Giá trị (thực thời)", "Đáp ứng tín hiệu tăng?", "Khuyến nghị"])
    return df

def overall_recommendation(r1h, r4h, r1d):
    # Ưu tiên 1D > 4H > 1H
    for r in [r1d, r4h, r1h]:
        if r and r["suggest"] in ("BUY", "SELL"):
            return r["suggest"], r["tp"], r["sl"], r["frame"]
    # nếu không rõ ràng
    return "HOLD", None, None, None

# =========================
# SIDEBAR & NHẬP LIỆU
# =========================
with st.sidebar:
    st.header("Thiết lập")
    symbol = st.text_input(
        "Mã giao dịch:",
        value="XAUUSD=X",
        help="Mặc định: Spot Gold trên Yahoo Finance. Bạn có thể nhập GC=F (Gold Futures) hoặc mã khác tương thích Yahoo Finance."
    )
    st.caption("Gợi ý: XAUUSD=X (Spot), GC=F (Futures). 4H được tổng hợp từ 1H.")

# =========================
# HÀNG GIÁ “THỰC THỜI”
# =========================
tz = pytz.timezone("Asia/Bangkok")
st.subheader("🎯 Giá thời gian thực")
colp, colt = st.columns([1, 2])

# tải 1m/5m để snapshot? Yahoo giới hạn. Lấy 1h mới nhất để hiển thị ổn định
df_last_1h = fetch_yf(symbol, period="5d", interval="1h")
if not df_last_1h.empty:
    last_row = df_last_1h.iloc[-1]
    prev_row = df_last_1h.iloc[-2] if len(df_last_1h) >= 2 else None
    last_price = float(last_row['Close'])
    delta = (last_price - float(prev_row['Close'])) if prev_row is not None else 0.0
    colp.metric(label=f"{symbol} (khung 1h gần nhất)", value=f"{last_price:.2f}", delta=f"{delta:+.2f}")
    colt.write(f"Cập nhật: {last_row.name.tz_localize('UTC').tz_convert(tz).strftime('%Y-%m-%d %H:%M:%S %Z')}")
else:
    colp.metric(label=f"{symbol}", value="N/A")
    colt.write("Chưa lấy được giá gần nhất.")

st.divider()

# =========================
# NÚT "TÌM KIẾM" & PHÂN TÍCH
# =========================
if st.button("🔎 Tìm kiếm & Phân tích"):
    with st.spinner("Đang tải dữ liệu & phân tích chỉ báo..."):
        df_1h, df_4h, df_1d, fut_1h, fut_4h, fut_1d = fetch_price_and_frames(symbol)

        r1h = evaluate_frame(df_1h, fut_1h, "1H")
        r4h = evaluate_frame(df_4h, fut_4h, "4H")
        r1d = evaluate_frame(df_1d, fut_1d, "1D")

        # Hiển thị từng khung với bảng tiêu chuẩn
        for res in [r1h, r4h, r1d]:
            st.markdown(f"### ⏱ Khung {res['frame']}")
            top_cols = st.columns(4)
            top_cols[0].write(f"**Close:** {res['close']:.2f}" if res["close"] else "**Close:** -")
            top_cols[1].write(f"**RSI14:** {res['rsi']:.2f}" if res["rsi"] else "**RSI14:** -")
            top_cols[2].write(f"**MA20/50:** {res['ma20']:.2f} / {res['ma50']:.2f}" if (res["ma20"] and res["ma50"]) else "**MA20/50:** -")
            top_cols[3].write(f"**Xu hướng:** {res['trend_note']}")

            table = build_indicator_table(res)
            st.dataframe(table, use_container_width=True)

        st.divider()

        # KHUYẾN NGHỊ TỔNG HỢP
        final_sig, tp, sl, base_tf = overall_recommendation(r1h, r4h, r1d)
        st.markdown("## 🧭 Khuyến nghị tổng hợp")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Tín hiệu", final_sig)
        colB.metric("Khung tham chiếu", base_tf if base_tf else "-")
        colC.metric("Take Profit", f"{tp:.2f}" if tp else "-")
        colD.metric("Cut Loss", f"{sl:.2f}" if sl else "-")

        st.info(
            "Nguyên tắc kết luận: **RSI phân kỳ tăng + MACD cắt lên + Giá vượt MA20/50 + Volume tăng** ⇒ ưu tiên BUY. "
            "Ngược lại nếu RSI quá mua, dưới MA và MACD chưa xác nhận ⇒ cân nhắc SELL. "
            "TP/SL gợi ý dựa trên **ATR(14)** (1.5×ATR cho TP, 1.0×ATR cho SL)."
        )

st.caption(
    "⚠️ Miễn trừ trách nhiệm: Thông tin chỉ nhằm mục đích tham khảo và không phải khuyến nghị đầu tư. "
    "Luôn quản trị rủi ro và tự chịu trách nhiệm với quyết định của bạn."
)
