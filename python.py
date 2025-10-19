import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="AI Analyst Pro - XAUUSD", layout="wide")
st.title("🤖 AI chuyên gia phân tích vàng (XAUUSD – Pro v2 + Chart)")
st.caption(
    "Lấy dữ liệu thực từ Yahoo Finance → phân tích RSI, MACD, MA20/50, Volume. "
    "Hiển thị biểu đồ nến + RSI + MACD và khuyến nghị Buy/Sell (+ TP/SL)."
)

# --- chỉ báo ---
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def sma(s, n): return s.rolling(n).mean()
def rsi(c, n=14):
    d = c.diff(); u = np.where(d>0, d, 0); l = np.where(d<0, -d, 0)
    up = pd.Series(u,index=c.index).rolling(n).mean()
    down = pd.Series(l,index=c.index).rolling(n).mean()
    rs = up/down.replace(0,np.nan)
    return 100-(100/(1+rs))
def macd(c, f=12, s=26, sig=9):
    m = ema(c,f)-ema(c,s); sg = ema(m,sig); return m, sg, m-sg
def atr(df, n=14):
    tr = pd.concat([(df.High-df.Low),
                    (df.High-df.Close.shift()).abs(),
                    (df.Low-df.Close.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean()

# --- tải dữ liệu ---
@st.cache_data(ttl=600)
def fetch(symbol="XAUUSD=X", interval="1h", period="90d"):
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    df.rename(columns=str.capitalize, inplace=True)
    return df

# --- phân tích ---
def analyze(df):
    r = {"trend":"-", "rsi":None,"ma20":None,"ma50":None,
         "macd_cross":False,"vol_spike":False,"suggest":"HOLD","tp":None,"sl":None}
    if df.empty: r["trend"]="Không có dữ liệu"; return r
    df["RSI"]=rsi(df.Close); df["MA20"]=sma(df.Close,20); df["MA50"]=sma(df.Close,50)
    m,s,h=macd(df.Close); df["MACD"],df["SIGNAL"],df["HIST"]=m,s,h; df["ATR"]=atr(df)
    last,prev=df.iloc[-1],df.iloc[-2]
    r["rsi"],r["ma20"],r["ma50"]=last.RSI,last.MA20,last.MA50
    r["macd_cross"]=prev.MACD<=prev.SIGNAL and last.MACD>last.SIGNAL
    r["vol_spike"]=last.Volume>1.5*df.Volume.rolling(20).mean().iloc[-1]
    r["trend"]="Tăng" if last.Close>last.MA20>last.MA50 else "Giảm"
    if r["macd_cross"] and r["trend"]=="Tăng" and r["vol_spike"]: r["suggest"]="BUY"
    elif r["rsi"]>70: r["suggest"]="SELL"
    atrv=last.ATR if not pd.isna(last.ATR) else 0
    if r["suggest"]=="BUY": r["tp"],r["sl"]=round(last.Close+1.5*atrv,2),round(last.Close-1*atrv,2)
    elif r["suggest"]=="SELL": r["tp"],r["sl"]=round(last.Close-1.5*atrv,2),round(last.Close+1*atrv,2)
    return r

# --- vẽ biểu đồ ---
def make_chart(df,frame):
    fig=go.Figure()
    # nến + MA
    fig.add_trace(go.Candlestick(x=df.index,open=df.Open,high=df.High,
                                 low=df.Low,close=df.Close,
                                 name="Giá",increasing_line_color="green",
                                 decreasing_line_color="red"))
    fig.add_trace(go.Scatter(x=df.index,y=df.MA20,mode="lines",name="MA20",line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df.index,y=df.MA50,mode="lines",name="MA50",line=dict(color="blue")))
    fig.update_layout(title=f"XAUUSD ({frame})",xaxis_rangeslider_visible=False,height=400)
    return fig

def make_rsi(df):
    f=go.Figure(); f.add_trace(go.Scatter(x=df.index,y=df.RSI,name="RSI",line=dict(color="purple")))
    f.add_hrect(y0=30,y1=70,fillcolor="gray",opacity=0.2,line_width=0)
    f.update_layout(title="RSI(14)",height=200); return f

def make_macd(df):
    f=go.Figure()
    f.add_trace(go.Scatter(x=df.index,y=df.MACD,name="MACD",line=dict(color="orange")))
    f.add_trace(go.Scatter(x=df.index,y=df.SIGNAL,name="Signal",line=dict(color="blue")))
    f.add_trace(go.Bar(x=df.index,y=df.HIST,name="Histogram",marker_color="gray"))
    f.update_layout(title="MACD",height=200); return f

# --- giao diện ---
st.sidebar.header("Cấu hình")
frame=st.sidebar.selectbox("Khung:",["1h","4h","1d"])
df=fetch("XAUUSD=X",frame,"90d")

st.subheader("💰 Giá thời gian thực")
if not df.empty:
    last=df.iloc[-1]; prev=df.iloc[-2] if len(df)>1 else last
    st.metric("XAUUSD (USD)",f"{last.Close:.2f}",f"{last.Close-prev.Close:+.2f}")
else:
    st.warning("Không tải được dữ liệu từ Yahoo Finance.")

if st.button("🔍 Phân tích"):
    res=analyze(df)
    st.markdown("### 📊 Kết quả")
    st.dataframe(pd.DataFrame([
        ["Xu hướng",res["trend"]],
        ["RSI(14)",f"{res['rsi']:.2f}" if res['rsi'] else "-"],
        ["Giá>MA20/50","Có" if res["trend"]=="Tăng" else "Không"],
        ["MACD","Cắt lên" if res["macd_cross"] else "Chưa"],
        ["Volume","Tăng mạnh" if res["vol_spike"] else "Bình thường"],
        ["Khuyến nghị",res["suggest"]],
        ["TP",res["tp"] if res["tp"] else "-"],
        ["SL",res["sl"] if res["sl"] else "-"]
    ],columns=["Chỉ tiêu","Giá trị"]),use_container_width=True)

    st.plotly_chart(make_chart(df,frame.upper()),use_container_width=True)
    st.plotly_chart(make_rsi(df),use_container_width=True)
    st.plotly_chart(make_macd(df),use_container_width=True)

st.caption("⚠️ Dữ liệu lấy từ Yahoo Finance (cập nhật trễ ~5-10 phút). Không phải lời khuyên đầu tư.")
