import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
from textblob import TextBlob
from bs4 import BeautifulSoup
from sklearn.linear_model import Ridge
from datetime import datetime, timedelta
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Market Predator v24.1 [CENTRAL]", layout="wide", page_icon="🦅")

# --- AUTHENTICATION ---
CREDENTIALS = {"admin": "predator", "dad": "silver"}

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🦅 MARKET PREDATOR")
    st.markdown("### 🔒 Secure Access")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        with st.form("login"):
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if user in CREDENTIALS and CREDENTIALS[user] == pwd:
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = user
                    st.rerun()
                else:
                    st.error("Access Denied")
    st.stop()

# --- CSS: HACKER CONSOLE + COUNTDOWN ---
st.markdown("""
    <style>
    .stApp { background-color: #0e0e0e; color: #e0e0e0; font-family: 'Consolas', monospace; }
    [data-testid="stSidebar"] { background-color: #1a1a1a; }
    .stTextInput > div > div > input { color: white; background-color: #333333; }
    .stButton > button { background-color: #c92a2a; color: white; border: none; font-weight: bold; }
    
    /* HUD Styles */
    .hud-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #1a1a1a;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 20px;
    }
    .hud-symbol { font-size: 40px; font-weight: bold; color: #c92a2a; margin: 0; }
    .hud-price { font-size: 60px; font-weight: bold; color: white; margin: 0; }
    .hud-meta { text-align: right; color: gray; font-size: 14px; }
    .hud-timer { color: #3498db; font-weight: bold; font-size: 18px; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIC CLASSES ---
class BlackScholes:
    @staticmethod
    def price(S, K, T, r, sigma, type="C"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2) if type == "C" else K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

class DataHarvester:
    def get_combined_sentiment(self, ticker):
        try:
            headlines = []
            total = 0
            count = 0
            sources = [ticker.replace("=F", ""), "SPY"]
            triggers = ["GOVERNMENT", "FED", "RATE", "SHUTDOWN", "REOPEN", "WAR", "CPI", "JOBS"]
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            for sym in sources:
                try:
                    r = requests.get(f'https://finviz.com/quote.ashx?t={sym}', headers=headers, timeout=4)
                    soup = BeautifulSoup(r.content, 'html.parser')
                    table = soup.find(id='news-table')
                    if table:
                        for x in table.findAll('tr'):
                            if not x.a: continue
                            txt = x.a.get_text()
                            weight = 2.5 if any(k in txt.upper() for k in triggers) else 1.0
                            total += (TextBlob(txt).sentiment.polarity * weight)
                            count += 1
                            if len(headlines) < 8: headlines.append(f"• {txt}")
                            if count >= 20: break
                except: continue
            return (total/count) if count > 0 else 0, headlines
        except: return 0, ["Error fetching news"]

    def get_macro_pulse(self, ticker):
        try:
            d = yf.download("GC=F SI=F ^TNX", period="5d", progress=False).iloc[-1]
            gsr = d['GC=F']/d['SI=F']
            yield_chg = d['^TNX'] - 0.0 
            bias, note = 0, "Neutral"
            if "SLV" in ticker or "SI" in ticker:
                if gsr > 85: bias, note = 0.004, f"GSR Bullish ({gsr:.1f})"
                elif gsr < 70: bias, note = -0.002, f"GSR Bearish ({gsr:.1f})"
            elif "GLD" in ticker or "GC" in ticker:
                if yield_chg < 4.0: bias, note = 0.003, "Yields Calm"
            return bias, note
        except: return 0, "Macro Unavail"

    def get_hotlist(self, bias):
        try:
            targets = ["PLTR", "HL", "SOXL", "NVDL", "AG", "GDX", "PAAS", "TSLA", "MARA", "COIN", "AMD", "GME"]
            data = yf.download(targets, period="5d", progress=False)['Close']
            bulls, bears = [], []
            for s in targets:
                try:
                    p = data[s].dropna()
                    if len(p)<2: continue
                    last, prev = p.iloc[-1], p.iloc[-2]
                    pct = (last - prev) / prev
                    pot = (pct + 0.04) * 100 if pct > 0 else (pct - 0.04) * 100
                    if pct > 0: bulls.append((s, last, pot))
                    else: bears.append((s, last, pot))
                except: continue
            bulls.sort(key=lambda x: x[2], reverse=True)
            bears.sort(key=lambda x: x[2])
            txt = "\n🔥 INTRADAY HOTLIST (Top 5):\n" + "-"*35 + "\n"
            for x in bulls[:5]: txt += f"{x[0]:<6} | ${x[1]:<7.2f} | +{x[2]:.2f}%\n"
            txt += "\n❄️ INTRADAY COLDLIST (Top 5):\n" + "-"*35 + "\n"
            for x in bears[:5]: txt += f"{x[0]:<6} | ${x[1]:<7.2f} | {x[2]:.2f}%\n"
            return txt
        except: return "Hotlist Unavailable"

    def get_forecast_text(self, df):
        last = df.iloc[-1]
        p = (last['High']+last['Low']+last['Close'])/3
        r1, s1 = (2*p)-last['Low'], (2*p)-last['High']
        return f"DAY TRADING PIVOTS:\n[R1]: ${r1:.2f}\n>> PIVOT: ${p:.2f} <<\n[S1]: ${s1:.2f}"

# --- JS COUNTDOWN SCRIPT ---
countdown_js = """
<script>
function startTimer(duration, display) {
    var timer = duration, minutes, seconds;
    var interval = setInterval(function () {
        minutes = parseInt(timer / 60, 10);
        seconds = parseInt(timer % 60, 10);

        minutes = minutes < 10 ? "0" + minutes : minutes;
        seconds = seconds < 10 ? "0" + seconds : seconds;

        display.textContent = minutes + ":" + seconds;

        if (--timer < 0) {
            timer = duration;
        }
    }, 1000);
}
window.onload = function () {
    var fiveMinutes = 300,
        display = document.querySelector('#time');
    startTimer(fiveMinutes, display);
};
</script>
"""

# --- SIDEBAR & REFRESH LOGIC ---
with st.sidebar:
    st.markdown("## 🦅 PREDATOR\n**CONTROL PANEL**")
    
    # 1. SIDEBAR INPUT (Kept for convenience)
    sidebar_input = st.text_input("Enter Target Symbol", value="", placeholder="e.g. SLV", key="sb_input").upper()
    
    auto_update = st.checkbox("Auto-Update (5m)", value=True)
    if auto_update:
        st_autorefresh(interval=300000, limit=None, key="predator_counter")
    
    run_btn = st.button("RUN LIVE SCAN")
    
    st.markdown("---")
    st.markdown("**RECENT SCANS:**")
    if st.button("SLV"): sidebar_input = "SLV"
    if st.button("GLD"): sidebar_input = "GLD"
    if st.button("URA"): sidebar_input = "URA"
    
    st.markdown("---")
    if st.button("Logout"):
        st.session_state["authenticated"] = False
        st.rerun()

# --- STANDBY SCREEN (With Central Input) ---
# Logic: If no sidebar input, show central input.
# If central input is used, it acts as the "Active Ticker"

active_ticker = sidebar_input

if not active_ticker:
    st.info("🦅 PREDATOR IS READY. ENTER A TARGET TO BEGIN.")
    
    # --- NEW CENTRAL SEARCH BAR ---
    c_col1, c_col2, c_col3 = st.columns([1, 2, 1])
    with c_col2:
        central_input = st.text_input("🎯 ENTER SYMBOL:", placeholder="Type Ticker (e.g. NVDA) & Hit Enter", key="center_input").upper()
    
    if central_input:
        active_ticker = central_input # Promote central input to active
    else:
        st.stop() # Wait here until input is given

# --- MAIN LOGIC ---
# If we have an active ticker (from either box), run the app
if active_ticker:
    harvester = DataHarvester()
    try:
        stock = yf.Ticker(active_ticker)
        df = stock.history(period="2y", interval="1d")
        curr_price = df['Close'].iloc[-1]
        
        sent, headlines = harvester.get_combined_sentiment(active_ticker)
        macro_bias, macro_note = harvester.get_macro_pulse(active_ticker)
        hotlist_txt = harvester.get_hotlist(macro_bias)
        pivots_txt = harvester.get_forecast_text(df)
        
        try: iv = stock.option_chain(stock.options[0]).calls.iloc[0]['impliedVolatility'] * 100
        except: iv = 30
        
        df['Return'] = df['Close'].pct_change()
        df['Target'] = df['Return'].shift(-1)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df.dropna(inplace=True)
        model = Ridge(alpha=1.0).fit(df[['Return', 'RSI']], df['Target'])
        
        # --- HUD HEADER ---
        now_str = datetime.now().strftime('%H:%M:%S')
        st.markdown(f"""
        <div class="hud-container">
            <div>
                <p class="hud-symbol">{active_ticker}</p>
                <p style="color:gray; margin:0;">LAST SCAN: {now_str}</p>
            </div>
            <div>
                <p class="hud-price">${curr_price:.2f}</p>
            </div>
            <div style="text-align:right;">
                <p style="color:gray; margin:0;">NEXT REFRESH</p>
                <p class="hud-timer" id="time">05:00</p>
            </div>
        </div>
        {countdown_js} 
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("<h4 style='color: #3498db; text-align: center;'>OPTION SAFETY</h4>", unsafe_allow_html=True)
            st.progress(min(iv/50, 1.0))
            status = "GREEN LIGHT" if iv < 30 else "RED LIGHT"
            st.markdown(f"<p style='text-align: center; color: gray;'>{status} ({iv:.1f}%)</p>", unsafe_allow_html=True)
        with c2:
            st.markdown("<h4 style='color: #e74c3c; text-align: center;'>MACRO GRAVITY</h4>", unsafe_allow_html=True)
            st.progress(min(max(0.5 + (macro_bias * 50), 0.0), 1.0))
            st.markdown(f"<p style='text-align: center; color: gray;'>{macro_note}</p>", unsafe_allow_html=True)
        with c3:
            st.markdown("<h4 style='color: #2ecc71; text-align: center;'>AI MOMENTUM</h4>", unsafe_allow_html=True)
            st.progress(0.7)
            st.markdown(f"<p style='text-align: center; color: gray;'>Bullish Bias</p>", unsafe_allow_html=True)

        tab_ai, tab_news = st.tabs(["🔮 AI Oracle", "📰 Global Wire"])
        
        report_txt = f"--- AI ORACLE: LIVE STREAM ---\nPremiums: {status}\nMacro:    {macro_note}\n" + "-"*65 + "\n"
        report_txt += f"{'DATE':<10} | {'OPEN':<8} | {'HIGH':<8} | {'LOW':<8} | {'CLOSE':<8}\n" + "-"*65 + "\n"
        
        run_p = curr_price
        atr_pct = df['ATR'].iloc[-1] / curr_price
        for i in range(14):
            d = datetime.now() + timedelta(days=i+1)
            if d.weekday() >= 5: continue
            move = model.predict([[0.001, df['RSI'].iloc[-1]]])[0] + macro_bias + (sent*0.001)
            run_p *= (1+move)
            high_p = run_p * (1 + (atr_pct * 0.5))
            low_p = run_p * (1 - (atr_pct * 0.5))
            report_txt += f"{d.strftime('%b %d'):<10} | ${run_p:<8.2f} | ${high_p:<8.2f} | ${low_p:<8.2f} | ${run_p:<8.2f}\n"
            
        report_txt += "\n📅 8-WEEK PROJECTION:\n" + "-"*30 + "\n"
        w_p = curr_price
        for w in range(1, 9):
            w_p *= (1 + (move * 5))
            report_txt += f"Week {w}: ${w_p:.2f}\n"
            
        report_txt += "\n🛡️ STRATEGIC ACTION: " + ("DEEP ITM CALLS\n" if iv > 40 else "GROWTH CALLS\n")
        report_txt += f"Target Strike: ${curr_price * 1.05:.0f}\nExp: >120 Days"

        with tab_ai: st.code(report_txt, language="text")
        with tab_news: st.code(f"{pivots_txt}\n\n=== 📰 GLOBAL WIRE ===\n" + "".join([f"{h}\n" for h in headlines]) + f"\n{hotlist_txt}", language="text")

    except Exception as e: st.error(f"Scan Error: {e}")
