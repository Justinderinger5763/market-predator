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

# --- PAGE CONFIG ---
st.set_page_config(page_title="Market Predator v21 [WEB]", layout="wide", page_icon="🦅")

# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- LOGIC CLASSES (Identical to your Desktop App) ---
class BlackScholes:
    @staticmethod
    def price(S, K, T, r, sigma, type="C"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if type == "C":
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

class DataHarvester:
    def get_combined_sentiment(self, ticker):
        try:
            headlines = []
            total_score = 0
            count = 0
            clean_ticker = ticker.replace("=F", "")
            sources = [clean_ticker, "SPY"]
            macro_triggers = ["GOVERNMENT", "FED", "RATE", "SHUTDOWN", "REOPEN", "WAR", "CPI", "JOBS", "TRUMP", "BIDEN"]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            for sym in sources:
                url = f'https://finviz.com/quote.ashx?t={sym}'
                req = requests.get(url, headers=headers, timeout=4)
                if req.status_code != 200: continue
                
                soup = BeautifulSoup(req.content, 'html.parser')
                news_table = soup.find(id='news-table')
                
                if news_table:
                    for x in news_table.findAll('tr'):
                        try:
                            if not x.a: continue
                            text = x.a.get_text()
                            blob = TextBlob(text)
                            
                            is_macro = any(k in text.upper() for k in macro_triggers)
                            weight = 2.5 if is_macro else 1.0
                            
                            total_score += (blob.sentiment.polarity * weight)
                            count += 1
                            
                            if len(headlines) < 8:
                                prefix = "🚨 [MACRO]" if sym == "SPY" else "📰"
                                headlines.append(f"{prefix} {text}")
                            if count >= 20: break 
                        except: continue
            
            avg_score = (total_score / count) if count > 0 else 0
            return avg_score, headlines
        except:
            return 0, ["Error fetching news (Cloud Blocked)"]

    def get_macro_pulse(self, current_ticker):
        try:
            data = yf.download("GC=F SI=F ^TNX", period="5d", interval="1d", progress=False)
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            
            last = data.iloc[-1]
            gsr = last['GC=F'] / last['SI=F']
            yield_change = last['^TNX'] - data.iloc[-2]['^TNX']
            
            bias = 0
            note = "Neutral"
            is_gold = "GLD" in current_ticker or "GC" in current_ticker
            is_silver = "SLV" in current_ticker or "SI" in current_ticker
            
            if is_silver:
                if gsr > 85: bias, note = 0.004, f"GSR Bullish ({gsr:.1f})"
                elif gsr < 70: bias, note = -0.002, f"GSR Bearish ({gsr:.1f})"
            elif is_gold:
                if yield_change < -0.05: bias, note = 0.003, "Yields Drop (Bullish)"
                elif yield_change > 0.05: bias, note = -0.004, "Yields Spike (Bearish)"
            
            if yield_change > 0.10: bias -= 0.002
            return bias, note
        except: return 0, "Macro Unavail"

    def get_market_hotlist(self, macro_bias):
        try:
            targets = ["PLTR", "HL", "SOXL", "NVDL", "AG", "GDX", "PAAS", "TSLA", "MARA", "COIN", "AMD", "GME"]
            data = yf.download(targets, period="5d", interval="1d", progress=False)
            if isinstance(data.columns, pd.MultiIndex): closes = data['Close']
            else: return pd.DataFrame()

            hotlist = []
            for sym in targets:
                try:
                    prices = closes[sym].dropna()
                    if len(prices) < 2: continue
                    last, prev = prices.iloc[-1], prices.iloc[-2]
                    pct = (last - prev) / prev
                    daily_vol = 0.04
                    
                    if pct > 0:
                        pot = (pct + daily_vol) * 100
                        hotlist.append({"Ticker": sym, "Price": last, "Type": "🔥 Long", "Potential": pot})
                    else:
                        pot = (pct - daily_vol) * 100
                        hotlist.append({"Ticker": sym, "Price": last, "Type": "❄️ Short", "Potential": pot})
                except: continue
            
            return pd.DataFrame(hotlist).sort_values(by="Potential", ascending=False)
        except: return pd.DataFrame()

# --- MAIN APP UI ---
st.title("🦅 MARKET PREDATOR [WEB]")

with st.sidebar:
    st.header("Control Panel")
    ticker = st.text_input("Enter Ticker", value="SLV").upper()
    run_btn = st.button("RUN ANALYSIS", type="primary")
    st.info("Web Version v21.0")

if run_btn:
    harvester = DataHarvester()
    
    with st.spinner(f"Hunting data for {ticker}..."):
        try:
            # 1. Fetch Data
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y", interval="1d")
            curr_price = df['Close'].iloc[-1]
            
            # 2. Get Analysis
            sent, headlines = harvester.get_combined_sentiment(ticker)
            macro_bias, macro_note = harvester.get_macro_pulse(ticker)
            hotlist_df = harvester.get_market_hotlist(macro_bias)
            
            # 3. Calculate IV & Safety
            try:
                exp = stock.options[0]
                opt = stock.option_chain(exp).calls
                iv = opt.iloc[(opt['strike'] - curr_price).abs().argsort()[:1]]['impliedVolatility'].values[0] * 100
            except: iv = 30
            
            safety_status = "🟢 SAFE" if iv < 30 else "🔴 HIGH RISK"
            if iv > 45: safety_status = "🔴 VOLATILE (>45%)"
            
            # 4. AI Prediction
            df['Return'] = df['Close'].pct_change()
            df['Target'] = df['Return'].shift(-1)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
            df.dropna(inplace=True)
            
            model = Ridge(alpha=1.0).fit(df[['Return', 'RSI']], df['Target'])
            
            # --- DASHBOARD LAYOUT ---
            
            # Top Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${curr_price:.2f}")
            c2.metric("Implied Volatility", f"{iv:.1f}%", delta="-Safe" if iv<30 else "+Risky", delta_color="inverse")
            c3.metric("Macro Trend", macro_note, delta=f"{macro_bias*1000:.1f} pts")

            # Prediction Table
            st.subheader("🔮 AI Oracle Forecast (14 Days)")
            
            future_prices = []
            dates = []
            run_p = curr_price
            
            # Generate 14 day prediction
            for i in range(14):
                d = datetime.now() + timedelta(days=i+1)
                if d.weekday() >= 5: continue
                
                move = model.predict([[0.001, df['RSI'].iloc[-1]]])[0] + macro_bias + (sent * 0.001)
                run_p = run_p * (1 + move)
                future_prices.append(run_p)
                dates.append(d.strftime('%b %d'))
            
            chart_data = pd.DataFrame({"Date": dates, "Predicted Price": future_prices})
            st.line_chart(chart_data.set_index("Date"))
            
            # Strategy Recommendation
            target = future_prices[-1]
            slope = (target - curr_price) / curr_price
            
            st.divider()
            s1, s2 = st.columns([2, 1])
            
            with s1:
                st.subheader("🛡️ Strategic Action")
                if slope > 0:
                    if iv > 40:
                        st.error(f"**BUY DEEP ITM CALLS** (Strike: ${curr_price*0.9:.0f})")
                        st.caption("Reason: Bullish trend, but IV is high. Avoid premium burn.")
                    else:
                        st.success(f"**BUY GROWTH CALLS** (Strike: ${curr_price + (target-curr_price)*0.6:.0f})")
                        st.caption(f"Reason: Bullish trend to ${target:.2f}. Low IV allows leverage.")
                else:
                    st.warning(f"**BUY PUTS / HEDGE** (Strike: ${curr_price*0.95:.0f})")
                    st.caption("Reason: Bearish trend detected.")

            with s2:
                st.subheader("📰 Global Wire")
                for h in headlines:
                    st.text(h)

            st.divider()
            st.subheader("🔥 Intraday Hotlist")
            if not hotlist_df.empty:
                st.dataframe(hotlist_df, use_container_width=True)
            else:
                st.write("No high-beta setups found right now.")

        except Exception as e:
            st.error(f"Scan Error: {str(e)}")
