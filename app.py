import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Import your strategy functions
from main import build_features, ThompsonBandit, simulate, ARMS, SEED

st.title("Trading Decision Dashboard")

# Sidebar inputs
st.sidebar.header("Parameters")
ticker = st.sidebar.text_input("Ticker", value="AAPL")
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"], index=4)
start = st.sidebar.date_input("Start Date", value=datetime(2025, 8, 21))
end = st.sidebar.date_input("End Date", value=datetime(2025, 8, 21))
optimize = st.sidebar.checkbox("Optimize Parameters", value=False)

if st.sidebar.button("Run Analysis"):
    st.write(f"Fetching data for {ticker} from {start} to {end} with interval {interval}...")
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        st.error("No data fetched. Check ticker/dates/interval.")
    else:
        df.index = pd.to_datetime(df.index)
        df_feat = build_features(df, ticker)
        bandit = ThompsonBandit(n_features=6, arms=ARMS, seed=SEED)
        trades, equity = simulate(df_feat, bandit)
        total_pnl = sum(t["pnl"] for t in trades) if trades else 0.0
        wins = sum(1 for t in trades if t["pnl"] > 0)
        losses = sum(1 for t in trades if t["pnl"] <= 0)
        win_rate = (wins / len(trades) * 100) if trades else 0.0
        st.subheader(f"Trade Summary for {ticker}")
        st.write(f"Trades: {len(trades)} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.2f}% | Total PnL: {total_pnl:.2f}")
        if trades:
            st.write(pd.DataFrame(trades))
        else:
            st.info("No trades taken in this period.")
        st.line_chart(equity)
        # Show contextual bandit decisions for each row
        st.subheader("Contextual Bandit Decisions")
        decisions = []
        for i in range(len(df_feat)):
            row = df_feat.iloc[i]
            x = [float(row.get(k, 0)) for k in ['rv','vwap_dev_z','vol_pct','gap_z','imb_proxy','tod_bucket']]
            allowed = [True]*len(ARMS)
            arm_idx = bandit.select(np.array(x), allowed)
            decisions.append(ARMS[arm_idx])
        df_feat['decision'] = decisions
        st.write(df_feat[['Close','decision']])

        # Calculate PnL for all trades suggested by the model
        trades = []
        pnl = 0.0
        for i in range(len(df_feat)-1):
            dec = df_feat['decision'].iloc[i]
            if dec != 'no_trade':
                entry_px = df_feat['Close'].iloc[i]
                exit_px = df_feat['Close'].iloc[i+1]
                trade_pnl = exit_px - entry_px
                trades.append({
                    'entry_time': df_feat.index[i],
                    'exit_time': df_feat.index[i+1],
                    'decision': dec,
                    'entry_px': entry_px,
                    'exit_px': exit_px,
                    'pnl': trade_pnl
                })
                pnl += trade_pnl
        st.subheader(f"Model-Suggested Trades & PnL")
        st.write(pd.DataFrame(trades))
        st.success(f"Total PnL for all model-suggested trades: {pnl:.2f}")
        # Optionally add parameter optimization here
        if optimize:
            st.info("Grid search optimization coming soon!")
