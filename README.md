# All-India-Algo-Dev-Challenge-2025---Nubra
# Contextual Bandit Trading Decision Tool

A Python toolkit and Streamlit dashboard for evaluating simple trading playbooks using a contextual Thompson Sampling bandit over engineered market features, with backtesting, cost modeling, and optional parameter search. [See code references in main.py, app.py, and code.py]

## Overview

- Data: Downloads historical/intraday OHLCV via yfinance with auto-adjusted prices.  
- Features: Realized volatility, ATR(5), VWAP deviation z-score, rolling volume percentile, gap z-score, imbalance proxy, and time-of-day bucket.  
- Strategies: Open-range breakout (ORB), VWAP mean reversion, and pullback-in-trend.  
- Policy: Contextual Thompson Sampling selects among strategies plus a no-trade arm.  
- Execution: Size by ATR-based stop distance with a per-trade risk budget and trading costs; equity curve and trades are recorded.  
- Interfaces: CLI script (main.py) and Streamlit app (app.py); an additional standalone backtest variant is in code.py.  

## Features

- Thompson Sampling contextual bandit over a 6D feature vector: rv, vwap_dev_z, vol_pct, gap_z, imb_proxy, tod_bucket.  
- ATR-based stop placement with R_MULT_ATR and budget-based sizing via HEAT_BP.  
- Trading frictions: half-spread and commission in bps of notional.  
- “Live” intraday mode and grid-search optimization for key parameters.  
- Equity curve plotting with trade markers and file output.

