import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# =========================
# Config
# =========================
SEED = 42
TICKER = "AAPL"
START = "2023-01-01"
END = "2025-01-01"
BAR = "1d"  # use "1m"/"5m" for intraday once ready

# Costs
HALF_SPREAD = 0.005  # $/share per side
COMMISSION_BP = 1.0  # bps of notional per trade

# Risk sizing
R_MULT_ATR = 1.2  # stop distance multiple of ATR(5)
HEAT_BP = 20  # per-trade risk budget as bps of equity
EQUITY0 = 100_000.0

# Strategy parameters
OPEN_RANGE_MIN_BARS = 5  # with daily bars, ORB is illustrative; with 1m use 30 bars
VWAP_Z_ENTRY = 1.5
RV_WIN = 20
ATR_WIN = 5

# =========================
# Utilities
# =========================
def rng(seed=SEED):
    return np.random.default_rng(seed)

def close_series(df: pd.DataFrame) -> pd.Series:
    c = df["Close"]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return pd.to_numeric(c, errors="coerce").astype(float)

def hlc(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    H = pd.to_numeric(df["High"], errors="coerce").astype(float)
    L = pd.to_numeric(df["Low"], errors="coerce").astype(float)
    C = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    return H, L, C

def atr(df: pd.DataFrame, win=ATR_WIN) -> pd.Series:
    H, L, C = hlc(df)
    prev_c = C.shift()
    tr = (H - L).abs()
    tr = np.maximum(tr, (H - prev_c).abs())
    tr = np.maximum(tr, (L - prev_c).abs())
    out = pd.Series(tr, index=df.index).rolling(win, min_periods=1).mean().bfill()
    return pd.to_numeric(out, errors="coerce").astype(float)

def realized_vol(close: pd.Series, win=RV_WIN) -> pd.Series:
    r = np.log(close).diff()
    out = r.rolling(win, min_periods=1).std().fillna(0.0)
    return pd.to_numeric(out, errors="coerce").astype(float)

def vwap_series(df: pd.DataFrame) -> pd.Series:
    # Robust daily VWAP approximation: cumulative typical price * volume / cumulative volume
    high = pd.to_numeric(df["High"], errors="coerce").astype(float)
    low = pd.to_numeric(df["Low"], errors="coerce").astype(float)
    close = pd.to_numeric(df["Close"], errors="coerce").astype(float)
    vol = pd.to_numeric(df["Volume"], errors="coerce").astype(float).replace(0, np.nan)

    typical = (high + low + close) / 3.0
    num = (typical * vol).cumsum()
    den = vol.cumsum()
    vwap = (num / den).fillna(typical)
    vwap.name = "vwap"
    return pd.to_numeric(vwap, errors="coerce").astype(float)

def close_location_value(df: pd.DataFrame) -> pd.Series:
    H, L, C = hlc(df)
    rng_ = (H - L).replace(0, np.nan)
    clv = ((C - L) - (H - C)) / rng_
    return pd.to_numeric(clv, errors="coerce").fillna(0.0).astype(float)

def cost_per_share(price: float, commission_bp=COMMISSION_BP, half_spread=HALF_SPREAD) -> float:
    comm = float(price) * float(commission_bp) * 1e-4
    return float(half_spread) + comm

# =========================
# Context features & regime heuristics
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Ensure numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    out["close"] = close_series(out)
    out["ret"] = np.log(out["close"]).diff().fillna(0.0)
    out["rv"] = realized_vol(out["close"], RV_WIN)
    out["atr5"] = atr(out, ATR_WIN)

    # VWAP and deviations (robust)
    out["vwap"] = vwap_series(out)
    if not isinstance(out["vwap"], pd.Series):
        out["vwap"] = pd.Series(out["vwap"], index=out.index, dtype=float)
    out["vwap"] = pd.to_numeric(out["vwap"], errors="coerce").astype(float)

    out["vwap_dev"] = pd.to_numeric(out["close"], errors="coerce").astype(float) - out["vwap"]
    dev_mean = out["vwap_dev"].rolling(60, min_periods=10).mean()
    dev_std = out["vwap_dev"].rolling(60, min_periods=10).std().replace(0, np.nan)
    out["vwap_dev_z"] = ((out["vwap_dev"] - dev_mean) / dev_std).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Volume percentile (rolling)
    out["vol_pct"] = out["Volume"].rolling(60, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.0, raw=False
    ).fillna(0.0)

    # Gap z vs prior ATR
    prev_close = out["close"].shift(1)
    out["gap"] = (out["Close"] - prev_close).fillna(0.0)
    den = out["atr5"].shift().replace(0, np.nan)
    out["gap_z"] = (out["gap"] / den).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Imbalance proxy
    out["clv"] = close_location_value(out)
    out["imb_proxy"] = (np.sign(out["Close"] - out["Open"]) * out["vol_pct"] + 0.5 * out["clv"]).fillna(0.0)

    # Time bucket (daily: day-of-week; intraday: minute-of-day)
    out["tod_bucket"] = out.index.dayofweek.astype(float)

    # Heuristic regimes (portable; replace with HMM when ready)
    rolling_med_rv = out["rv"].rolling(60, min_periods=20).median().fillna(method="bfill")
    regimes = []
    for i in range(len(out)):
        rv_i = out["rv"].iloc[i]
        med = rolling_med_rv.iloc[i]
        z = abs(out["vwap_dev_z"].iloc[i])
        gapz = abs(out["gap_z"].iloc[i])

        if gapz > 1.5:
            regimes.append("news")
        elif (rv_i > med) and (z > 0.8):
            regimes.append("trend")
        elif (rv_i < med) and (z < 0.3):
            regimes.append("chop")
        else:
            regimes.append("mean")
    out["regime"] = regimes

    # Final numeric coercion
    for col in ["close", "rv", "atr5", "vwap", "vwap_dev", "vwap_dev_z", "vol_pct", "gap_z", "imb_proxy", "tod_bucket"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).astype(float)

    return out

def context_vector(row: pd.Series) -> np.ndarray:
    return np.array([
        float(row["rv"]),
        float(row["vwap_dev_z"]),
        float(row["vol_pct"]),
        float(row["gap_z"]),
        float(row["imb_proxy"]),
        float(row["tod_bucket"]),
    ], dtype=float)

# =========================
# Strategies (playbooks)
# =========================
def orb_signal(df_feat: pd.DataFrame) -> Dict:
    n = OPEN_RANGE_MIN_BARS
    if len(df_feat) < n + 1:
        return {"entry": False, "dir": 0}
    recent = df_feat.iloc[-(n+1):]
    orh = float(recent["High"].iloc[:-1].max())
    long = (float(recent["Close"].iloc[-1]) > orh) and (float(recent["vol_pct"].iloc[-1]) > 0.7)
    return {"entry": bool(long), "dir": 1 if long else 0}

def vwap_revert_signal(df_feat: pd.DataFrame) -> Dict:
    if len(df_feat) < 30:
        return {"entry": False, "dir": 0}
    z = float(df_feat["vwap_dev_z"].iloc[-1])
    rv = float(df_feat["rv"].iloc[-1])
    med_rv = float(df_feat["rv"].rolling(60, min_periods=20).median().iloc[-1] or rv)
    allow = rv < med_rv * 1.2  # avoid high vol spikes
    if not allow:
        return {"entry": False, "dir": 0}
    if z > VWAP_Z_ENTRY:
        return {"entry": True, "dir": -1}
    if z < -VWAP_Z_ENTRY:
        return {"entry": True, "dir": 1}
    return {"entry": False, "dir": 0}

def pullback_trend_signal(df_feat: pd.DataFrame) -> Dict:
    if len(df_feat) < 65:
        return {"entry": False, "dir": 0}
    ma_fast = float(df_feat["Close"].rolling(20).mean().iloc[-1])
    ma_slow = float(df_feat["Close"].rolling(60).mean().iloc[-1])
    uptrend = ma_fast > ma_slow
    close = float(df_feat["Close"].iloc[-1])
    prev = float(df_feat["Close"].iloc[-2])
    pullback = (close < ma_fast) and uptrend
    trigger = pullback and (close > prev)
    return {"entry": bool(trigger), "dir": 1 if trigger else 0}

STRATEGY_FUNCS = {
    "orb": orb_signal,
    "vwap_revert": vwap_revert_signal,
    "pullback": pullback_trend_signal,
}
ARMS = list(STRATEGY_FUNCS.keys()) + ["no_trade"]

# =========================
# Contextual Thompson Sampling Bandit
# =========================
class ThompsonBandit:
    def __init__(self, n_features: int, arms: List[str], seed: int = SEED):
        self.arms = arms
        self.n = len(arms)
        self.mu = np.zeros((self.n, n_features))
        self.cov = np.array([np.eye(n_features) for _ in range(self.n)])
        self.rng = np.random.default_rng(seed)

    def select(self, x: np.ndarray, allowed_mask: List[bool]) -> int:
        scores = []
        for i in range(self.n):
            if not allowed_mask[i]:
                scores.append(-1e12)
                continue
            theta = self.rng.multivariate_normal(self.mu[i], self.cov[i])
            scores.append(float(theta @ x))
        return int(np.argmax(scores))

    def update(self, x: np.ndarray, arm_idx: int, reward: float):
        x = x.reshape(-1, 1).astype(float)
        cov = self.cov[arm_idx]
        mu = self.mu[arm_idx].reshape(-1, 1)
        cov_post = np.linalg.inv(np.linalg.inv(cov) + x @ x.T)
        mu_post = cov_post @ (np.linalg.inv(cov) @ mu + float(reward) * x)
        self.cov[arm_idx] = cov_post
        self.mu[arm_idx] = mu_post.ravel()

# =========================
# Execution & backtest
# =========================
@dataclass
class Position:
    entry_time: pd.Timestamp
    entry_px: float
    qty: int
    dir: int
    stop: float
    strategy: str
    regime: str

def select_allowed_arms(regime: str) -> List[bool]:
    allowed = [True] * len(ARMS)
    if regime == "chop":
        if "orb" in STRATEGY_FUNCS:
            allowed[ARMS.index("orb")] = False
    if regime == "news":
        if "vwap_revert" in STRATEGY_FUNCS:
            allowed[ARMS.index("vwap_revert")] = False
    return allowed

def simulate(df_feat: pd.DataFrame, bandit: ThompsonBandit) -> Tuple[List[dict], pd.Series]:
    equity = EQUITY0
    open_pos: Optional[Position] = None
    trades: List[dict] = []
    curve = []

    for i in range(len(df_feat) - 1):
        row = df_feat.iloc[i]
        nxt = df_feat.iloc[i + 1]
        ts = df_feat.index[i]
        nxt_ts = df_feat.index[i + 1]

        # Manage open position: stop or EOD
        if open_pos is not None:
            stopped = False
            if open_pos.dir == 1 and float(row["Low"]) <= open_pos.stop:
                exit_px = max(open_pos.stop, float(row["Low"]))
                stopped = True
            elif open_pos.dir == -1 and float(row["High"]) >= open_pos.stop:
                exit_px = min(open_pos.stop, float(row["High"]))
                stopped = True

            eod = (i + 2 == len(df_feat))
            if stopped or eod:
                per_share_cost = cost_per_share(exit_px if stopped else float(row["Close"]))
                if not stopped:
                    exit_px = float(row["Close"])
                pnl_gross = (exit_px - open_pos.entry_px) * open_pos.qty * open_pos.dir
                pnl_net = pnl_gross - per_share_cost * abs(open_pos.qty) * 2
                equity += pnl_net
                trades.append({
                    "entry_time": open_pos.entry_time,
                    "exit_time": ts,
                    "entry_px": open_pos.entry_px,
                    "exit_px": exit_px,
                    "qty": open_pos.qty,
                    "dir": open_pos.dir,
                    "strategy": open_pos.strategy,
                    "regime": open_pos.regime,
                    "pnl": pnl_net,
                })
                open_pos = None

        # Build strategy signals up to current bar
        df_sub = df_feat.iloc[:i + 1]
        signals = {name: fn(df_sub) for name, fn in STRATEGY_FUNCS.items()}

        # Choose arm
        x = context_vector(row)
        allowed = select_allowed_arms(str(row["regime"]))
        arm_idx = bandit.select(x, allowed)
        chosen = ARMS[arm_idx]

        # Enter only if flat and not no-trade
        if open_pos is None and chosen != "no_trade":
            sig = signals[chosen]
            if sig["entry"]:
                direction = int(sig["dir"])
                stop_dist = R_MULT_ATR * float(row["atr5"])
                if stop_dist > 0:
                    risk_dollars = equity * HEAT_BP * 1e-4
                    qty = int(max(0, risk_dollars / stop_dist))
                    if qty > 0:
                        entry_px = float(nxt["Open"])
                        per_share_cost = cost_per_share(entry_px)
                        stop = entry_px - stop_dist if direction == 1 else entry_px + stop_dist
                        open_pos = Position(
                            entry_time=nxt_ts,
                            entry_px=entry_px,
                            qty=qty,
                            dir=direction,
                            stop=stop,
                            strategy=chosen,
                            regime=str(row["regime"]),
                        )
                        equity -= per_share_cost * qty  # entry cost

                        # Policy Card (stub; hook for LLM later)
                        _policy_card = f"[{nxt_ts}] {chosen.upper()} | regime={row['regime']} | " \
                                       f"rv={row['rv']:.4f}, z={row['vwap_dev_z']:.2f}, vol%={row['vol_pct']:.2f} | " \
                                       f"stop_dist={stop_dist:.3f}, qty={qty}"

        # Reward for bandit (shaped)
        reward = 0.0
        if open_pos is None:
            reward = -0.00001  # tiny penalty for doing nothing
        else:
            reward = (float(row["Close"]) - open_pos.entry_px) * open_pos.qty * open_pos.dir

        bandit.update(x, arm_idx, reward=float(reward))
        curve.append(equity)

    return trades, pd.Series(curve, index=df_feat.index[:len(curve)])

# =========================
# Main
# =========================
def main():
    np.random.seed(SEED)

    df = yf.download(TICKER, start=START, end=END, interval=BAR, auto_adjust=True, progress=False)
    if df.empty:
        print("No data fetched. Check ticker/dates.")
        return

    # Ensure a clean DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df_feat = build_features(df)

    bandit = ThompsonBandit(n_features=6, arms=ARMS, seed=SEED)
    trades, equity = simulate(df_feat, bandit)

    total_pnl = sum(t["pnl"] for t in trades) if trades else 0.0
    print(f"Trades: {len(trades)} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.2f}%")

