from __future__ import annotations

import base64
import csv
import io
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SES import AmazonSES


@dataclass
class IterationResult:
    date: pd.Timestamp
    action: str  # "BUY", "SELL", "SHORT", "COVER", "HOLD", "NOOP"
    state: float  # -1, 0, +1
    pos_exec: float  # yesterday's state used for today's return
    lev_exec: float  # yesterday's leverage used for today's return
    exposure: float  # signed exposure used for today's return
    r_oo: float  # open-to-open return
    turnover: float
    costs: float
    ret: float  # strategy return for the day (net)
    debug: Dict[str, Any]  # signal/indicator snapshot


# -----------------------------
# Helpers
# -----------------------------


def atr(high, low, close, n=20):
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def donchian_channels(high, low, n):
    upper = high.rolling(n).max()
    lower = low.rolling(n).min()
    return upper, lower


# ============================================================
# Main algo (single daily iteration)
# ============================================================


def run_single_iteration(
    df: pd.DataFrame,
    *,
    prev_state: float = 0.0,  # -1,0,+1 carried from yesterday
    prev_exposure: float = 0.0,  # exposure carried from yesterday (pos_exec*lev_exec)
    breakout_n: int = 55,
    exit_n: int = 20,
    atr_n: int = 20,
    ema_span: int = 200,
    use_ema_filter: bool = True,
    target_annual_vol: float = 0.20,
    max_leverage: float = 2.0,
    cost_bps: float = 2.0,
    slippage_bps: float = 1.0,
    periods_per_year: int = 252,
    # Optional execution hook: you asked to "leave an empty function that buys and sells"
    execute_order: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> IterationResult:
    """
    Single-step version of backtest_futures_trend_long_short.

    Assumptions (same as your backtest):
      - Signals computed on the *close* of day t-1 (via shift(1) rules).
      - Trades are executed at the *open* of day t.
      - Strategy PnL is based on open-to-open returns.

    Inputs:
      df: must contain at least enough history to compute indicators and include the latest row.
          Required columns: ["Open","High","Low","Close"].

      prev_state: yesterday's "pos" state (-1,0,+1) *after* yesterday's signal processing.
      prev_exposure: yesterday's exposure (pos_exec*lev_exec) used for turnover costs.

    Returns:
      IterationResult for the latest date in df.
    """
    if df is None or len(df) < 3:
        raise ValueError("df must have at least 3 rows (need t-1 and t)")

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    px = df.copy()

    # --- Indicators (computed on full history up to latest row) ---
    px["ATR"] = atr(px["High"], px["Low"], px["Close"], n=atr_n)
    px["ATR_PCT"] = px["ATR"] / px["Close"]
    px["EMA"] = ema(px["Close"], span=ema_span)

    up, dn = donchian_channels(px["High"], px["Low"], n=breakout_n)
    px["DONCH_UP"] = up
    px["DONCH_DN"] = dn

    exit_up, exit_dn = donchian_channels(px["High"], px["Low"], n=exit_n)
    px["EXIT_UP"] = exit_up
    px["EXIT_DN"] = exit_dn

    # --- Focus on today's index (t) and yesterday (t-1) ---
    idx = px.index
    t = idx[-1]
    t1 = idx[-2]  # yesterday

    # Signals at close; trade next open
    long_entry_t = bool(px.loc[t, "Close"] > px.loc[t, "DONCH_UP"].shift(1).loc[t])
    short_entry_t = bool(px.loc[t, "Close"] < px.loc[t, "DONCH_DN"].shift(1).loc[t])

    long_exit_t = bool(px.loc[t, "Close"] < px.loc[t, "EXIT_DN"].shift(1).loc[t])
    short_exit_t = bool(px.loc[t, "Close"] > px.loc[t, "EXIT_UP"].shift(1).loc[t])

    if use_ema_filter:
        long_ok_t = bool(px.loc[t, "Close"] > px.loc[t, "EMA"])
        short_ok_t = bool(px.loc[t, "Close"] < px.loc[t, "EMA"])
    else:
        long_ok_t = True
        short_ok_t = True

    # --- Position state machine (single update for day t) ---
    state = float(prev_state)
    action = "HOLD"

    if state == 0.0:
        le = bool(long_entry_t and long_ok_t)
        se = bool(short_entry_t and short_ok_t)
        if le and not se:
            state = 1.0
            action = "BUY"
        elif se and not le:
            state = -1.0
            action = "SHORT"
        else:
            action = "NOOP"

    elif state == 1.0:
        # exit or filter fail
        if long_exit_t or (use_ema_filter and not long_ok_t):
            state = 0.0
            action = "SELL"
        else:
            # allow reversal
            if short_entry_t and short_ok_t:
                state = -1.0
                action = "SELL/SHORT"

    elif state == -1.0:
        if short_exit_t or (use_ema_filter and not short_ok_t):
            state = 0.0
            action = "COVER"
        else:
            if long_entry_t and long_ok_t:
                state = 1.0
                action = "COVER/BUY"

    # --- Vol targeting (leverage for day t, but EXEC at next open => use t-1 for execution today) ---
    target_daily_vol = float(target_annual_vol) / float(np.sqrt(periods_per_year))

    # Daily vol estimate series
    est_daily_vol = px["ATR_PCT"].replace(0, np.nan)

    lev_series = (target_daily_vol / est_daily_vol).clip(upper=max_leverage).fillna(0.0)

    # Execute at today's open: use yesterday's (t-1) state and leverage
    pos_exec = float(prev_state)  # yesterday's final state -> today's position
    lev_exec = float(lev_series.loc[t1])  # yesterday's vol estimate -> today's sizing
    exposure = pos_exec * lev_exec  # signed exposure used for today's PnL

    # Open-to-open return for today
    r_oo = float(px["Open"].pct_change().loc[t])
    if np.isnan(r_oo):
        r_oo = 0.0

    # Costs on exposure change (turnover)
    turnover = float(abs(exposure - float(prev_exposure)))
    total_bps = float(cost_bps + slippage_bps)
    costs = turnover * (total_bps / 1e4)

    ret = exposure * r_oo - costs

    # Optional execution hook (EMPTY by default)
    if execute_order is not None:
        # You can implement broker/paper order logic outside and pass it in.
        # We'll call it with action + a payload snapshot.
        payload = {
            "date": t,
            "action": action,
            "target_state": state,
            "pos_exec": pos_exec,
            "lev_exec": lev_exec,
            "exposure": exposure,
            "open": float(px.loc[t, "Open"]),
            "close": float(px.loc[t, "Close"]),
        }
        execute_order(action, payload)

    debug = {
        "Close": float(px.loc[t, "Close"]),
        "Open": float(px.loc[t, "Open"]),
        "EMA": float(px.loc[t, "EMA"]) if pd.notna(px.loc[t, "EMA"]) else np.nan,
        "DONCH_UP": (
            float(px.loc[t, "DONCH_UP"]) if pd.notna(px.loc[t, "DONCH_UP"]) else np.nan
        ),
        "DONCH_DN": (
            float(px.loc[t, "DONCH_DN"]) if pd.notna(px.loc[t, "DONCH_DN"]) else np.nan
        ),
        "EXIT_UP": (
            float(px.loc[t, "EXIT_UP"]) if pd.notna(px.loc[t, "EXIT_UP"]) else np.nan
        ),
        "EXIT_DN": (
            float(px.loc[t, "EXIT_DN"]) if pd.notna(px.loc[t, "EXIT_DN"]) else np.nan
        ),
        "ATR_PCT_t1": (
            float(px.loc[t1, "ATR_PCT"]) if pd.notna(px.loc[t1, "ATR_PCT"]) else np.nan
        ),
        "lev_t1": float(lev_series.loc[t1]) if pd.notna(lev_series.loc[t1]) else 0.0,
        "signals": {
            "long_entry": long_entry_t,
            "short_entry": short_entry_t,
            "long_exit": long_exit_t,
            "short_exit": short_exit_t,
            "long_ok": long_ok_t,
            "short_ok": short_ok_t,
        },
    }

    return IterationResult(
        date=pd.Timestamp(t),
        action=action,
        state=state,  # this is your *new* state to carry forward
        pos_exec=pos_exec,
        lev_exec=lev_exec,
        exposure=exposure,
        r_oo=r_oo,
        turnover=turnover,
        costs=costs,
        ret=ret,
        debug=debug,
    )


# ============================================================
# Portfolio state + CSV persistence
# ============================================================


@dataclass
class PortfolioState:
    cash: float
    qty: int
    equity: float
    pos_state: float  # -1,0,+1 (your state machine)
    exposure: float  # last exposure used for turnover/costs
    last_date: Optional[str] = None


PORTFOLIO_COLUMNS = [
    "date",
    "symbol",
    "equity",
    "cash",
    "qty",
    "pos_state",
    "exposure",
    "open",
    "close",
    "ret",
    "costs",
    "turnover",
    "action",
    # trade fields
    "trade_side",
    "trade_qty",
    "trade_fill_price",
    "trade_notional",
]


# ============================================================
# Paper Trading Helpers
# ============================================================


def _read_last_portfolio_row(path: str) -> Optional[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows[-1] if rows else None


def parse_float_field(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    """
    Safe float parse for CSV row fields.
    - Strips whitespace
    - Returns default on empty/missing/parse error
    """
    v = (row.get(key) or "").strip()
    if v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def parse_int_field(row: Dict[str, str], key: str, default: int = 0) -> int:
    """
    Safe int parse for CSV row fields.
    - Accepts numeric strings like "10" or "10.0"
    - Returns default on empty/missing/parse error
    """
    v = (row.get(key) or "").strip()
    if v == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def portfolio_state_from_last_row(last: Dict[str, str]) -> "PortfolioState":
    """
    Build PortfolioState from the last CSV row.
    Uses cash if present; otherwise falls back to equity for cash (legacy compatibility).
    """
    cash = parse_float_field(
        last, "cash", default=parse_float_field(last, "equity", 0.0)
    )

    return PortfolioState(
        cash=cash,
        qty=parse_int_field(last, "qty", 0),
        equity=parse_float_field(last, "equity", 0.0),
        pos_state=parse_float_field(last, "pos_state", 0.0),
        exposure=parse_float_field(last, "exposure", 0.0),
        last_date=(last.get("date") or None),
    )


def initial_portfolio_state(initial_equity: float) -> "PortfolioState":
    """
    Default initial PortfolioState when no CSV exists.
    """
    init_equity = float(initial_equity)
    return PortfolioState(
        cash=init_equity,
        qty=0,
        equity=init_equity,
        pos_state=0.0,
        exposure=0.0,
        last_date=None,
    )


def append_portfolio_row(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PORTFOLIO_COLUMNS)
        if not exists:
            w.writeheader()
        out = {k: row.get(k, "") for k in PORTFOLIO_COLUMNS}
        w.writerow(out)


def load_portfolio_state(
    portfolio_csv_path: str,
    *,
    initial_equity_env: str = "INITIAL_PORTFOLIO_VALUE",
) -> "PortfolioState":
    """
    Load portfolio state from CSV if present; otherwise initialize from env var.

    Depends on:
      - _read_last_portfolio_row(path) -> Optional[Dict[str,str]]
      - PortfolioState dataclass
    """
    last = _read_last_portfolio_row(portfolio_csv_path)
    if last is None:
        initial_equity = float(os.environ.get(initial_equity_env, "100000"))
        return initial_portfolio_state(initial_equity)

    return portfolio_state_from_last_row(last)


# ============================================================
# Trade execution (paper)
# ============================================================
def place_order(
    *,
    symbol: str,
    cash: float,
    qty: int,
    target_qty: int,
    fill_price: float,
) -> Tuple[float, int, Dict[str, Any]]:
    """
    Paper rebalance to target_qty at fill_price.

    Returns:
      (new_cash, new_qty, trade_record)

    trade_record is suitable to write into portfolio CSV row.
    """
    diff = int(target_qty - qty)
    if diff == 0:
        return (
            float(cash),
            int(qty),
            {
                "trade_side": "",
                "trade_qty": 0,
                "trade_fill_price": "",
                "trade_notional": 0.0,
            },
        )

    side = "BUY" if diff > 0 else "SELL"
    trade_qty = abs(diff)
    notional = float(trade_qty * float(fill_price))

    # BUY reduces cash, SELL increases cash
    signed_notional = float(diff * float(fill_price))
    new_cash = float(cash - signed_notional)
    new_qty = int(target_qty)

    trade_record = {
        "trade_side": side,
        "trade_qty": trade_qty,
        "trade_fill_price": float(fill_price),
        "trade_notional": float(notional),
    }
    return new_cash, new_qty, trade_record


def mark_to_market_equity(cash: float, qty: int, price: float) -> float:
    return float(cash + qty * float(price))


# ============================================================
# Simple sizing helper (shares-style)
# ============================================================


def target_qty_from_state(
    *,
    state: float,
    equity: float,
    price: float,
    max_gross_leverage: float = 1.0,
) -> int:
    """
    Map state (-1/0/+1) to a target quantity.

    NOTE: This is “shares style”. For futures, swap this for contract sizing.
    """
    if state == 0.0 or equity <= 0 or price <= 0:
        return 0
    gross = float(equity) * float(max_gross_leverage)
    q = int(np.floor(gross / float(price)))
    return q if state > 0 else -q


# ============================================================
# Paper Trading iteration of algo (single daily iteration)
# ============================================================


def run_daily_algo_once(
    *,
    symbol: str,
    prices_csv_path: str,
    portfolio_csv_path: str,
    lookback_days: int = 400,
    interval: str = "d",
    initial_equity_env: str = "INITIAL_PORTFOLIO_VALUE",
    max_gross_leverage: float = 1.0,
    strategy_kwargs: Optional[Dict[str, Any]] = None,
    stooq: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    1) Update price history (StooqDownloader.update_prices)
    2) Load last X days into df
    3) Load portfolio state from CSV (or env initial)
    4) Run run_single_iteration(df, prev_state, prev_exposure) -> tells us NEW state for tomorrow
    5) Convert NEW state to target_qty and execute via place_order() at today's OPEN
    6) Append portfolio row (includes trade + portfolio state)
    """
    if strategy_kwargs is None:
        strategy_kwargs = {}
    if stooq is None:
        raise ValueError("Pass stooq=StooqDownloader() instance")

    # (1) Update prices CSV to latest
    stooq.update_prices(symbol, prices_csv_path, interval=interval)

    # (2) Load last X days to df
    raw = pd.read_csv(prices_csv_path)

    if "Date" not in raw.columns:
        raise ValueError(f"{prices_csv_path} missing 'Date' column")

    # Parse both "23 Apr 1993" and ISO; do two-pass parse
    dt1 = pd.to_datetime(raw["Date"], errors="coerce", format="%d %b %Y")
    dt2 = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Date"] = dt1.fillna(dt2)

    raw = raw.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    for col in ["Open", "High", "Low", "Close"]:
        if col not in raw.columns:
            raise ValueError(f"Price CSV missing required column '{col}'")

    df = raw[["Open", "High", "Low", "Close"]].tail(int(lookback_days)).copy()
    if len(df) < 3:
        raise ValueError(
            "Not enough history to run a single iteration (need >= 3 rows)"
        )

    # (3) Load portfolio state
    p = load_portfolio_state(portfolio_csv_path, initial_equity_env=initial_equity_env)

    today = df.index[-1]
    today_str = today.strftime("%Y-%m-%d")

    if p.last_date == today_str:
        return {
            "status": "skipped",
            "reason": "already_processed",
            "date": today_str,
            "symbol": symbol,
        }

    open_px = float(df["Open"].iloc[-1])
    close_px = float(df["Close"].iloc[-1])

    # (4) Run strategy iteration (determines NEW state)
    res = run_single_iteration(
        df,
        prev_state=p.pos_state,
        prev_exposure=p.exposure,
        **strategy_kwargs,
    )

    # (5) Decide target qty from NEW state; execute at OPEN via place_order()
    target_qty = target_qty_from_state(
        state=res.state,
        equity=p.equity,
        price=open_px,
        max_gross_leverage=max_gross_leverage,
    )

    new_cash, new_qty, trade_rec = place_order(
        symbol=symbol,
        cash=p.cash,
        qty=p.qty,
        target_qty=target_qty,
        fill_price=open_px,
    )

    new_equity = mark_to_market_equity(new_cash, new_qty, close_px)

    # Carry forward:
    # - pos_state becomes NEW state from run_single_iteration
    # - exposure becomes TODAY's exposure used for today PnL/costs (res.exposure)
    p_next = PortfolioState(
        cash=new_cash,
        qty=new_qty,
        equity=new_equity,
        pos_state=res.state,
        exposure=res.exposure,
        last_date=today_str,
    )

    # (6) Append portfolio row including trade + state
    append_portfolio_row(
        portfolio_csv_path,
        {
            "date": today_str,
            "symbol": symbol,
            "equity": round(p_next.equity, 6),
            "cash": round(p_next.cash, 6),
            "qty": p_next.qty,
            "pos_state": round(p_next.pos_state, 6),
            "exposure": round(p_next.exposure, 6),
            "open": round(open_px, 6),
            "close": round(close_px, 6),
            "ret": round(float(res.ret), 12),
            "costs": round(float(res.costs), 12),
            "turnover": round(float(res.turnover), 12),
            "action": res.action,
            **trade_rec,
        },
    )

    return {
        "status": "ok",
        "date": today_str,
        "symbol": symbol,
        "action": res.action,
        "prev_equity": p.equity,
        "new_equity": p_next.equity,
        "prev_qty": p.qty,
        "new_qty": p_next.qty,
        "trade": trade_rec,
        "debug": res.debug,
    }


# ---------------------------------------------------------------------
# Paper Trade Reporter
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
@dataclass
class PerfMetrics:
    cagr: float
    sharpe: float
    sortino: float
    maxdd: float
    start_equity: float
    end_equity: float
    start_date: str
    end_date: str
    n_days: int


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())  # negative


def _cagr(equity: np.ndarray, n_days: int, periods_per_year: int = 252) -> float:
    if n_days <= 1 or equity.size == 0:
        return 0.0
    start = float(equity[0])
    end = float(equity[-1])
    if start <= 0 or end <= 0:
        return 0.0
    years = n_days / float(periods_per_year)
    if years <= 0:
        return 0.0
    return float((end / start) ** (1.0 / years) - 1.0)


def _sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if returns.size < 2:
        return 0.0
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd <= 0:
        return 0.0
    return float((mu / sd) * math.sqrt(periods_per_year))


def _sortino(returns: np.ndarray, periods_per_year: int = 252) -> float:
    if returns.size < 2:
        return 0.0
    mu = float(np.mean(returns))
    downside = returns[returns < 0]
    if downside.size < 2:
        return 0.0
    ds = float(np.std(downside, ddof=1))
    if ds <= 0:
        return 0.0
    return float((mu / ds) * math.sqrt(periods_per_year))


def compute_metrics_from_equity(
    df: pd.DataFrame,
    *,
    equity_col: str = "equity",
    date_col: str = "date",
    periods_per_year: int = 252,
) -> PerfMetrics:
    if equity_col not in df.columns:
        raise ValueError(f"df missing equity_col='{equity_col}'")
    if date_col not in df.columns:
        raise ValueError(f"df missing date_col='{date_col}'")

    dd = df.copy()
    dd[date_col] = pd.to_datetime(dd[date_col], errors="coerce")
    dd = dd.dropna(subset=[date_col, equity_col]).sort_values(date_col)

    if len(dd) < 2:
        raise ValueError("Need at least 2 rows to compute metrics")

    equity = dd[equity_col].astype(float).to_numpy()
    rets = pd.Series(equity).pct_change().fillna(0.0).to_numpy()

    n_days = int(len(dd))
    start_date = dd[date_col].iloc[0].strftime("%Y-%m-%d")
    end_date = dd[date_col].iloc[-1].strftime("%Y-%m-%d")

    return PerfMetrics(
        cagr=_cagr(equity, n_days=n_days, periods_per_year=periods_per_year),
        sharpe=_sharpe(rets, periods_per_year=periods_per_year),
        sortino=_sortino(rets, periods_per_year=periods_per_year),
        maxdd=_max_drawdown(equity),
        start_equity=float(equity[0]),
        end_equity=float(equity[-1]),
        start_date=start_date,
        end_date=end_date,
        n_days=n_days,
    )


# ---------------------------------------------------------------------
# Plot (only when enough days)
# ---------------------------------------------------------------------
def equity_curve_png_base64(
    df: pd.DataFrame,
    *,
    equity_col: str = "equity",
    date_col: str = "date",
    title: str = "Equity Curve",
) -> str:
    dd = df.copy()
    dd[date_col] = pd.to_datetime(dd[date_col], errors="coerce")
    dd = dd.dropna(subset=[date_col, equity_col]).sort_values(date_col)

    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(dd[date_col], dd[equity_col].astype(float))
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------
# Email report with threshold behavior
# ---------------------------------------------------------------------
def send_equity_report_email(
    *,
    region: str,
    access_key: str,
    secret_key: str,
    from_address: str,
    to_addresses: Sequence[str],
    portfolio_csv_path: str,
    lookback_days: int = 60,
    plot_day_threshold: int = 20,  # <-- NEW: only include chart if >= this many days
    equity_col: str = "equity",
    date_col: str = "date",
    symbol_col: Optional[str] = "symbol",
    symbol_filter: Optional[str] = None,
    periods_per_year: int = 252,
    subject_prefix: str = "Strategy Report",
) -> None:
    """
    If number of rows in the last `lookback_days` is:
      - >= plot_day_threshold: send chart + metrics
      - <  plot_day_threshold: send only portfolio value + metrics (no chart)

    Metrics included always: CAGR, Sharpe, MaxDD, Sortino, end equity.
    """
    if not to_addresses:
        raise ValueError("to_addresses must be non-empty")

    df = pd.read_csv(portfolio_csv_path)

    if symbol_filter and symbol_col and symbol_col in df.columns:
        df = df[df[symbol_col].astype(str).str.lower() == str(symbol_filter).lower()]

    if len(df) < 2:
        raise ValueError("Portfolio CSV does not have enough rows to report")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = (
        df.dropna(subset=[date_col, equity_col])
        .sort_values(date_col)
        .tail(int(lookback_days))
    )

    if len(df) < 2:
        raise ValueError("Not enough rows after lookback_days for reporting")

    metrics = compute_metrics_from_equity(
        df, equity_col=equity_col, date_col=date_col, periods_per_year=periods_per_year
    )

    def fmt_pct(x: float) -> str:
        return f"{x*100:,.2f}%"

    # Subject
    subject = f"{subject_prefix} — {metrics.end_date} (last {metrics.n_days} days)"

    # Body (chart only if enough days)
    if metrics.n_days < int(plot_day_threshold):
        html_content = f"""
        <html>
          <body style="font-family: Arial, sans-serif; line-height: 1.35;">
            <h2>{subject_prefix}: Snapshot</h2>

            <p><b>Date range:</b> {metrics.start_date} → {metrics.end_date} &nbsp; | &nbsp;
               <b>Days:</b> {metrics.n_days}</p>

            <p>
              <b>Current equity:</b> {metrics.end_equity:,.2f}
            </p>

            <h3>Performance</h3>
            <ul>
              <li><b>CAGR:</b> {fmt_pct(metrics.cagr)}</li>
              <li><b>Max Drawdown:</b> {fmt_pct(metrics.maxdd)}</li>
              <li><b>Sharpe:</b> {metrics.sharpe:,.2f}</li>
              <li><b>Sortino:</b> {metrics.sortino:,.2f}</li>
            </ul>

            <p style="color:#666; font-size: 12px;">
              Chart omitted because history &lt; {plot_day_threshold} days.
            </p>
          </body>
        </html>
        """
    else:
        title = f"Equity Curve (last {metrics.n_days} days)"
        png_b64 = equity_curve_png_base64(
            df, equity_col=equity_col, date_col=date_col, title=title
        )

        html_content = f"""
        <html>
          <body style="font-family: Arial, sans-serif; line-height: 1.35;">
            <h2>{subject_prefix}: Last {metrics.n_days} Trading Days</h2>

            <p><b>Date range:</b> {metrics.start_date} → {metrics.end_date} &nbsp; | &nbsp;
               <b>Days:</b> {metrics.n_days}</p>

            <p>
              <b>Start equity:</b> {metrics.start_equity:,.2f}<br/>
              <b>Current equity:</b> {metrics.end_equity:,.2f}
            </p>

            <h3>Performance</h3>
            <ul>
              <li><b>CAGR:</b> {fmt_pct(metrics.cagr)}</li>
              <li><b>Max Drawdown:</b> {fmt_pct(metrics.maxdd)}</li>
              <li><b>Sharpe:</b> {metrics.sharpe:,.2f}</li>
              <li><b>Sortino:</b> {metrics.sortino:,.2f}</li>
            </ul>

            <h3>Equity Curve</h3>
            <img alt="equity_curve" style="max-width: 100%; border: 1px solid #ddd;"
                 src="data:image/png;base64,{png_b64}" />

            <p style="color:#666; font-size: 12px;">
              Notes: Metrics computed from daily equity pct-change. MaxDD is the minimum drawdown (negative %).
            </p>
          </body>
        </html>
        """

    ses = AmazonSES(
        region=region,
        access_key=access_key,
        secret_key=secret_key,
        from_address=from_address,
    )
    ses.send_html_email_many(to_addresses, subject, html_content)
