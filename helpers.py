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


def str2bool(value):
    valid = {
        "true": True,
        "t": True,
        "1": True,
        "on": True,
        "false": False,
        "f": False,
        "0": False,
    }

    if isinstance(value, bool):
        return value

    lower_value = value.lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError('invalid literal for boolean: "%s"' % value)


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
    prev_state: float = 0.0,  # -1,0,+1 carried from yesterday (already known at yesterday close)
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
    entry_signal_mode: str = "close",  # "close" or "intraday"
    force_reentry_after_noop_days: int = 0,
    noop_streak: int = 0,
    # Optional execution hook (you can ignore or pass your empty place_order wrapper)
    execute_order: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> IterationResult:
    """
    Single-step version of backtest_futures_trend_long_short.

    Timing (matches your backtest):
      - Signals computed using today's CLOSE compared to yesterday's channel levels (shift(1)).
      - Trades execute NEXT OPEN.
      - Today's PnL uses yesterday's executed position (prev_state) and yesterday's leverage estimate.

    Inputs:
      df: must contain at least 3 rows and columns: Open, High, Low, Close
      prev_state: yesterday's state after its signal processing
      prev_exposure: yesterday's exposure used to compute turnover costs today

    Returns:
      IterationResult for the last date in df.
    """
    if df is None or len(df) < 3:
        raise ValueError("df must have at least 3 rows (need t-1 and t)")

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    px = df.copy()

    # --- Indicators over full history ---
    px["ATR"] = atr(px["High"], px["Low"], px["Close"], n=atr_n)
    px["ATR_PCT"] = px["ATR"] / px["Close"]
    px["EMA"] = ema(px["Close"], span=ema_span)

    up, dn = donchian_channels(px["High"], px["Low"], n=breakout_n)
    px["DONCH_UP"] = up
    px["DONCH_DN"] = dn

    exit_up, exit_dn = donchian_channels(px["High"], px["Low"], n=exit_n)
    px["EXIT_UP"] = exit_up
    px["EXIT_DN"] = exit_dn

    # --- Identify t and t-1 ---
    idx = px.index
    t = idx[-1]
    t1 = idx[-2]

    # --- Pre-shifted series (FIX: shift the SERIES, not a scalar) ---
    donch_up_1 = px["DONCH_UP"].shift(1)
    donch_dn_1 = px["DONCH_DN"].shift(1)
    exit_up_1 = px["EXIT_UP"].shift(1)
    exit_dn_1 = px["EXIT_DN"].shift(1)

    # --- Helper comparisons with NaN safety ---
    def _gt(a, b) -> bool:
        return pd.notna(a) and pd.notna(b) and float(a) > float(b)

    def _lt(a, b) -> bool:
        return pd.notna(a) and pd.notna(b) and float(a) < float(b)

    close_t = px.loc[t, "Close"]
    high_t = px.loc[t, "High"]
    low_t = px.loc[t, "Low"]

    if entry_signal_mode not in ("close", "intraday"):
        raise ValueError("entry_signal_mode must be one of: 'close', 'intraday'")

    # Entry breakout mode:
    # - close: use close vs shifted channel (original behavior)
    # - intraday: use high/low breaches vs shifted channel
    if entry_signal_mode == "intraday":
        long_entry_t = _gt(high_t, donch_up_1.loc[t])
        short_entry_t = _lt(low_t, donch_dn_1.loc[t])
    else:
        long_entry_t = _gt(close_t, donch_up_1.loc[t])
        short_entry_t = _lt(close_t, donch_dn_1.loc[t])

    long_exit_t = _lt(close_t, exit_dn_1.loc[t])
    short_exit_t = _gt(close_t, exit_up_1.loc[t])

    # EMA filter (evaluated on close_t)
    if use_ema_filter:
        ema_t = px.loc[t, "EMA"]
        long_ok_t = pd.notna(ema_t) and float(close_t) > float(ema_t)
        short_ok_t = pd.notna(ema_t) and float(close_t) < float(ema_t)
    else:
        ema_t = px.loc[t, "EMA"]
        long_ok_t = True
        short_ok_t = True

    # --- State machine update (produces NEW state to carry forward) ---
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
            if (
                int(force_reentry_after_noop_days) > 0
                and int(noop_streak) >= int(force_reentry_after_noop_days)
                and bool(long_ok_t)
            ):
                state = 1.0
                action = "BUY_REENTRY"
            else:
                action = "NOOP"

    elif state == 1.0:
        # Long exits; optional reversal
        if bool(long_exit_t) or (use_ema_filter and not bool(long_ok_t)):
            state = 0.0
            action = "SELL"
        else:
            if bool(short_entry_t and short_ok_t):
                state = -1.0
                action = "SELL/SHORT"

    elif state == -1.0:
        # Short exits; optional reversal
        if bool(short_exit_t) or (use_ema_filter and not bool(short_ok_t)):
            state = 0.0
            action = "COVER"
        else:
            if bool(long_entry_t and long_ok_t):
                state = 1.0
                action = "COVER/BUY"

    # --- Vol targeting (daily) ---
    target_daily_vol = float(target_annual_vol) / float(np.sqrt(periods_per_year))
    est_daily_vol = px["ATR_PCT"].replace(0, np.nan)

    lev_series = (target_daily_vol / est_daily_vol).clip(upper=max_leverage).fillna(0.0)

    # Execute at today's open => use yesterday's state and yesterday's leverage
    pos_exec = float(prev_state)
    lev_exec = float(lev_series.loc[t1]) if pd.notna(lev_series.loc[t1]) else 0.0
    exposure = pos_exec * lev_exec

    # Open-to-open return for today
    r_oo = float(px["Open"].pct_change().loc[t])
    if np.isnan(r_oo):
        r_oo = 0.0

    # Turnover + costs on exposure changes
    turnover = float(abs(exposure - float(prev_exposure)))
    total_bps = float(cost_bps + slippage_bps)
    costs = turnover * (total_bps / 1e4)

    ret = exposure * r_oo - costs

    # Optional execution hook
    if execute_order is not None:
        payload = {
            "date": pd.Timestamp(t),
            "action": action,
            "new_state": state,
            "prev_state": float(prev_state),
            "pos_exec": pos_exec,
            "lev_exec": lev_exec,
            "exposure": exposure,
            "open": float(px.loc[t, "Open"]),
            "close": float(close_t),
        }
        execute_order(action, payload)

    debug = {
        "Close": float(close_t),
        "Open": float(px.loc[t, "Open"]),
        "EMA": float(ema_t) if pd.notna(ema_t) else np.nan,
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
        "DONCH_UP_shift1": (
            float(donch_up_1.loc[t]) if pd.notna(donch_up_1.loc[t]) else np.nan
        ),
        "DONCH_DN_shift1": (
            float(donch_dn_1.loc[t]) if pd.notna(donch_dn_1.loc[t]) else np.nan
        ),
        "EXIT_UP_shift1": (
            float(exit_up_1.loc[t]) if pd.notna(exit_up_1.loc[t]) else np.nan
        ),
        "EXIT_DN_shift1": (
            float(exit_dn_1.loc[t]) if pd.notna(exit_dn_1.loc[t]) else np.nan
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
            "long_ok": bool(long_ok_t),
            "short_ok": bool(short_ok_t),
        },
    }

    return IterationResult(
        date=pd.Timestamp(t),
        action=action,
        state=float(state),
        pos_exec=pos_exec,
        lev_exec=lev_exec,
        exposure=float(exposure),
        r_oo=float(r_oo),
        turnover=float(turnover),
        costs=float(costs),
        ret=float(ret),
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


def _is_valid_portfolio_row(row: Dict[str, str]) -> bool:
    date_val = (row.get("date") or "").strip()
    if not date_val:
        return False

    try:
        pd.to_datetime(date_val, errors="raise")
    except Exception:
        return False

    try:
        equity = float((row.get("equity") or "").strip())
        cash = float((row.get("cash") or "").strip())
        qty = int(float((row.get("qty") or "").strip()))
        pos_state = float((row.get("pos_state") or "").strip())
        exposure = float((row.get("exposure") or "").strip())
    except Exception:
        return False

    if not np.isfinite(equity) or equity <= 0:
        return False
    if not np.isfinite(cash):
        return False
    if not np.isfinite(float(qty)):
        return False
    if not np.isfinite(pos_state):
        return False
    if not np.isfinite(exposure):
        return False
    if pos_state not in (-1.0, 0.0, 1.0):
        return False

    return True


def sanitize_portfolio_csv(path: str) -> Optional[Dict[str, str]]:
    """
    Remove invalid trailing rows from portfolio CSV and return last valid row.
    If no valid rows exist, leave file with header only and return None.
    """
    if not path or not os.path.exists(path):
        return None

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or PORTFOLIO_COLUMNS)
        rows = list(reader)

    if not rows:
        return None

    last_valid_idx = -1
    for i, row in enumerate(rows):
        if _is_valid_portfolio_row(row):
            last_valid_idx = i

    if last_valid_idx < 0:
        kept_rows = []
        last_valid = None
    else:
        kept_rows = rows[: last_valid_idx + 1]
        last_valid = kept_rows[-1]

    # Prune stale frozen tail:
    # repeated flat NOOP rows with identical equity/cash/qty at the file end.
    # Keep the first row of that tail as the last good anchor.
    def _is_flat_noop(row: Dict[str, str]) -> bool:
        action = (row.get("action") or "").strip().upper()
        trade_side = (row.get("trade_side") or "").strip()
        return (
            action == "NOOP"
            and parse_int_field(row, "qty", 0) == 0
            and parse_float_field(row, "pos_state", 0.0) == 0.0
            and parse_float_field(row, "exposure", 0.0) == 0.0
            and trade_side == ""
            and parse_int_field(row, "trade_qty", 0) == 0
            and parse_float_field(row, "trade_notional", 0.0) == 0.0
        )

    def _same_portfolio_triplet(a: Dict[str, str], b: Dict[str, str]) -> bool:
        return (
            parse_float_field(a, "equity", np.nan)
            == parse_float_field(b, "equity", np.nan)
            and parse_float_field(a, "cash", np.nan)
            == parse_float_field(b, "cash", np.nan)
            and parse_int_field(a, "qty", 0) == parse_int_field(b, "qty", 0)
        )

    # Require a small streak so isolated NOOP rows are preserved.
    min_stale_streak = 3
    if len(kept_rows) >= min_stale_streak:
        streak_len = 1
        for i in range(len(kept_rows) - 1, 0, -1):
            curr = kept_rows[i]
            prev = kept_rows[i - 1]
            if _is_flat_noop(curr) and _is_flat_noop(prev) and _same_portfolio_triplet(curr, prev):
                streak_len += 1
            else:
                break

        if streak_len >= min_stale_streak:
            cut_idx = len(kept_rows) - streak_len
            kept_rows = kept_rows[: cut_idx + 1]
            last_valid = kept_rows[-1] if kept_rows else None

    if len(kept_rows) != len(rows):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in kept_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    return last_valid


def read_portfolio_rows(path: str) -> Sequence[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_consecutive_flat_noops(rows: Sequence[Dict[str, str]]) -> int:
    n = 0
    for row in reversed(rows):
        action = (row.get("action") or "").strip().upper()
        if action != "NOOP":
            break
        if parse_int_field(row, "qty", 0) != 0:
            break
        if parse_float_field(row, "pos_state", 0.0) != 0.0:
            break
        if parse_float_field(row, "exposure", 0.0) != 0.0:
            break
        n += 1
    return n


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


def append_portfolio_row(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PORTFOLIO_COLUMNS)
        if not exists:
            w.writeheader()
        out = {k: row.get(k, "") for k in PORTFOLIO_COLUMNS}
        w.writerow(out)


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
    initial_portfolio_value: float,  # <-- NEW: explicit initial capital
    lookback_days: int = 400,
    interval: str = "d",
    max_gross_leverage: float = 1.0,
    strategy_kwargs: Optional[Dict[str, Any]] = None,
    stooq: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    One scheduler iteration (with backfill):
      1) Update price history via StooqDownloader
      2) Load last N days into DataFrame
      3) Load portfolio state from CSV (or initialize with initial_portfolio_value)
      4) Process each missing trading day in order
      5) Rebalance via place_order for each day
      6) Append one row per processed day to portfolio CSV
    """

    if strategy_kwargs is None:
        strategy_kwargs = {}
    if stooq is None:
        raise ValueError("Pass stooq=StooqDownloader() instance")

    # --------------------------------------------------
    # 1) Update price history
    # --------------------------------------------------
    stooq.update_prices(symbol, prices_csv_path, interval=interval)

    # --------------------------------------------------
    # 2) Load last N days into DataFrame
    # --------------------------------------------------
    raw = pd.read_csv(prices_csv_path)

    if "Date" not in raw.columns:
        raise ValueError(f"{prices_csv_path} missing 'Date' column")

    # Parse flexible date formats
    dt1 = pd.to_datetime(raw["Date"], errors="coerce", format="%d %b %Y")
    dt2 = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Date"] = dt1.fillna(dt2)

    raw = raw.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    for col in ["Open", "High", "Low", "Close"]:
        if col not in raw.columns:
            raise ValueError(f"Price CSV missing required column '{col}'")

    full_df = raw[["Open", "High", "Low", "Close"]].copy()

    if len(full_df) < 3:
        raise ValueError("Not enough history to run a single iteration")

    today = full_df.index[-1]
    today_str = today.strftime("%Y-%m-%d")

    # --------------------------------------------------
    # 3) Load portfolio state
    # --------------------------------------------------
    last_row = sanitize_portfolio_csv(portfolio_csv_path)
    prior_rows = read_portfolio_rows(portfolio_csv_path)
    noop_streak = count_consecutive_flat_noops(prior_rows)

    if last_row is None:
        # Explicit initialization
        p = PortfolioState(
            cash=float(initial_portfolio_value),
            qty=0,
            equity=float(initial_portfolio_value),
            pos_state=0.0,
            exposure=0.0,
            last_date=None,
        )
    else:
        p = portfolio_state_from_last_row(last_row)

    # Build set of dates to process (backfill missing trading days)
    if p.last_date is None:
        dates_to_process = [today]
    else:
        last_dt = pd.to_datetime(p.last_date, errors="coerce")
        if pd.isna(last_dt):
            dates_to_process = [today]
        else:
            dates_to_process = [d for d in full_df.index if d > last_dt]

    if not dates_to_process:
        return {
            "status": "skipped",
            "reason": "already_processed",
            "date": today_str,
            "symbol": symbol,
        }

    first_prev_equity = p.equity
    first_prev_qty = p.qty
    last_action = "NOOP"
    last_trade: Dict[str, Any] = {}

    for d in dates_to_process:
        day_str = d.strftime("%Y-%m-%d")
        day_df = full_df.loc[:d].tail(int(lookback_days)).copy()
        if len(day_df) < 3:
            continue

        open_px = float(day_df["Open"].iloc[-1])
        close_px = float(day_df["Close"].iloc[-1])

        res = run_single_iteration(
            day_df,
            prev_state=p.pos_state,
            prev_exposure=p.exposure,
            noop_streak=noop_streak,
            **strategy_kwargs,
        )

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
        p = PortfolioState(
            cash=new_cash,
            qty=new_qty,
            equity=new_equity,
            pos_state=res.state,
            exposure=res.exposure,
            last_date=day_str,
        )

        append_portfolio_row(
            portfolio_csv_path,
            {
                "date": day_str,
                "symbol": symbol,
                "equity": round(p.equity, 6),
                "cash": round(p.cash, 6),
                "qty": p.qty,
                "pos_state": round(p.pos_state, 6),
                "exposure": round(p.exposure, 6),
                "open": round(open_px, 6),
                "close": round(close_px, 6),
                "ret": round(float(res.ret), 12),
                "costs": round(float(res.costs), 12),
                "turnover": round(float(res.turnover), 12),
                "action": res.action,
                **trade_rec,
            },
        )

        if res.action == "NOOP" and p.qty == 0 and p.pos_state == 0.0 and p.exposure == 0.0:
            noop_streak += 1
        else:
            noop_streak = 0

        last_action = res.action
        last_trade = trade_rec

    return {
        "status": "ok",
        "date": p.last_date,
        "symbol": symbol,
        "action": last_action,
        "prev_equity": first_prev_equity,
        "new_equity": p.equity,
        "prev_qty": first_prev_qty,
        "new_qty": p.qty,
        "trade": last_trade,
        "processed_days": len(dates_to_process),
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
    plot_day_threshold: int = 20,
    equity_col: str = "equity",
    date_col: str = "date",
    symbol_col: Optional[str] = "symbol",
    symbol_filter: Optional[str] = None,
    periods_per_year: int = 252,
    subject_prefix: str = "Strategy Report",
) -> None:
    """
    Updated behavior:
      - If portfolio CSV has <2 usable rows, DO NOT raise.
      - Instead email a short message explaining the issue.
      - Otherwise behave as before (metrics only if <threshold, chart+metrics if enough days).
    """
    if not to_addresses:
        raise ValueError("to_addresses must be non-empty")

    ses = AmazonSES(
        region=region,
        access_key=access_key,
        secret_key=secret_key,
        from_address=from_address,
    )

    # --- Try read CSV ---
    try:
        df = pd.read_csv(portfolio_csv_path)
    except Exception as e:
        subject = f"{subject_prefix} — Unable to read portfolio CSV"
        html = f"""
        <html><body style="font-family: Arial, sans-serif;">
          <h3>{subject_prefix}</h3>
          <p>Could not read portfolio CSV at: <code>{portfolio_csv_path}</code></p>
          <p>Error: <code>{type(e).__name__}: {str(e)}</code></p>
        </body></html>
        """
        ses.send_html_email_many(to_addresses, subject, html)
        return

    # Optional symbol filter
    if symbol_filter and symbol_col and symbol_col in df.columns:
        df = df[df[symbol_col].astype(str).str.lower() == str(symbol_filter).lower()]

    # Not enough rows => EMAIL MESSAGE ONLY (no raise)
    if len(df) < 2:
        subject = f"{subject_prefix} — Not enough data yet"
        msg = (
            "Portfolio CSV does not have enough rows to report.\n"
            f"Need at least 2 rows (days) but found {len(df)}.\n"
            f"Path: {portfolio_csv_path}\n"
        )
        html = f"""
        <html><body style="font-family: Arial, sans-serif; line-height:1.35;">
          <h3>{subject_prefix}</h3>
          <pre style="background:#f6f6f6; padding:12px; border:1px solid #ddd;">{msg}</pre>
        </body></html>
        """
        ses.send_html_email_many(to_addresses, subject, html)
        return

    # ---- Continue normal behavior (existing logic) ----
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = (
        df.dropna(subset=[date_col, equity_col])
        .sort_values(date_col)
        .tail(int(lookback_days))
    )

    # If still not enough after cleaning => EMAIL MESSAGE ONLY
    if len(df) < 2:
        subject = f"{subject_prefix} — Not enough usable rows"
        msg = (
            "Portfolio CSV does not have enough usable rows to report after parsing dates/equity.\n"
            f"Need at least 2 usable rows but found {len(df)}.\n"
            f"Path: {portfolio_csv_path}\n"
            f"Required columns: {date_col!r}, {equity_col!r}\n"
        )
        html = f"""
        <html><body style="font-family: Arial, sans-serif; line-height:1.35;">
          <h3>{subject_prefix}</h3>
          <pre style="background:#f6f6f6; padding:12px; border:1px solid #ddd;">{msg}</pre>
        </body></html>
        """
        ses.send_html_email_many(to_addresses, subject, html)
        return

    # --- metrics ---
    metrics = compute_metrics_from_equity(
        df, equity_col=equity_col, date_col=date_col, periods_per_year=periods_per_year
    )

    def fmt_pct(x: float) -> str:
        return f"{x*100:,.2f}%"

    subject = f"{subject_prefix} — {metrics.end_date} (last {metrics.n_days} days)"

    if metrics.n_days < int(plot_day_threshold):
        html_content = f"""
        <html>
          <body style="font-family: Arial, sans-serif; line-height: 1.35;">
            <h2>{subject_prefix}: Snapshot</h2>
            <p><b>Date range:</b> {metrics.start_date} → {metrics.end_date} &nbsp; | &nbsp;
               <b>Days:</b> {metrics.n_days}</p>
            <p><b>Current equity:</b> {metrics.end_equity:,.2f}</p>
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
        image_cid = "equity_curve_png"
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
                 src="cid:{image_cid}" />
          </body>
        </html>
        """
        inline_images = [
            {
                "content_id": image_cid,
                "filename": "equity_curve.png",
                "data": base64.b64decode(png_b64),
                "subtype": "png",
            }
        ]
        ses.send_html_email_many_with_inline_images(
            to_addresses, subject, html_content, inline_images
        )
        return

    ses.send_html_email_many(to_addresses, subject, html_content)
