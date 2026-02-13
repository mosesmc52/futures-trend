import os

from helpers import run_daily_algo_once, send_equity_report_email
from services.stooq import StooqDownloader

stooq = StooqDownloader(page_delay=0.5, page_jitter=0.2)


out = run_daily_algo_once(
    symbol="cl.f",
    prices_csv_path="./data/CL.csv",
    portfolio_csv_path="./data/portfolio_cl.csv",
    lookback_days=400,
    stooq=stooq,
    strategy_kwargs={
        "breakout_n": 55,
        "exit_n": 20,
        "atr_n": 20,
        "ema_span": 200,
        "use_ema_filter": True,
        "target_annual_vol": 0.20,
        "max_leverage": 2.0,
        "cost_bps": 2.0,
        "slippage_bps": 1.0,
    },
)


print(out["status"], out["date"], out["action"], out["new_equity"])

send_equity_report_email(
    region=os.environ["AWS_REGION"],
    access_key=os.environ["AWS_ACCESS_KEY_ID"],
    secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    from_address="reports@yourdomain.com",
    to_addresses=["you@gmail.com"],
    portfolio_csv_path="./data/portfolio_cl.csv",
    lookback_days=60,
    plot_day_threshold=20,  # if <20 days, send metrics only
    subject_prefix="Oil Trend Paper Strategy",
)
