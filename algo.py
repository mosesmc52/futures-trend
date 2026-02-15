import os

from dotenv import find_dotenv, load_dotenv
from helpers import run_daily_algo_once, send_equity_report_email, str2bool
from services.stooq import StooqDownloader

load_dotenv(find_dotenv())

stooq = StooqDownloader(page_delay=0.5, page_jitter=0.2)


out = run_daily_algo_once(
    symbol="cl.f",
    prices_csv_path="data/CL.csv",
    portfolio_csv_path="data/portfolio_cl.csv",
    lookback_days=400,
    stooq=stooq,
    initial_portfolio_value=40_000,
    strategy_kwargs={
        "breakout_n": 20,
        "exit_n": 10,
        "atr_n": 20,
        "ema_span": 100,
        "use_ema_filter": True,
        "target_annual_vol": 0.20,
        "max_leverage": 2.0,
        "cost_bps": 0.0,
        "slippage_bps": 1.0,
    },
)

#
# print(out["status"], out["date"], out["action"], out["new_equity"])
if str2bool(os.getenv("EMAIL_POSITIONS", False)):
    send_equity_report_email(
        region=os.environ["AWS_SES_REGION_NAME"],
        access_key=os.environ["AWS_SES_ACCESS_KEY_ID"],
        secret_key=os.environ["AWS_SES_SECRET_ACCESS_KEY"],
        from_address=os.environ["FROM_ADDRESS"],
        to_addresses=os.environ["TO_ADDRESSES"].split(","),
        portfolio_csv_path="data/portfolio_cl.csv",
        lookback_days=60,
        plot_day_threshold=20,  # if <20 days, send metrics only
        subject_prefix="Oil Trend Paper Strategy",
    )
